#!/usr/bin/env python
"""Training script for sandhi-viccheda model."""

import argparse
from pathlib import Path

import torch

from sphota.data import make_dataloaders
from sphota.model import CharTokenizer, SandhiTransformer
from sphota.training import Trainer


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train sandhi-viccheda transformer model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="sandhi_dataset",
        help="Directory containing train/val/test TSV files",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="sandhi_dataset/vocab_devanagari.txt",
        help="Path to vocabulary file",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (32 recommended for Jetson with fp16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=4000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (fp16)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    print(f"Using device: {args.device}")

    print("Loading tokenizer...")
    tokenizer = CharTokenizer(str(vocab_path))
    print(f"Vocab size: {tokenizer.vocab_size}")

    print("Creating dataloaders...")
    train_loader, val_loader, _ = make_dataloaders(
        str(data_dir),
        tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("Creating model...")
    model = SandhiTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        max_seq_len=args.max_len,
        pad_idx=tokenizer.PAD_IDX,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.2f}M")

    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        accum_steps=args.accum_steps,
        label_smoothing=0.1,
        use_amp=not args.no_amp,
    )

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Using AMP: {not args.no_amp}")
    trainer.fit(args.epochs, resume_from=args.resume)

    print("Training complete!")
    best_checkpoint = Path(args.checkpoint_dir) / "checkpoint_best.pt"
    print(f"Best checkpoint saved to: {best_checkpoint}")


if __name__ == "__main__":
    main()

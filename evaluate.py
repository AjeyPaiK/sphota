#!/usr/bin/env python
"""Evaluation script for sandhi-viccheda model."""

import argparse
from pathlib import Path

import torch

from sphota.data import SandhiDataset
from sphota.model import CharTokenizer, SandhiTransformer


def compute_metrics(predictions: list, references: list) -> dict:
    """Compute CER, WER, and BLEU-like metrics.

    Args:
        predictions: List of predicted strings
        references: List of reference strings

    Returns:
        Dict with metrics
    """
    total_chars = 0
    correct_chars = 0
    total_words = 0
    correct_words = 0

    for pred, ref in zip(predictions, references):
        pred_chars = list(pred.replace(" ", ""))
        ref_chars = list(ref.replace(" ", ""))

        for p, r in zip(pred_chars, ref_chars):
            total_chars += 1
            if p == r:
                correct_chars += 1

        pred_words = pred.split()
        ref_words = ref.split()

        for p, r in zip(pred_words, ref_words):
            total_words += 1
            if p == r:
                correct_words += 1

    cer = 1.0 - (correct_chars / max(total_chars, 1))
    wer = 1.0 - (correct_words / max(total_words, 1))

    return {
        "CER": cer,
        "WER": wer,
        "char_accuracy": correct_chars / max(total_chars, 1),
        "word_accuracy": correct_words / max(total_words, 1),
    }


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate sandhi-viccheda model on test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="sandhi_dataset",
        help="Directory containing test TSV file",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default="sandhi_dataset/vocab_devanagari.txt",
        help="Path to vocabulary file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam search width",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate (None = all)",
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    print(f"Using device: {args.device}")

    print("Loading tokenizer...")
    tokenizer = CharTokenizer(str(vocab_path))

    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model = SandhiTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        max_seq_len=args.max_len,
        pad_idx=tokenizer.PAD_IDX,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(args.device)
    model.eval()

    print("Loading test dataset...")
    test_path = data_dir / "test_devanagari.tsv"
    if not test_path.exists():
        test_path = data_dir / "val_devanagari.tsv"
        print(f"Test file not found, using validation set: {test_path}")

    test_dataset = SandhiDataset(str(test_path), tokenizer, max_len=args.max_len)

    if args.num_samples:
        test_dataset.pairs = test_dataset.pairs[: args.num_samples]

    print(f"Evaluating on {len(test_dataset)} samples...")

    predictions = []
    references = []

    with torch.no_grad():
        for idx, (sandhi, vicchheda) in enumerate(test_dataset.pairs):
            src_ids, _ = tokenizer.encode_pair(sandhi, vicchheda, args.max_len)
            src = torch.tensor([src_ids], dtype=torch.long, device=args.device)

            tgt_ids = model.generate(src, max_len=args.max_len, beam_width=args.beam_width)
            pred_text = tokenizer.decode(tgt_ids[0].tolist(), skip_special_tokens=True)
            ref_text = vicchheda

            predictions.append(pred_text)
            references.append(ref_text)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(test_dataset)} samples")

    metrics = compute_metrics(predictions, references)

    print("\n=== Evaluation Results ===")
    print(f"Character Error Rate (CER): {metrics['CER']:.4f}")
    print(f"Word Error Rate (WER):       {metrics['WER']:.4f}")
    print(f"Character Accuracy:         {metrics['char_accuracy']:.4f}")
    print(f"Word Accuracy:              {metrics['word_accuracy']:.4f}")

    print("\n=== Sample Predictions ===")
    for i in range(min(5, len(predictions))):
        print(f"Input:      {test_dataset.pairs[i][0]}")
        print(f"Reference:  {references[i]}")
        print(f"Predicted:  {predictions[i]}")
        print()


if __name__ == "__main__":
    main()

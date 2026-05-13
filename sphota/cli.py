"""Command-line interface for sphota tools."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sphota.data import SandhiDataset, make_dataloaders
from sphota.model import CharTokenizer, SandhiTransformer
from sphota.sandhi import build_dataset, split_dataset
from sphota.training import Trainer

console = Console()

try:
    import plotext as plt
    HAS_PLOTEXT = True
except ImportError:
    HAS_PLOTEXT = False


def cmd_build(args):
    """Build sandhi dataset from CoNLL-U files."""
    if not os.path.exists(args.conllu_dir):
        console.print("[red]❌ Error: CoNLL-U directory not found:[/red] " + args.conllu_dir)
        console.print("\n[yellow]This is required to build the sandhi dataset.[/yellow]")
        console.print("\n[cyan]To set up the Sanskrit repository:[/cyan]")
        console.print("  [dim]git clone https://github.com/OliverHellwig/sanskrit.git sanskrit[/dim]")
        console.print("\n[cyan]After cloning, the data should be at:[/cyan]")
        console.print("  [dim]sanskrit/dcs/data/conllu/files/[/dim]")
        console.print("\n[yellow]You can also specify a custom path with --conllu-dir[/yellow]")
        sys.exit(1)

    console.print("[bold cyan]Building sandhi dataset...[/bold cyan]")
    build_dataset(
        conllu_dir=args.conllu_dir,
        out_dir=args.out_dir,
        max_files=args.max_files,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    if not args.skip_split:
        console.print("[bold cyan]Splitting into train/val/test...[/bold cyan]")
        split_dataset(
            out_dir=args.out_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    console.print("[bold green]✓ Done![/bold green]")


def cmd_train(args):
    """Train sandhi-viccheda model."""
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    console.print(f"[cyan]Using device:[/cyan] [bold]{args.device}[/bold]")

    console.print("[cyan]Loading tokenizer...[/cyan]")
    tokenizer = CharTokenizer(str(vocab_path))
    console.print(f"[cyan]Vocab size:[/cyan] [bold]{tokenizer.vocab_size}[/bold]")

    console.print("[cyan]Creating dataloaders...[/cyan]")
    train_loader, val_loader, _ = make_dataloaders(
        str(data_dir),
        tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
    )
    console.print(f"[cyan]Train batches:[/cyan] [bold]{len(train_loader)}[/bold], [cyan]Val batches:[/cyan] [bold]{len(val_loader)}[/bold]")

    console.print("[cyan]Creating model...[/cyan]")
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
    console.print(f"[cyan]Model parameters:[/cyan] [bold]{total_params / 1e6:.2f}M[/bold]")

    console.print("[cyan]Creating trainer...[/cyan]")
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

    console.print(f"[bold cyan]Starting training for {args.epochs} epochs...[/bold cyan]")
    console.print(f"[cyan]Using AMP:[/cyan] [bold]{not args.no_amp}[/bold]")
    trainer.fit(args.epochs, resume_from=args.resume)

    console.print("[bold green]✓ Training complete![/bold green]")
    best_checkpoint = Path(args.checkpoint_dir) / "checkpoint_best.pt"
    console.print(f"[cyan]Best checkpoint saved to:[/cyan] [bold]{best_checkpoint}[/bold]")


def cmd_evaluate(args):
    """Evaluate model on test set."""

    def compute_metrics(predictions, references):
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

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    console.print(f"[cyan]Using device:[/cyan] [bold]{args.device}[/bold]")

    console.print("[cyan]Loading tokenizer...[/cyan]")
    tokenizer = CharTokenizer(str(vocab_path))

    console.print("[cyan]Loading model...[/cyan]")
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

    console.print("[cyan]Loading test dataset...[/cyan]")
    test_path = data_dir / "test_devanagari.tsv"
    if not test_path.exists():
        test_path = data_dir / "val_devanagari.tsv"
        console.print(f"[yellow]Test file not found, using validation set:[/yellow] {test_path}")

    test_dataset = SandhiDataset(str(test_path), tokenizer, max_len=args.max_len)

    if args.num_samples:
        test_dataset.pairs = test_dataset.pairs[: args.num_samples]

    console.print(f"[bold cyan]Evaluating on {len(test_dataset)} samples...[/bold cyan]")

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
                console.print(f"[dim]Processed {idx + 1}/{len(test_dataset)} samples[/dim]")

    metrics = compute_metrics(predictions, references)

    console.print("\n[bold cyan]=== Evaluation Results ===[/bold cyan]")
    console.print(f"[cyan]Character Error Rate (CER):[/cyan] [bold]{metrics['CER']:.4f}[/bold]")
    console.print(f"[cyan]Word Error Rate (WER):[/cyan] [bold]{metrics['WER']:.4f}[/bold]")
    console.print(f"[cyan]Character Accuracy:[/cyan] [bold]{metrics['char_accuracy']:.4f}[/bold]")
    console.print(f"[cyan]Word Accuracy:[/cyan] [bold]{metrics['word_accuracy']:.4f}[/bold]")

    console.print("\n[bold cyan]=== Sample Predictions ===[/bold cyan]")
    for i in range(min(5, len(predictions))):
        console.print(f"[yellow]Input:[/yellow] {test_dataset.pairs[i][0]}")
        console.print(f"[green]Reference:[/green] {references[i]}")
        console.print(f"[cyan]Predicted:[/cyan] {predictions[i]}")
        console.print()


def cmd_plot(args):
    """Plot training metrics (loss curves, accuracy, etc.)."""
    if not HAS_PLOTEXT:
        console.print(
            "[red]❌ Error: plotext not installed[/red]\n"
            "[cyan]Install with:[/cyan] [bold]pip install -e '.[viz]'[/bold]"
        )
        sys.exit(1)

    metrics_path = Path(args.checkpoint_dir) / "metrics.json"
    if not metrics_path.exists():
        console.print(
            f"[red]❌ Error: Metrics file not found:[/red] {metrics_path}\n"
            "[yellow]Run training first to generate metrics[/yellow]"
        )
        sys.exit(1)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    epochs = metrics["epochs"]
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]
    val_acc = metrics["val_acc"]

    if not epochs:
        console.print("[yellow]No metrics to plot yet[/yellow]")
        return

    # Create plots
    console.print("\n[bold cyan]Loss Curves[/bold cyan]\n")

    plt.clear_figure()
    plt.plot(epochs, train_loss, label="Train Loss", color="cyan", marker="o")
    plt.plot(epochs, val_loss, label="Val Loss", color="magenta", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.grid(True)
    plt.show()

    console.print("\n[bold cyan]Validation Accuracy[/bold cyan]\n")

    plt.clear_figure()
    plt.plot(epochs, val_acc, label="Val Accuracy", color="green", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True)
    plt.show()

    # Print summary table
    console.print("\n[bold cyan]Metrics Summary[/bold cyan]\n")

    table = Table(title="Training Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Epoch", style="cyan")
    table.add_column("Train Loss", style="yellow")
    table.add_column("Val Loss", style="magenta")
    table.add_column("Val Acc", style="green")

    for i, epoch in enumerate(epochs):
        table.add_row(
            str(epoch),
            f"{train_loss[i]:.4f}",
            f"{val_loss[i]:.4f}",
            f"{val_acc[i]:.4f}",
        )

    console.print(table)

    # Stats
    best_epoch = val_loss.index(min(val_loss))
    console.print(
        Panel(
            f"[cyan]Best Validation Loss:[/cyan] [bold]{min(val_loss):.4f}[/bold] at [bold]Epoch {epochs[best_epoch]}[/bold]\n"
            f"[cyan]Final Validation Accuracy:[/cyan] [bold]{val_acc[-1]:.4f}[/bold]",
            border_style="green",
            expand=False,
        )
    )


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Sphota: Sanskrit sandhi-viccheda tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sphota build-data --conllu-dir sanskrit/dcs/data/conllu/files
  sphota train --epochs 20 --batch-size 32 --device cuda
  sphota plot --checkpoint-dir checkpoints
  sphota evaluate --checkpoint checkpoints/checkpoint_best.pt
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # build-data command
    build_parser = subparsers.add_parser("build-data", help="Build sandhi dataset from CoNLL-U")
    build_parser.add_argument(
        "--conllu-dir",
        default="sanskrit/dcs/data/conllu/files",
        help="Directory containing .conllu files",
    )
    build_parser.add_argument(
        "--out-dir",
        default="sandhi_dataset",
        help="Output directory for datasets",
    )
    build_parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process",
    )
    build_parser.add_argument(
        "--min-tokens",
        type=int,
        default=2,
        help="Minimum tokens per sentence",
    )
    build_parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens per sentence",
    )
    build_parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Proportion for training set",
    )
    build_parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Proportion for validation set",
    )
    build_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    build_parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip train/val/test splitting",
    )
    build_parser.set_defaults(func=cmd_build)

    # train command
    train_parser = subparsers.add_parser("train", help="Train sandhi-viccheda model")
    train_parser.add_argument(
        "--data-dir",
        type=str,
        default="sandhi_dataset",
        help="Directory containing train/val/test TSV files",
    )
    train_parser.add_argument(
        "--vocab-path",
        type=str,
        default="sandhi_dataset/vocab_devanagari.txt",
        help="Path to vocabulary file",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    train_parser.add_argument(
        "--warmup-steps",
        type=int,
        default=4000,
        help="Number of warmup steps",
    )
    train_parser.add_argument(
        "--accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    train_parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    train_parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    train_parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (fp16)",
    )
    train_parser.set_defaults(func=cmd_train)

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on test set")
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    eval_parser.add_argument(
        "--data-dir",
        type=str,
        default="sandhi_dataset",
        help="Directory containing test TSV file",
    )
    eval_parser.add_argument(
        "--vocab-path",
        type=str,
        default="sandhi_dataset/vocab_devanagari.txt",
        help="Path to vocabulary file",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    eval_parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Maximum sequence length",
    )
    eval_parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam search width",
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    eval_parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of test samples to evaluate",
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # plot command
    plot_parser = subparsers.add_parser("plot", help="Plot training metrics (loss curves, accuracy)")
    plot_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing metrics.json",
    )
    plot_parser.set_defaults(func=cmd_plot)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

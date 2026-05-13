"""Command-line interface for sphota tools."""

import argparse
import os
import sys
from .sandhi import build_dataset, split_dataset


def _check_conllu_dir(conllu_dir):
    """Check if the CoNLL-U directory exists and provide helpful error message."""
    if not os.path.exists(conllu_dir):
        print(f"\n❌ Error: CoNLL-U directory not found: {conllu_dir}\n", file=sys.stderr)
        print("This is required to build the sandhi dataset.", file=sys.stderr)
        print("\nTo set up the Sanskrit repository:\n", file=sys.stderr)
        print("  git clone https://github.com/OliverHellwig/sanskrit.git sanskrit\n", file=sys.stderr)
        print("After cloning, the data should be at:", file=sys.stderr)
        print("  sanskrit/dcs/data/conllu/files/\n", file=sys.stderr)
        print("You can also specify a custom path with --conllu-dir\n", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Build Sanskrit sandhi datasets from CoNLL-U files"
    )
    parser.add_argument(
        "--conllu-dir",
        default="sanskrit/dcs/data/conllu/files",
        help="Directory containing .conllu files",
    )
    parser.add_argument(
        "--out-dir",
        default="sandhi_dataset",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=2,
        help="Minimum tokens per sentence",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens per sentence",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Proportion for training set",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Proportion for validation set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip train/val/test splitting",
    )

    args = parser.parse_args()

    _check_conllu_dir(args.conllu_dir)

    print("Building sandhi dataset...")
    build_dataset(
        conllu_dir=args.conllu_dir,
        out_dir=args.out_dir,
        max_files=args.max_files,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    if not args.skip_split:
        print("\nSplitting into train/val/test...")
        split_dataset(
            out_dir=args.out_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()

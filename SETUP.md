# Setup Guide

This guide walks you through setting up the Sphota environment for working with Sanskrit language data and building sandhi datasets.

## Prerequisites

- Python 3.8 or higher
- Git
- ~2-3 GB of disk space (for Sanskrit repo + datasets)

## Step 1: Install Sphota Package

Clone and install the sphota package:

```bash
git clone <sphota-repo-url>
cd sphota
pip install -e ".[transliteration]"
```

The `[transliteration]` extra includes the indic-transliteration library needed for script conversion.

## Step 2: Clone Sanskrit Repository (REQUIRED for dataset building)

The Sanskrit repository is **not** included as a dependency - it must be cloned separately:

```bash
# Clone into the same parent directory
cd ..
git clone https://github.com/OliverHellwig/sanskrit.git
```

Your directory structure should now look like:

```
parent-directory/
├── sphota/
│   ├── sphota/
│   ├── sandhi_dataset/
│   ├── pyproject.toml
│   └── README.md
└── sanskrit/
    ├── dcs/
    │   └── data/
    │       └── conllu/
    │           └── files/     # CoNLL-U files (this is what we need)
    ├── papers/
    └── ...
```

## Step 3: Build Sandhi Dataset

Once both repositories are set up, you can build the dataset:

```bash
cd sphota
sphota-build-sandhi
```

Or with custom paths:

```bash
sphota-build-sandhi --conllu-dir ../sanskrit/dcs/data/conllu/files --out-dir my_datasets
```

## Troubleshooting

### Missing Sanskrit Repository

If you see an error about missing CoNLL-U files:

```
❌ Error: CoNLL-U directory not found: sanskrit/dcs/data/conllu/files
```

Make sure you've cloned the Sanskrit repository and it's in the correct location:

```bash
# From the sphota directory
git clone https://github.com/OliverHellwig/sanskrit.git ../sanskrit
```

### Import Errors

If you get `ModuleNotFoundError` for transliteration:

```bash
pip install -e ".[transliteration]"
```

### Memory Issues

If you run out of memory during dataset building, process fewer files:

```bash
sphota-build-sandhi --max-files 100
```

## Full Development Setup

For development with all dependencies (ML frameworks, testing, etc.):

```bash
pip install -e ".[all]"
```

This includes:
- All transliteration and data processing libraries
- PyTorch and TensorFlow
- Testing and linting tools
- All paper research code dependencies

## Next Steps

Once your dataset is built, check out the sandhi_dataset/ directory for:

- `train_iast.tsv`, `val_iast.tsv`, `test_iast.tsv` - IAST transliteration splits
- `train_devanagari.tsv`, etc. - Devanagari script splits
- `sandhi_pairs.json` - Full metadata
- `vocab_devanagari.txt` - Character vocabulary

See README.md for usage examples.

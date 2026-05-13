# Sphota

Sanskrit linguistic tools and datasets, including sandhi processing and analysis.

## Installation

Install from source:

```bash
pip install -e .
```

Or with transliteration support:

```bash
pip install -e ".[transliteration]"
```

### Required: Sanskrit Repository

**⚠️ IMPORTANT:** To build sandhi datasets, you **must** have the Sanskrit repository cloned separately. This is not automatically installed.

```bash
git clone https://github.com/OliverHellwig/sanskrit.git sanskrit
```

The Sanskrit repository is kept separate because:
- It's large (includes papers, lexical data, and corpus files)
- It's managed independently and updated frequently
- Not all users need dataset building functionality
- It maintains clear separation of concerns

**Expected structure after cloning:**
```
your-project/
├── sphota/              (this package)
├── sanskrit/            (Sanskrit repo - must be cloned separately)
│   └── dcs/data/conllu/files/  (dataset files needed for building)
```

## Usage

### Command-line

Build a sandhi dataset from CoNLL-U files:

```bash
sphota-build-sandhi --conllu-dir path/to/conllu/files --out-dir output/path
```

Options:
- `--conllu-dir`: Directory containing .conllu files (default: `sanskrit/dcs/data/conllu/files`)
- `--out-dir`: Output directory (default: `sandhi_dataset`)
- `--max-files`: Maximum number of files to process
- `--min-tokens`: Minimum tokens per sentence (default: 2)
- `--max-tokens`: Maximum tokens per sentence (default: 50)
- `--train-ratio`: Proportion for training set (default: 0.9)
- `--val-ratio`: Proportion for validation set (default: 0.05)
- `--seed`: Random seed for splitting (default: 42)
- `--skip-split`: Skip train/val/test splitting

### Python API

```python
from sphota.sandhi import build_dataset, split_dataset

# Build the dataset
build_dataset(
    conllu_dir="path/to/conllu",
    out_dir="sandhi_dataset",
    min_tokens=2,
    max_tokens=50
)

# Split into train/val/test
split_dataset(out_dir="sandhi_dataset")
```

## Output

The dataset builder generates:
- `sandhi_pairs_iast.tsv` - Sanskrit word pairs in IAST romanization
- `sandhi_pairs_devanagari.tsv` - Word pairs in Devanagari script (requires transliteration support)
- `sandhi_pairs.json` - Full sentence metadata in JSON format
- `vocab_devanagari.txt` - Character vocabulary (with transliteration support)
- `train_*.tsv`, `val_*.tsv`, `test_*.tsv` - Dataset splits

## Working with Sanskrit Data

The `build_sandhi_dataset.py` script processes Sanskrit text in CoNLL-U format. The default path expects:

```
sanskrit/dcs/data/conllu/files/
```

You can override this with the `--conllu-dir` parameter.

## Development

Install with development dependencies:

```bash
pip install -e ".[dev,transliteration]"
```

Run tests:

```bash
pytest
```

## License

MIT License - see LICENSE file for details.

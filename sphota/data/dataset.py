"""PyTorch dataset for sandhi-viccheda pairs."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from sphota.model.tokenizer import CharTokenizer


class SandhiDataset(Dataset):
    """Dataset for sandhi-viccheda pairs."""

    def __init__(
        self,
        tsv_path: str,
        tokenizer: CharTokenizer,
        max_len: int = 256,
    ):
        """Load dataset from TSV file.

        Args:
            tsv_path: Path to TSV file with columns (sandhi, vicchheda)
            tokenizer: CharTokenizer instance
            max_len: Maximum sequence length
        """
        self.tsv_path = Path(tsv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = []

        if not self.tsv_path.exists():
            raise FileNotFoundError(f"TSV file not found: {tsv_path}")

        with open(tsv_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                sandhi, vicchheda = parts
                self.pairs.append((sandhi, vicchheda))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sandhi, vicchheda = self.pairs[idx]
        src_ids, tgt_ids = self.tokenizer.encode_pair(sandhi, vicchheda, self.max_len)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Collate batch with dynamic padding.

    Args:
        batch: List of (src, tgt) tuples

    Returns:
        Dict with 'src' and 'tgt' tensors, padded to max length in batch
    """
    src_list = [item[0] for item in batch]
    tgt_list = [item[1] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_list, batch_first=True, padding_value=0
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_list, batch_first=True, padding_value=0
    )

    return {
        "src": src_padded,
        "tgt": tgt_padded,
    }


def make_dataloaders(
    data_dir: str,
    tokenizer: CharTokenizer,
    batch_size: int = 32,
    num_workers: int = 2,
    max_len: int = 256,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create train/val/test dataloaders.

    Args:
        data_dir: Directory containing train/val/test TSV files
        tokenizer: CharTokenizer instance
        batch_size: Batch size
        num_workers: Number of worker processes
        max_len: Maximum sequence length

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)

    train_dataset = SandhiDataset(
        data_dir / "train_devanagari.tsv",
        tokenizer,
        max_len=max_len,
    )
    val_dataset = SandhiDataset(
        data_dir / "val_devanagari.tsv",
        tokenizer,
        max_len=max_len,
    )

    test_path = data_dir / "test_devanagari.tsv"
    test_dataset = SandhiDataset(test_path, tokenizer, max_len=max_len) if test_path.exists() else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader

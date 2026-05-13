"""Character-level tokenizer for Sanskrit Devanagari."""

from pathlib import Path
from typing import List, Tuple


class CharTokenizer:
    """Character-level tokenizer using predefined vocabulary."""

    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, vocab_path: str):
        """Load vocabulary from file.

        Args:
            vocab_path: Path to vocab file (one token per line, 1-indexed)
        """
        vocab_path = Path(vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        self.vocab = []
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                self.vocab.append(line.rstrip("\n"))

        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: If True, prepend <bos> and append <eos>

        Returns:
            List of token IDs
        """
        ids = []
        if add_special_tokens:
            ids.append(self.BOS_IDX)

        for char in text:
            idx = self.char2idx.get(char, self.UNK_IDX)
            ids.append(idx)

        if add_special_tokens:
            ids.append(self.EOS_IDX)

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs
            skip_special_tokens: If True, skip <pad>, <bos>, <eos> tokens

        Returns:
            Decoded text
        """
        chars = []
        for idx in ids:
            if idx not in self.idx2char:
                continue
            char = self.idx2char[idx]
            if skip_special_tokens and idx in (self.PAD_IDX, self.BOS_IDX, self.EOS_IDX):
                continue
            chars.append(char)
        return "".join(chars)

    def encode_pair(
        self, src: str, tgt: str, max_len: int = 256
    ) -> Tuple[List[int], List[int]]:
        """Encode source and target with padding and special tokens.

        Args:
            src: Source text (sandhi form)
            tgt: Target text (viccheda form)
            max_len: Maximum sequence length (applies to both src and tgt)

        Returns:
            Tuple of (src_ids, tgt_ids), both truncated/padded to max_len
        """
        src_ids = self.encode(src[:max_len], add_special_tokens=True)
        tgt_ids = self.encode(tgt[:max_len], add_special_tokens=True)

        src_ids = src_ids[:max_len]
        tgt_ids = tgt_ids[:max_len]

        return src_ids, tgt_ids

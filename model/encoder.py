"""
Character-level encoder for Sanskrit verb forms.

Takes SLP1-encoded input (one character per phoneme) and produces
contextualized representations using BiLSTM.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SLP1Vocab:
    """SLP1 character vocabulary."""

    # SLP1 encoding: IAST → SLP1 for phase 1 roots
    # Reference: https://en.wikipedia.org/wiki/SLP1
    CONSONANTS = "kKgGNcCjJYwWqQRlMnp"
    VOWELS = "aAiIuUfFxXeEoO"
    SPECIAL = "MH~"  # Anusvara, Visarga, nukta
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self):
        # Build vocabulary
        chars = list(self.VOWELS + self.CONSONANTS + self.SPECIAL)
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.char_to_id[self.PAD_TOKEN] = len(self.char_to_id)
        self.char_to_id[self.UNK_TOKEN] = len(self.char_to_id)

        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.pad_id = self.char_to_id[self.PAD_TOKEN]
        self.unk_id = self.char_to_id[self.UNK_TOKEN]

    def encode(self, text: str) -> list:
        """Convert string to list of token IDs."""
        return [self.char_to_id.get(c, self.unk_id) for c in text]

    def decode(self, ids: list) -> str:
        """Convert list of token IDs back to string."""
        return "".join(self.id_to_char.get(i, "?") for i in ids)

    def __len__(self):
        return len(self.char_to_id)


class CharacterEncoder(nn.Module):
    """
    BiLSTM encoder over SLP1 characters.

    Input: (batch_size, seq_len)
    Output: (batch_size, seq_len, hidden_size)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # BiLSTM output: (batch_size, seq_len, 2*hidden_dim)
        self.output_dim = 2 * hidden_dim

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) binary mask

        Returns:
            (contextual_representations, (h_n, c_n))
            - contextual_representations: (batch_size, seq_len, output_dim)
            - (h_n, c_n): final hidden/cell states
        """
        embeddings = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)

        # Pack padded sequence if mask provided
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            embeddings = torch.nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False
            )

        output, (h_n, c_n) = self.bilstm(embeddings)  # output: (batch, seq_len, 2*hidden_dim)

        # Unpack if we packed
        if attention_mask is not None:
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, (h_n, c_n)


class ContextAggregator(nn.Module):
    """
    Aggregates contextual representations into a single vector.
    Uses attention-weighted pooling or max-pooling over the sequence.
    """

    def __init__(self, input_dim: int, method: str = "max"):
        super().__init__()
        self.input_dim = input_dim
        self.method = method

        if method == "attention":
            self.attention = nn.Linear(input_dim, 1)
        elif method != "max":
            raise ValueError(f"Unknown aggregation method: {method}")

    def forward(
        self, representations: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            representations: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) binary mask

        Returns:
            aggregated: (batch_size, input_dim)
        """
        if self.method == "max":
            # Max pooling with masking
            if mask is not None:
                representations = representations.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
            aggregated, _ = torch.max(representations, dim=1)
            return aggregated

        elif self.method == "attention":
            # Attention-weighted pooling
            scores = self.attention(representations)  # (batch, seq_len, 1)
            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
            weights = torch.softmax(scores, dim=1)
            aggregated = (representations * weights).sum(dim=1)
            return aggregated


if __name__ == "__main__":
    # Test
    vocab = SLP1Vocab()
    print(f"Vocab size: {len(vocab)}")
    print(f"Sample encoding 'Bavati': {vocab.encode('Bavati')}")

    encoder = CharacterEncoder(
        vocab_size=len(vocab),
        embedding_dim=32,
        hidden_dim=64,
        num_layers=1,
    )

    # Dummy batch
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, len(vocab), (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    output, (h_n, c_n) = encoder(input_ids, mask)
    print(f"Encoder output shape: {output.shape}")  # (4, 10, 128)

    aggregator = ContextAggregator(encoder.output_dim, method="max")
    pooled = aggregator(output, mask)
    print(f"Pooled output shape: {pooled.shape}")  # (4, 128)

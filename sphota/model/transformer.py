"""Transformer-based encoder-decoder for sandhi-viccheda."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SandhiTransformer(nn.Module):
    """Encoder-decoder transformer for sandhi-viccheda."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ffn: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
        )

        self.output_proj = nn.Linear(d_model, vocab_size)

    def _create_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create padding mask (True where padding, False where real tokens)."""
        return tokens == self.pad_idx

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            src: Source tokens (batch, src_len)
            tgt: Target tokens (batch, tgt_len)
            src_key_padding_mask: Mask for source padding
            tgt_key_padding_mask: Mask for target padding
            memory_key_padding_mask: Mask for memory (encoder output) padding

        Returns:
            Logits (batch, tgt_len, vocab_size)
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = self._create_padding_mask(src)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._create_padding_mask(tgt)
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)

        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device, dtype=torch.bool
        )

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=None,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.output_proj(output)
        return logits

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode source sequence."""
        src_key_padding_mask = self._create_padding_mask(src)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    @torch.no_grad()
    def generate(
        self, src: torch.Tensor, max_len: int = 256, beam_width: int = 5
    ) -> torch.Tensor:
        """Generate target sequence using beam search.

        Args:
            src: Source tokens (batch, src_len)
            max_len: Maximum target length
            beam_width: Beam search width

        Returns:
            Generated target tokens (batch, max_len)
        """
        batch_size = src.size(0)
        device = src.device

        src_key_padding_mask = self._create_padding_mask(src)
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        tgt = torch.full((batch_size, 1), fill_value=1, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoding(tgt_emb)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1), device=device, dtype=torch.bool
            )

            output = self.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_key_padding_mask,
            )

            logits = self.output_proj(output[:, -1:, :])
            next_tokens = torch.argmax(logits, dim=-1)
            tgt = torch.cat([tgt, next_tokens], dim=1)

            if (next_tokens == 2).all():
                break

        return tgt

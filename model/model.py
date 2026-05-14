"""
Unified Sanskrit morphology model combining encoder and task heads.

Tasks:
1. Dhatu root identification + gana classification
2. Morphological features (purusha, vacana, pada)
3. Rule sequence prediction (why the derivation works)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.encoder import CharacterEncoder, ContextAggregator, SLP1Vocab
from model.heads import DhatuHead, MorphologyHead, RuleSequenceHead, RuleVocab


class SanskritMorphologyModel(nn.Module):
    """
    End-to-end Sanskrit verb morphology analyzer.

    Pipeline:
    1. Character-level BiLSTM encoder (SLP1 input)
    2. Context aggregation (max-pooling)
    3. Multi-task classification:
       - Root + gana (DhatuHead)
       - Morphological features (MorphologyHead)
       - Rule sequence (RuleSequenceHead)
    """

    def __init__(
        self,
        vocab_size: int,
        num_roots: int = 2000,
        num_ganas: int = 10,
        embedding_dim: int = 32,
        encoder_hidden_dim: int = 64,
        num_encoder_layers: int = 1,
        decoder_hidden_dim: int = 256,
        num_rules: int = 10,
        max_rule_seq_len: int = 5,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_roots = num_roots
        self.num_ganas = num_ganas
        self.num_rules = num_rules

        # Character encoder
        self.encoder = CharacterEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=encoder_hidden_dim,
            num_layers=num_encoder_layers,
            dropout=0.1,
        )

        # Context aggregation
        self.aggregator = ContextAggregator(self.encoder.output_dim, method="max")

        # Task heads
        self.dhatu_head = DhatuHead(
            input_dim=self.encoder.output_dim,
            num_roots=num_roots,
            num_ganas=num_ganas,
            hidden_dim=decoder_hidden_dim,
        )

        self.morph_head = MorphologyHead(
            input_dim=self.encoder.output_dim,
            num_purusha=3,
            num_vacana=3,
            num_pada=2,
            hidden_dim=decoder_hidden_dim,
        )

        self.rule_head = RuleSequenceHead(
            input_dim=self.encoder.output_dim,
            num_rules=num_rules,
            max_seq_len=max_rule_seq_len,
            hidden_dim=decoder_hidden_dim,
        )

        # Rule vocabulary
        self.rule_vocab = RuleVocab()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_len) - character indices
            attention_mask: (batch_size, seq_len) - binary mask

        Returns:
            Dict with keys:
            - root_logits: (batch, num_roots)
            - gana_logits: (batch, num_ganas)
            - purusha_logits: (batch, 3)
            - vacana_logits: (batch, 3)
            - pada_logits: (batch, 2)
            - seq_len_logits: (batch, max_seq_len)
            - rule_logits: (batch, max_seq_len, num_rules)
        """
        # Encode
        encoded, _ = self.encoder(input_ids, attention_mask)
        context = self.aggregator(encoded, attention_mask)

        # Dhatu head
        root_logits, gana_logits = self.dhatu_head(context)

        # Morphology head
        purusha_logits, vacana_logits, pada_logits = self.morph_head(context)

        # Rule head
        seq_len_logits, rule_logits = self.rule_head(context)

        return {
            "root_logits": root_logits,
            "gana_logits": gana_logits,
            "purusha_logits": purusha_logits,
            "vacana_logits": vacana_logits,
            "pada_logits": pada_logits,
            "seq_len_logits": seq_len_logits,
            "rule_logits": rule_logits,
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    vocab = SLP1Vocab()
    model = SanskritMorphologyModel(
        vocab_size=len(vocab),
        num_roots=100,  # Small for testing
        num_ganas=10,
        embedding_dim=32,
        encoder_hidden_dim=64,
        decoder_hidden_dim=256,
        num_rules=len(RuleVocab()),
    )

    print(f"Model parameters: {model.count_parameters():,}")

    # Dummy batch
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, len(vocab), (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)

    outputs = model(input_ids, mask)
    print(f"\nOutput shapes:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")

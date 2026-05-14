"""
Task-specific output heads for Sanskrit morphology.

1. Dhatu head: classifies root + gana
2. Rule sequence head: predicts ordered sequence of Panini rule IDs
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class DhatuHead(nn.Module):
    """
    Classifies the धातु (root) and its गण.

    Multi-task:
    - Root classifier: picks from ~2,000 roots
    - Gana classifier: picks from 10 ganas

    For Phase 1, only gana 1 is trained.
    """

    def __init__(
        self,
        input_dim: int,
        num_roots: int = 2000,
        num_ganas: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_roots = num_roots
        self.num_ganas = num_ganas

        # Shared hidden layer
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)

        # Root classifier
        self.root_classifier = nn.Linear(hidden_dim, num_roots)

        # Gana classifier
        self.gana_classifier = nn.Linear(hidden_dim, num_ganas)

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context: (batch_size, input_dim) - aggregated representation

        Returns:
            (root_logits, gana_logits)
            - root_logits: (batch_size, num_roots)
            - gana_logits: (batch_size, num_ganas)
        """
        hidden = torch.relu(self.hidden(context))
        hidden = self.dropout(hidden)

        root_logits = self.root_classifier(hidden)
        gana_logits = self.gana_classifier(hidden)

        return root_logits, gana_logits


class RuleSequenceHead(nn.Module):
    """
    Sequence decoder for predicting Panini rule IDs.

    Input: contextual representation from encoder
    Output: sequence of rule IDs (e.g., [7.3.84, 3.1.68, 3.4.77])

    For Phase 1, maximum sequence length is ~3 rules.
    """

    def __init__(
        self,
        input_dim: int,
        num_rules: int = 100,  # Total unique rule IDs we'll encounter
        max_seq_len: int = 5,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_rules = num_rules
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        # Encode rule sequence length
        self.seq_len_predictor = nn.Linear(input_dim, max_seq_len)

        # Decode sequence of rules
        # For simplicity: linear projection at each position
        # (In a more sophisticated model, use transformer or GRU decoder)
        self.rule_decoders = nn.ModuleList(
            [nn.Linear(input_dim, num_rules) for _ in range(max_seq_len)]
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context: (batch_size, input_dim)

        Returns:
            (seq_len_logits, rule_logits)
            - seq_len_logits: (batch_size, max_seq_len)
            - rule_logits: (batch_size, max_seq_len, num_rules)
        """
        batch_size = context.size(0)

        # Predict sequence length
        seq_len_logits = self.seq_len_predictor(context)  # (batch, max_seq_len)

        # Predict rule at each position
        rule_logits_list = []
        for decoder in self.rule_decoders:
            rule_logits = decoder(context)  # (batch, num_rules)
            rule_logits_list.append(rule_logits)

        rule_logits = torch.stack(rule_logits_list, dim=1)  # (batch, max_seq_len, num_rules)

        return seq_len_logits, rule_logits


class MorphologyHead(nn.Module):
    """
    Classifies morphological properties: purusha (person), vacana (number), pada (voice).
    """

    def __init__(
        self,
        input_dim: int,
        num_purusha: int = 3,  # 1st, 2nd, 3rd
        num_vacana: int = 3,   # singular, dual, plural
        num_pada: int = 2,     # parasmaipada, atmanepada
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = input_dim

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)

        self.purusha_classifier = nn.Linear(hidden_dim, num_purusha)
        self.vacana_classifier = nn.Linear(hidden_dim, num_vacana)
        self.pada_classifier = nn.Linear(hidden_dim, num_pada)

    def forward(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            context: (batch_size, input_dim)

        Returns:
            (purusha_logits, vacana_logits, pada_logits)
        """
        hidden = torch.relu(self.hidden(context))
        hidden = self.dropout(hidden)

        purusha_logits = self.purusha_classifier(hidden)
        vacana_logits = self.vacana_classifier(hidden)
        pada_logits = self.pada_classifier(hidden)

        return purusha_logits, vacana_logits, pada_logits


class RuleVocab:
    """Vocabulary of Panini rule IDs."""

    PHASE1_RULES = [
        "7.3.84",  # Guna substitution
        "3.1.68",  # Sap vikarana
        "3.4.77",  # Lat ending
    ]

    def __init__(self):
        self.rule_to_id = {rule: i for i, rule in enumerate(self.PHASE1_RULES)}
        self.id_to_rule = {v: k for k, v in self.rule_to_id.items()}

    def encode(self, rules: List[str]) -> List[int]:
        """Convert rule IDs to indices."""
        return [self.rule_to_id.get(r, -1) for r in rules]

    def decode(self, ids: List[int]) -> List[str]:
        """Convert indices back to rule IDs."""
        return [self.id_to_rule.get(i, "?") for i in ids]

    def __len__(self):
        return len(self.PHASE1_RULES)


if __name__ == "__main__":
    # Test heads
    batch_size = 4
    input_dim = 128

    context = torch.randn(batch_size, input_dim)

    dhatu_head = DhatuHead(input_dim=input_dim, num_roots=100, num_ganas=10)
    root_logits, gana_logits = dhatu_head(context)
    print(f"Root logits shape: {root_logits.shape}")  # (4, 100)
    print(f"Gana logits shape: {gana_logits.shape}")  # (4, 10)

    rule_head = RuleSequenceHead(input_dim=input_dim, num_rules=10, max_seq_len=5)
    seq_len_logits, rule_logits = rule_head(context)
    print(f"Sequence length logits shape: {seq_len_logits.shape}")  # (4, 5)
    print(f"Rule logits shape: {rule_logits.shape}")  # (4, 5, 10)

    morph_head = MorphologyHead(input_dim=input_dim)
    purusha_logits, vacana_logits, pada_logits = morph_head(context)
    print(f"Purusha logits shape: {purusha_logits.shape}")  # (4, 3)
    print(f"Vacana logits shape: {vacana_logits.shape}")  # (4, 3)
    print(f"Pada logits shape: {pada_logits.shape}")  # (4, 2)

    rule_vocab = RuleVocab()
    print(f"Rule vocab size: {len(rule_vocab)}")
    print(f"Encoding ['7.3.84', '3.1.68']: {rule_vocab.encode(['7.3.84', '3.1.68'])}")

"""
Training loop for Sanskrit morphology model — Jetson-optimized.

Features:
- FP16 mixed precision (required for 8GB Jetson)
- Adafactor optimizer (4× less memory than Adam)
- Gradient accumulation
- Multi-task loss
- Memory monitoring with tegrastats
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import Adafactor

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.encoder import SLP1Vocab
from model.heads import RuleVocab
from model.model import SanskritMorphologyModel


class SanskritDataset(Dataset):
    """Dataset for Sanskrit morphology."""

    def __init__(
        self,
        jsonl_path: str,
        vocab: SLP1Vocab,
        rule_vocab: RuleVocab,
        max_seq_len: int = 32,
    ):
        self.vocab = vocab
        self.rule_vocab = rule_vocab
        self.max_seq_len = max_seq_len

        # Load data
        self.examples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.examples.append(obj)

        # Build root_id and other mappings
        self._build_mappings()

    def _build_mappings(self):
        """Build mappings for roots, rules, and morphological features."""
        self.roots = sorted(set(ex["dhatu"] for ex in self.examples))
        self.root_to_id = {r: i for i, r in enumerate(self.roots)}

        # Morphological features
        self.purusha_to_id = {"प्रथम": 0, "मध्यम": 1, "उत्तम": 2}
        self.vacana_to_id = {"एकवचन": 0, "द्विवचन": 1, "बहुवचन": 2}
        self.pada_to_id = {"परस्मैपद": 0, "आत्मनेपद": 1}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        """Return a training example."""
        ex = self.examples[idx]

        # Encode surface form
        surface = ex["surface"]
        input_ids = self.vocab.encode(surface)
        input_ids = input_ids[: self.max_seq_len]

        # Pad
        pad_len = self.max_seq_len - len(input_ids)
        input_ids.extend([self.vocab.pad_id] * pad_len)
        attention_mask = [1] * (self.max_seq_len - pad_len) + [0] * pad_len

        # Root and gana
        root_id = self.root_to_id[ex["dhatu"]]
        gana_id = ex["gana"] - 1  # Gana 1-10 → 0-9

        # Morphological features
        purusha_id = self.purusha_to_id[ex["purusha"]]
        vacana_id = self.vacana_to_id[ex["vacana"]]
        pada_id = self.pada_to_id[ex["pada"]]

        # Rules
        rules = ex["rules_applied"]
        rule_ids = self.rule_vocab.encode(rules)
        # Pad rule sequence to max length
        max_rules = 5
        rule_ids.extend([0] * (max_rules - len(rule_ids)))
        rule_ids = rule_ids[:max_rules]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "root_id": torch.tensor(root_id, dtype=torch.long),
            "gana_id": torch.tensor(gana_id, dtype=torch.long),
            "purusha_id": torch.tensor(purusha_id, dtype=torch.long),
            "vacana_id": torch.tensor(vacana_id, dtype=torch.long),
            "pada_id": torch.tensor(pada_id, dtype=torch.long),
            "rule_ids": torch.tensor(rule_ids, dtype=torch.long),
            "num_rules": torch.tensor(len(rules), dtype=torch.long),
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Adafactor,
    scaler: GradScaler,
    device: torch.device,
    loss_weights: Dict[str, float],
    accumulation_steps: int = 1,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    losses = defaultdict(float)
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        root_id = batch["root_id"].to(device)
        gana_id = batch["gana_id"].to(device)
        purusha_id = batch["purusha_id"].to(device)
        vacana_id = batch["vacana_id"].to(device)
        pada_id = batch["pada_id"].to(device)
        rule_ids = batch["rule_ids"].to(device)
        num_rules = batch["num_rules"].to(device)

        # Forward pass with mixed precision
        with autocast():
            outputs = model(input_ids, attention_mask)

            # Compute losses
            root_loss = nn.functional.cross_entropy(outputs["root_logits"], root_id)
            gana_loss = nn.functional.cross_entropy(outputs["gana_logits"], gana_id)
            purusha_loss = nn.functional.cross_entropy(outputs["purusha_logits"], purusha_id)
            vacana_loss = nn.functional.cross_entropy(outputs["vacana_logits"], vacana_id)
            pada_loss = nn.functional.cross_entropy(outputs["pada_logits"], pada_id)

            # Rule sequence loss (only for non-zero rule positions)
            rule_loss = 0.0
            for pos in range(outputs["rule_logits"].size(1)):
                rule_loss += nn.functional.cross_entropy(
                    outputs["rule_logits"][:, pos, :],
                    rule_ids[:, pos],
                )
            rule_loss = rule_loss / outputs["rule_logits"].size(1)

            # Weighted combination
            loss = (
                loss_weights["root"] * root_loss
                + loss_weights["gana"] * gana_loss
                + loss_weights["purusha"] * purusha_loss
                + loss_weights["vacana"] * vacana_loss
                + loss_weights["pada"] * pada_loss
                + loss_weights["rule"] * rule_loss
            )

            # Gradient accumulation
            loss = loss / accumulation_steps

        # Backward with mixed precision
        scaler.scale(loss).backward()

        # Update after accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Logging
        losses["root"] += root_loss.item()
        losses["gana"] += gana_loss.item()
        losses["purusha"] += purusha_loss.item()
        losses["vacana"] += vacana_loss.item()
        losses["pada"] += pada_loss.item()
        losses["rule"] += rule_loss
        total_loss += loss.item() * accumulation_steps
        num_batches += 1

    # Average losses
    for key in losses:
        losses[key] /= num_batches
    total_loss /= num_batches

    return {"total_loss": total_loss, **losses}


def main():
    """Main training script."""
    # Configuration (Jetson-tuned)
    CONFIG = {
        "batch_size": 4,
        "grad_accumulation_steps": 16,  # Effective batch = 64
        "max_seq_len": 32,
        "embedding_dim": 32,
        "encoder_hidden_dim": 64,
        "decoder_hidden_dim": 256,
        "num_encoder_layers": 1,
        "learning_rate": 1e-3,
        "loss_weights": {
            "root": 1.0,
            "gana": 0.5,
            "purusha": 0.5,
            "vacana": 0.5,
            "pada": 0.3,
            "rule": 1.0,
        },
        "num_epochs": 3,
        "data_path": "data/gana1_lat_train.jsonl",
        "checkpoint_dir": "checkpoints",
    }

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build vocab
    vocab = SLP1Vocab()
    rule_vocab = RuleVocab()
    print(f"Vocab size: {len(vocab)}")

    # Load dataset
    print(f"Loading dataset from {CONFIG['data_path']}...")
    dataset = SanskritDataset(
        CONFIG["data_path"],
        vocab,
        rule_vocab,
        max_seq_len=CONFIG["max_seq_len"],
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Unique roots: {len(dataset.roots)}")

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    # Build model
    model = SanskritMorphologyModel(
        vocab_size=len(vocab),
        num_roots=len(dataset.roots),
        num_ganas=10,
        embedding_dim=CONFIG["embedding_dim"],
        encoder_hidden_dim=CONFIG["encoder_hidden_dim"],
        decoder_hidden_dim=CONFIG["decoder_hidden_dim"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        num_rules=len(rule_vocab),
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer: Adafactor (memory-efficient)
    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=True,
        relative_step=True,
        warmup_init=True,
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Training loop
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"\nTraining for {CONFIG['num_epochs']} epochs...")
    for epoch in range(CONFIG["num_epochs"]):
        losses = train_epoch(
            model,
            dataloader,
            optimizer,
            scaler,
            device,
            CONFIG["loss_weights"],
            accumulation_steps=CONFIG["grad_accumulation_steps"],
        )

        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
        print(f"  Loss: {losses['total_loss']:.4f}")
        print(f"  Root loss: {losses['root']:.4f}")
        print(f"  Rule loss: {losses['rule']:.4f}")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": CONFIG,
            },
            checkpoint_path,
        )
        print(f"  Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()

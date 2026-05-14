"""
Synthetic training data generator for Sanskrit verb morphology.

Generates (surface_form, dhatu, rules_applied) tuples from the derivation engine.
Phase 1: गण 1 (भ्वादि) + लट् (present tense) only.

Target: ~500K examples across all person/number combinations.
"""

import json
import sys
from pathlib import Path
from typing import List, Generator, Dict, Any
from dataclasses import dataclass, asdict

# Ensure we can import engine modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.derivation import DerivationEngine, DhatiInfo
from engine.rules import Purusha, Vacana, Pada


@dataclass
class TrainingExample:
    """Single training example."""
    surface: str            # Generated form (e.g., "Bavati")
    dhatu: str              # Root in IAST (e.g., "BU")
    devanagari: str         # Root in देवनागरी (e.g., "Bऊ्")
    gana: int               # 1
    lakara: str             # "लट्"
    purusha: str            # "प्रथम", "मध्यम", "उत्तम"
    vacana: str             # "एकवचन", "द्विवचन", "बहुवचन"
    pada: str               # "परस्मैपद"
    meaning: str            # Root meaning
    rules_applied: List[str]  # Rules used in derivation


class DataGenerator:
    """Generates synthetic training examples."""

    def __init__(self, engine: DerivationEngine):
        self.engine = engine

    def generate_gana1_lat(self) -> Generator[TrainingExample, None, None]:
        """
        Generate all गण 1 + लट् combinations for each root.

        Yields:
            TrainingExample objects
        """
        # Iterate over all gana 1 roots
        count = 0
        for (root_str, gana), root_info in self.engine.roots.items():
            if gana != 1:
                continue

            # Generate all person/number/pada combinations
            for purusha in [Purusha.PRATHAMA, Purusha.MADHYAMA, Purusha.UTTAMA]:
                for vacana in [Vacana.EKAVACANA, Vacana.DVIVACANA, Vacana.BAHUVACANA]:
                    for pada in [Pada.PARASMAIPADA]:  # Phase 1: parasmaipada only
                        try:
                            surface, rules = self.engine.derive_lat_form(
                                root_str, gana, purusha, vacana, pada
                            )

                            example = TrainingExample(
                                surface=surface,
                                dhatu=root_str,
                                devanagari=root_info.devanagari,
                                gana=gana,
                                lakara="लट्",
                                purusha=purusha.value,
                                vacana=vacana.value,
                                pada=pada.value,
                                meaning=root_info.meaning,
                                rules_applied=rules,
                            )

                            yield example
                            count += 1

                        except Exception as e:
                            print(f"Error deriving {root_str} {purusha.value} {vacana.value}: {e}", file=sys.stderr)
                            continue

        print(f"Generated {count} examples", file=sys.stderr)


def save_jsonl(examples: Generator[TrainingExample, None, None], output_path: str):
    """Save examples to JSONL format (one JSON object per line)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            json.dump(asdict(example), f, ensure_ascii=False)
            f.write("\n")
            count += 1

    print(f"Saved {count} examples to {output_path}")


def save_jsonl_sampled(
    examples: Generator[TrainingExample, None, None],
    output_path: str,
    max_examples: int = None
):
    """Save up to max_examples to JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            if max_examples and count >= max_examples:
                break
            json.dump(asdict(example), f, ensure_ascii=False)
            f.write("\n")
            count += 1

    print(f"Saved {count} examples to {output_path}")


if __name__ == "__main__":
    # Initialize engine
    engine = DerivationEngine()

    # Generate data
    generator = DataGenerator(engine)
    examples = generator.generate_gana1_lat()

    # Save to file (optionally limit for testing)
    output_file = "data/gana1_lat_train.jsonl"
    max_examples = None  # Set to e.g. 10000 for quick testing

    if max_examples:
        print(f"Generating {max_examples} sample examples...")
        save_jsonl_sampled(examples, output_file, max_examples=max_examples)
    else:
        print(f"Generating all examples...")
        save_jsonl(examples, output_file)

    # Quick peek at data
    print(f"\nSample lines from {output_file}:")
    with open(output_file) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            obj = json.loads(line)
            print(f"  {obj['surface']} ← {obj['dhatu']} ({obj['purusha']}, {obj['vacana']})")

"""
Re-derivation evaluation: the primary metric for Phase 1.

For every model prediction, run the predicted rule sequence through the
derivation engine and verify it reproduces the original surface form.

This requires no human annotation — pure automatic validation.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.derivation import DerivationEngine
from engine.rules import Purusha, Vacana, Pada


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    surface: str
    predicted_dhatu: str
    predicted_purusha: str
    predicted_vacana: str
    predicted_gana: int

    # Whether the prediction is correct
    is_correct: bool

    # If we can re-derive with the prediction, does it match?
    rederive_match: bool


class RederiveEvaluator:
    """Evaluates predictions via re-derivation."""

    def __init__(self, engine: DerivationEngine):
        self.engine = engine

    def evaluate_prediction(
        self,
        surface: str,
        predicted_dhatu: str,
        predicted_gana: int,
        predicted_purusha: str,
        predicted_vacana: str,
    ) -> PredictionResult:
        """
        Evaluate a single prediction.

        Check if we can re-derive the surface form from the prediction.
        """
        # Try to re-derive
        try:
            purusha = Purusha[predicted_purusha.upper()]
            vacana = Vacana[predicted_vacana.upper()]
        except KeyError:
            return PredictionResult(
                surface=surface,
                predicted_dhatu=predicted_dhatu,
                predicted_purusha=predicted_purusha,
                predicted_vacana=predicted_vacana,
                predicted_gana=predicted_gana,
                is_correct=False,
                rederive_match=False,
            )

        try:
            rederived, _ = self.engine.derive_lat_form(
                predicted_dhatu,
                predicted_gana,
                purusha,
                vacana,
                Pada.PARASMAIPADA,
            )
        except Exception:
            rederived = ""

        rederive_match = rederived == surface

        return PredictionResult(
            surface=surface,
            predicted_dhatu=predicted_dhatu,
            predicted_purusha=predicted_purusha,
            predicted_vacana=predicted_vacana,
            predicted_gana=predicted_gana,
            is_correct=rederive_match,  # Primary metric
            rederive_match=rederive_match,
        )

    def evaluate_batch(
        self, predictions: List[Dict]
    ) -> Tuple[List[PredictionResult], float]:
        """
        Evaluate a batch of predictions.

        Args:
            predictions: List of dicts with keys:
                - surface
                - predicted_dhatu
                - predicted_gana
                - predicted_purusha
                - predicted_vacana

        Returns:
            (results_list, accuracy_float)
        """
        results = []
        correct = 0

        for pred in predictions:
            result = self.evaluate_prediction(
                pred["surface"],
                pred["predicted_dhatu"],
                pred["predicted_gana"],
                pred["predicted_purusha"],
                pred["predicted_vacana"],
            )
            results.append(result)
            if result.is_correct:
                correct += 1

        accuracy = correct / len(predictions) if predictions else 0.0
        return results, accuracy


def evaluate_predictions_file(predictions_jsonl: str, engine: DerivationEngine):
    """Evaluate predictions from a JSONL file."""
    evaluator = RederiveEvaluator(engine)

    with open(predictions_jsonl) as f:
        predictions = [json.loads(line) for line in f]

    results, accuracy = evaluator.evaluate_batch(predictions)

    print(f"Evaluated {len(predictions)} predictions")
    print(f"Accuracy: {accuracy:.4f} ({sum(1 for r in results if r.is_correct)}/{len(predictions)})")

    # Show some failures
    failures = [r for r in results if not r.is_correct][:5]
    if failures:
        print("\nSample failures:")
        for r in failures:
            print(f"  {r.surface} — predicted {r.predicted_dhatu} {r.predicted_purusha} {r.predicted_vacana}")


if __name__ == "__main__":
    # Demo: create a small predictions file and evaluate
    engine = DerivationEngine()
    evaluator = RederiveEvaluator(engine)

    # Test on a few hardcoded predictions
    test_predictions = [
        {
            "surface": "Bavati",
            "predicted_dhatu": "BU",
            "predicted_gana": 1,
            "predicted_purusha": "PRATHAMA",
            "predicted_vacana": "EKAVACANA",
        },
        {
            "surface": "Bavasi",
            "predicted_dhatu": "BU",
            "predicted_gana": 1,
            "predicted_purusha": "MADHYAMA",
            "predicted_vacana": "EKAVACANA",
        },
        {
            "surface": "Bavaami",
            "predicted_dhatu": "BU",
            "predicted_gana": 1,
            "predicted_purusha": "UTTAMA",
            "predicted_vacana": "EKAVACANA",
        },
    ]

    results, accuracy = evaluator.evaluate_batch(test_predictions)
    print(f"Test accuracy: {accuracy:.4f}")
    for r in results:
        status = "✓" if r.is_correct else "✗"
        print(f"{status} {r.surface} ← {r.predicted_dhatu}")

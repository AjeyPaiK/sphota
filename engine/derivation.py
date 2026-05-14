"""
DerivationEngine: Orchestrates Pāṇinian morphological derivation.

Pipeline:
1. Load root from dhatupatha.json
2. Apply vikarana (gana-specific stem suffix)
3. Apply lakara endings (tense/person/number)
4. Apply guna/vriddhi and phonological rules
5. Validate final form

Phase 1: गण 1 (भ्वादि) + लट् (present tense) only
"""

import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, asdict

from engine.rules import Purusha, Vacana, Pada, Lakara, MorphState, derive_gana1_lat
from engine.vikarana import get_vikarana_suffix


@dataclass
class DerivationStep:
    """Single step in a derivation."""
    step_num: int
    rule_id: str
    description: str
    before: str
    after: str


@dataclass
class DhatiInfo:
    """Information about a root from dhatupatha."""
    root: str          # IAST encoding
    devanagari: str
    meaning: str
    gana: int
    pada: str          # "parasmaipada" or "atmanepada"


class DerivationEngine:
    """Main morphological derivation engine."""

    def __init__(self, dhatupatha_path: str = "engine/dhatupatha.json"):
        """Initialize engine with dhatupatha data."""
        self.dhatupatha_path = Path(dhatupatha_path)
        self.roots: Dict[Tuple[str, int], DhatiInfo] = {}
        self._load_dhatupatha()

    def _load_dhatupatha(self):
        """Load root inventory from dhatupatha.json."""
        if not self.dhatupatha_path.exists():
            raise FileNotFoundError(f"dhatupatha.json not found at {self.dhatupatha_path}")

        with open(self.dhatupatha_path, encoding="utf-8") as f:
            data = json.load(f)

        # Extract gana data
        ganas = data.get("ganas", {})
        for gana_key, gana_data in ganas.items():
            gana_num = gana_data.get("gana_number")
            roots = gana_data.get("roots", [])

            for root_entry in roots:
                root_str = root_entry.get("root", "")
                key = (root_str, gana_num)
                self.roots[key] = DhatiInfo(
                    root=root_str,
                    devanagari=root_entry.get("devanagari", ""),
                    meaning=root_entry.get("meaning", ""),
                    gana=gana_num,
                    pada=root_entry.get("pada", "parasmaipada"),
                )

        print(f"Loaded {len(self.roots)} root entries from dhatupatha")

    def lookup_root(self, root: str, gana: int) -> Optional[DhatiInfo]:
        """Look up a root in the dhatupatha."""
        return self.roots.get((root, gana))

    def derive_lat_form(
        self,
        root: str,
        gana: int,
        purusha: Purusha,
        vacana: Vacana,
        pada: Pada = Pada.PARASMAIPADA,
    ) -> Tuple[str, List[str]]:
        """
        Derive a लट् (present) form for gana 1 only.

        Args:
            root: Root in IAST (e.g., "BU")
            gana: Gana number (1-10, but only 1 in Phase 1)
            purusha: Person (1st/2nd/3rd)
            vacana: Number (singular/dual/plural)
            pada: Voice (parasmaipada/atmanepada)

        Returns:
            (surface_form, rules_applied_list)
        """
        if gana != 1:
            raise NotImplementedError("Only गण 1 (भ्वादि) implemented in Phase 1")

        # Verify root exists
        root_info = self.lookup_root(root, gana)
        if not root_info:
            raise ValueError(f"Root '{root}' gana {gana} not found in dhatupatha")

        # Use the unified derive_gana1_lat function
        surface, rules = derive_gana1_lat(root, purusha, vacana)

        return surface, rules

    def rederive_check(
        self,
        root: str,
        gana: int,
        purusha: Purusha,
        vacana: Vacana,
        expected_surface: str,
    ) -> bool:
        """
        Verification: generate form and check against expected.

        This is the primary evaluation metric (no human annotation needed).
        """
        generated, _ = self.derive_lat_form(root, gana, purusha, vacana)
        return generated == expected_surface


def test_derivation_engine():
    """Simple test of engine on a known root."""
    engine = DerivationEngine()

    # Test root: BU (भू्) gana 1
    # Expected forms:
    # bhavati (3rd sg): BU + a (vikarana) + ti (ending) = Bavati
    # bhavasi (2nd sg): BU + a + si = Bavasi
    # bhavami (1st sg): BU + a + mi = Bavami

    test_cases = [
        ("BU", 1, Purusha.PRATHAMA, Vacana.EKAVACANA, "Bavati"),
        ("BU", 1, Purusha.MADHYAMA, Vacana.EKAVACANA, "Bavasi"),
        ("BU", 1, Purusha.UTTAMA, Vacana.EKAVACANA, "Bavami"),
    ]

    for root, gana, purusha, vacana, expected in test_cases:
        try:
            surface, rules = engine.derive_lat_form(root, gana, purusha, vacana)
            match = "✓" if surface == expected else "✗"
            print(f"{match} {root} {purusha.value} {vacana.value}: {surface} (expected {expected})")
            print(f"  Rules: {rules}")
        except Exception as e:
            print(f"✗ {root}: {e}")


if __name__ == "__main__":
    test_derivation_engine()

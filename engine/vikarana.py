"""
Vikarana (stem suffix) rules for Sanskrit verbs (Rule 3.1.1–3.1.72).

Vikarana is the thematic/stem suffix added to the root before person/number endings.
Each gana (class) has its characteristic vikarana.

Reference: Panini 3.1.1 "भ्वादयो धातुलङ्गमः शप्" — roots of bhvadi etc. take शप्
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class GanaType(Enum):
    """The ten ganas (verb classes) of Sanskrit."""
    BHVADI = 1      # भ्वादि: शप् (अ)
    ADADI = 2       # अदादि: none (athematic)
    JUHOTY_ADI = 3  # जुहोत्यादि: none + reduplication
    DIVADI = 4      # दिवादि: श्यन् (य)
    SVADI = 5       # स्वादि: श्नु (नु)
    TUDADI = 6      # तुदादि: श (अ, short)
    RUDHADI = 7     # रुधादि: श्नम् (न, infix)
    TANADI = 8      # तनादि: उ
    KRYADI = 9      # क्र्यादि: श्ना (ना/नी)
    CUADI = 10      # चुरादि: णिच् (अय)


@dataclass
class VikaranaInfo:
    """Information about a gana's vikarana."""
    gana: GanaType
    name: str           # e.g., "Bhvadi"
    vikarana: str       # Technical name, e.g., "शप्"
    vikarana_mark: str  # The actual suffix added, e.g., "a" for भ्वादि
    example_root: str   # Example root in IAST
    example_form: str   # Example 3rd sg present in IAST


VIKARANAS = {
    1: VikaranaInfo(
        gana=GanaType.BHVADI,
        name="Bhvadi",
        vikarana="शप्",
        vikarana_mark="a",
        example_root="BU",
        example_form="Bavati",  # भवति
    ),
    2: VikaranaInfo(
        gana=GanaType.ADADI,
        name="Adadi",
        vikarana="(none)",
        vikarana_mark="",
        example_root="ad",
        example_form="atti",  # अत्ति
    ),
    3: VikaranaInfo(
        gana=GanaType.JUHOTY_ADI,
        name="Juhoty-adi",
        vikarana="(none + reduplication)",
        vikarana_mark="",
        example_root="hu",
        example_form="juhoti",  # जुहोति
    ),
    4: VikaranaInfo(
        gana=GanaType.DIVADI,
        name="Divadi",
        vikarana="श्यन्",
        vikarana_mark="ya",
        example_root="div",
        example_form="dIvyati",  # दीव्यति
    ),
    5: VikaranaInfo(
        gana=GanaType.SVADI,
        name="Svadi",
        vikarana="श्नु",
        vikarana_mark="nu",
        example_root="su",
        example_form="sunoti",  # सुनोति
    ),
    6: VikaranaInfo(
        gana=GanaType.TUDADI,
        name="Tudadi",
        vikarana="श",
        vikarana_mark="a",
        example_root="tud",
        example_form="tudati",  # तुदति
    ),
    7: VikaranaInfo(
        gana=GanaType.RUDHADI,
        name="Rudhadi",
        vikarana="श्नम्",
        vikarana_mark="na",
        example_root="rudh",
        example_form="runddhi",  # रुणद्धि
    ),
    8: VikaranaInfo(
        gana=GanaType.TANADI,
        name="Tanadi",
        vikarana="उ",
        vikarana_mark="u",
        example_root="tan",
        example_form="tanuti",  # तनुति
    ),
    9: VikaranaInfo(
        gana=GanaType.KRYADI,
        name="Kry-adi",
        vikarana="श्ना",
        vikarana_mark="na",
        example_root="kri",
        example_form="krInAti",  # क्रीणाति
    ),
    10: VikaranaInfo(
        gana=GanaType.CUADI,
        name="Cur-adi (Causative)",
        vikarana="णिच्",
        vikarana_mark="aya",
        example_root="cur",
        example_form="corayati",  # चोरयति
    ),
}


def get_vikarana_suffix(gana: int) -> str:
    """Return the vikarana suffix for a given gana.

    Args:
        gana: Gana number (1-10)

    Returns:
        Vikarana suffix string (e.g., "a" for gana 1)
    """
    if gana not in VIKARANAS:
        raise ValueError(f"Invalid gana: {gana}. Must be 1-10.")
    return VIKARANAS[gana].vikarana_mark


def get_gana_info(gana: int) -> VikaranaInfo:
    """Retrieve full gana information."""
    if gana not in VIKARANAS:
        raise ValueError(f"Invalid gana: {gana}. Must be 1-10.")
    return VIKARANAS[gana]


# Phase 1: Only gana 1 is fully implemented
PHASE_1_GANAS = [1]


if __name__ == "__main__":
    for gana_num in range(1, 11):
        info = get_gana_info(gana_num)
        print(f"Gana {gana_num} ({info.name}): {info.vikarana} → '{info.vikarana_mark}'")
        print(f"  Example: {info.example_root} → {info.example_form}")

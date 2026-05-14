"""
Pāṇinian rule implementations for Sanskrit morphology.
Focus on गण 1 (भ्वादि) + लट् (present tense) for Phase 1.

Rule format: each rule is a callable that takes a morphological state
and modifies it in place or returns a new state.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class Lakara(Enum):
    LAT = "लट्"       # present


class Purusha(Enum):
    PRATHAMA = "प्रथम"   # 3rd person
    MADHYAMA = "मध्यम"   # 2nd person
    UTTAMA = "उत्तम"     # 1st person


class Vacana(Enum):
    EKAVACANA = "एकवचन"   # singular
    DVIVACANA = "द्विवचन"  # dual
    BAHUVACANA = "बहुवचन"  # plural


class Pada(Enum):
    PARASMAIPADA = "परस्मैपद"
    ATMANEPADA = "आत्मनेपद"


@dataclass
class MorphState:
    """Internal state during derivation."""
    dhatu: str              # root, e.g., "BU" in IAST or "BUY" in SLP1
    gana: int               # 1-10
    lakara: Lakara
    purusha: Purusha
    vacana: Vacana
    pada: Pada = Pada.PARASMAIPADA

    # Intermediate forms
    stem: str = ""          # After vikarana applied
    surface: str = ""       # Final form after all rules

    # Metadata for derivation trace
    rules_applied: List[str] = None

    def __post_init__(self):
        if self.rules_applied is None:
            self.rules_applied = []


# =============================================================================
# VIKARANA (Stem Suffix Rules) — 3.1.x
# =============================================================================

def apply_sap_vikara(state: MorphState) -> str:
    """Rule 3.1.68: शप् (अ) vikarana for gana 1 (Bhvadi).

    भ्वादयो धातुलङ्गमः शप् (3.1.68)
    Root + शप् + अ (dummy अ, replaced by ending)

    Returns the stem after vikarana.
    """
    # शप् is purely indicative; the actual suffix is अ
    stem = state.dhatu + "a"
    state.rules_applied.append("3.1.68")
    return stem


# =============================================================================
# LAKARA ENDINGS — 3.4.x (present/लट्)
# =============================================================================

def get_lat_ending(purusha: Purusha, vacana: Vacana, pada: Pada) -> str:
    """Rule 3.4.77–3.4.82: लट् (present) endings.

    Parasmaipada (active) endings:
    1st: -mi, -vas, -mas
    2nd: -si, -tha, -tha
    3rd: -ti, -tas, -anti

    Returns the suffix to append to stem.
    """
    if pada == Pada.PARASMAIPADA:
        if purusha == Purusha.UTTAMA:
            if vacana == Vacana.EKAVACANA:
                return "mi"
            elif vacana == Vacana.DVIVACANA:
                return "vas"
            else:  # BAHUVACANA
                return "mas"

        elif purusha == Purusha.MADHYAMA:
            if vacana == Vacana.EKAVACANA:
                return "si"
            elif vacana == Vacana.DVIVACANA:
                return "tha"
            else:  # BAHUVACANA
                return "tha"

        elif purusha == Purusha.PRATHAMA:
            if vacana == Vacana.EKAVACANA:
                return "ti"
            elif vacana == Vacana.DVIVACANA:
                return "tas"
            else:  # BAHUVACANA
                return "anti"

    else:  # ATMANEPADA (middle) — not in Phase 1
        raise NotImplementedError("Ātmanepada not yet implemented")

    return ""


def apply_lat_endings(state: MorphState, stem: str) -> str:
    """Attach लट् ending to stem."""
    ending = get_lat_ending(state.purusha, state.vacana, state.pada)
    state.rules_applied.append("3.4.77")  # Generic reference
    return stem + ending


# =============================================================================
# GUNA/VRIDDHI SUBSTITUTIONS — 7.3.82–7.3.86
# =============================================================================

def guna_map() -> dict:
    """Simple guna substitutions (IAST).

    Rule 7.3.82–7.3.84:
    अ/ह → ए
    इ/उ → यव् (semi-vowels)
    ऋ/लृ → अर्/अल्
    """
    return {
        "a": "e",
        "i": "iy",
        "u": "uv",
        "f": "ar",  # IAST ऋ
        "x": "al",  # IAST लृ
    }


def vriddhi_map() -> dict:
    """Simple vriddhi substitutions."""
    return {
        "a": "A",  # IAST आ
        "i": "Iy",
        "u": "Uv",
        "f": "Ar",
        "x": "Al",
    }


def apply_guna(state: MorphState, stem: str, rule_id: str = "7.3.84") -> str:
    """Apply guna grade substitution.

    Rule 7.3.84: अङ्गस्य (roots undergo guna when adding प्रत्यय)

    IAST guna mapping:
    - U (उ) → av
    - u (ु) → ov
    - I (ई) → ey
    - i (ि) → iy
    - f (ऋ) → ar
    - A (आ) → A (no change)
    - a (अ) → a (no change)
    """
    mapping = {
        "U": "av",
        "u": "ov",
        "I": "ey",
        "i": "iy",
        "f": "ar",
        "x": "al",
        "A": "A",
        "a": "a",
    }
    result = ""
    i = 0
    while i < len(stem):
        char = stem[i]
        if char in mapping:
            result += mapping[char]
        else:
            result += char
        i += 1
    state.rules_applied.append(rule_id)
    return result


# =============================================================================
# CONSONANT FINAL ADJUSTMENTS (Tripadi, 8.2–8.4)
# =============================================================================

def nominalization_marker(final_char: str) -> str:
    """Rule 8.2–8.4: Add nominalization marker if needed.

    For 3rd person singular masculine, the word often ends in a
    consonant + implicit -a vowel (pausal form).
    """
    if final_char in "tkpghdbm":
        return final_char + "u"  # Placeholder; actual rules are complex
    return final_char


# =============================================================================
# PHASE 1 MAIN DERIVATION PIPELINE (Gana 1 + Lat)
# =============================================================================

def derive_gana1_lat(dhatu: str, purusha: Purusha, vacana: Vacana) -> Tuple[str, List[str]]:
    """
    Derive a गण 1 + लट् form from a root.

    Pipeline (Rule priority):
    1. Apply गुण (guna) to root — Rule 7.3.84
    2. Apply शप् vikarana — Rule 3.1.68
    3. Apply लट् ending — Rule 3.4.77+

    Args:
        dhatu: Root in IAST encoding (e.g., "BU" for भू्)
        purusha: Person (1st/2nd/3rd)
        vacana: Number (singular/dual/plural)

    Returns:
        (surface_form, rules_applied_list)
    """
    state = MorphState(
        dhatu=dhatu,
        gana=1,
        lakara=Lakara.LAT,
        purusha=purusha,
        vacana=vacana,
        pada=Pada.PARASMAIPADA,
    )

    # Step 0: Apply guna to root (Rule 7.3.84)
    # Most गण 1 roots undergo गुण substitution when adding endings
    gunaized = apply_guna(state, dhatu)

    # Step 1: Apply शप् vikarana (Rule 3.1.68)
    stem = gunaized + "a"
    state.rules_applied.append("3.1.68")
    state.stem = stem

    # Step 2: Apply लट् ending (Rule 3.4.77)
    surface = apply_lat_endings(state, stem)
    state.surface = surface

    return surface, state.rules_applied


if __name__ == "__main__":
    # Quick test
    surface, rules = derive_gana1_lat(
        dhatu="BU",
        purusha=Purusha.PRATHAMA,
        vacana=Vacana.EKAVACANA,
    )
    print(f"bhavati (3rd sg): {surface}")
    print(f"Rules applied: {rules}")

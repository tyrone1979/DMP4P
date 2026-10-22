"""
4×9 structure flags + nutrient target flags → which cells may evolve.

frozen[m,j] = structure_ok AND (all nutrients ok OR major j unrelated to unmet nutrients)
evolve = not frozen
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from algo.structure_features import MAJOR_CODES, MEAL_NAMES, meal_major_gram_dict

# major codes responsible for repairing each nutrient when out of range
MAJOR_NUTRIENT_RESPONSIBILITY: dict[str, tuple[int, ...]] = {
    "protein": (1, 2, 3, 4),
    "carb": (5, 6, 7, 9),
    "fiber": (5, 6, 7, 4),
    "saturated_fat": (1, 2, 8, 9),
    "cholesterol": (2, 3),
    "sugar": (9, 6, 1),
    "sodium": (9, 8, 2, 5),
    "potassium": (6, 7, 2, 1),
    "calorie": (1, 2, 3, 4, 5, 6, 7, 8, 9),
    "vitamin_c": (6, 7),
    "vitamin_d": (1, 2, 3),
    "vitamin_b12": (1, 2, 3),
    "calcium": (1, 7),
    "iron": (2, 4, 5, 7),
    "folic_acid": (5, 7, 4),
    "phosphorus": (1, 2, 5),
}

UPPER_ONLY_DEFAULT = frozenset({8, 9})


def structure_cell_flags(
    sol_or_feat: pd.DataFrame | dict[str, float],
    personal: dict[str, Any],
) -> dict[str, Any]:
    """
    Return per-cell structure pass/fail.

    ``flags[meal][major]`` True = in band; ``grams`` same shape; ``false_keys`` list.
    """
    if isinstance(sol_or_feat, dict):
        feat = sol_or_feat
    else:
        feat = meal_major_gram_dict(sol_or_feat)
    bands = personal.get("bands") or {}
    upper = set(personal.get("upper_only_majors") or UPPER_ONLY_DEFAULT)
    flags: dict[str, dict[int, bool]] = {m: {} for m in MEAL_NAMES}
    grams: dict[str, dict[int, float]] = {m: {} for m in MEAL_NAMES}
    false_keys: list[tuple[str, int]] = []
    for meal in MEAL_NAMES:
        for mi in MAJOR_CODES:
            key = f"g_major_{mi}_{meal}"
            v = float(feat.get(key, 0.0) or 0.0)
            b = bands.get(key) or {}
            lo = float(b.get("lo", 0.0) or 0.0)
            hi = float(b["hi"]) if b.get("hi") is not None else 1e9
            ok = v <= hi + 1e-6
            if mi not in upper and lo > 0:
                ok = ok and v >= lo - 1e-6
            flags[meal][mi] = bool(ok)
            grams[meal][mi] = v
            if not ok:
                false_keys.append((meal, mi))
    return {
        "flags": flags,
        "grams": grams,
        "false_keys": false_keys,
        "ok": len(false_keys) == 0,
    }


def target_nutrient_flags(
    sol: pd.DataFrame,
    target_dict: dict,
) -> dict[str, Any]:
    """Per-nutrient in-range flags + direction low|high|ok."""
    flags: dict[str, bool] = {}
    direction: dict[str, str] = {}
    values: dict[str, float] = {}
    for nut, bounds in target_dict.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        if nut not in sol.columns:
            flags[nut] = True
            direction[nut] = "ok"
            values[nut] = 0.0
            continue
        v = float(pd.to_numeric(sol[nut], errors="coerce").fillna(0).sum())
        values[nut] = v
        if v < lo - 1e-6:
            flags[nut] = False
            direction[nut] = "low"
        elif v > hi + 1e-6:
            flags[nut] = False
            direction[nut] = "high"
        else:
            flags[nut] = True
            direction[nut] = "ok"
    return {
        "flags": flags,
        "direction": direction,
        "values": values,
        "ok": all(flags.values()) if flags else True,
    }


def unmet_responsible_majors(
    target_flags: dict[str, Any],
    *,
    responsibility: Optional[dict[str, tuple[int, ...]]] = None,
) -> set[int]:
    """Majors that should stay evolvable because a nutrient they own is unmet."""
    resp = MAJOR_NUTRIENT_RESPONSIBILITY if responsibility is None else responsibility
    out: set[int] = set()
    flags = target_flags.get("flags") or {}
    for nut, ok in flags.items():
        if ok:
            continue
        out.update(int(m) for m in resp.get(nut, ()))
    return out


def cell_evolve_mask(
    structure: dict[str, Any],
    target_flags: dict[str, Any],
    *,
    responsibility: Optional[dict[str, tuple[int, ...]]] = None,
) -> dict[str, Any]:
    """
    evolve[meal][major] True = may crossover/mutate.
    frozen = not evolve.
    """
    s_flags = structure["flags"]
    responsible = unmet_responsible_majors(target_flags, responsibility=responsibility)
    all_nut_ok = bool(target_flags.get("ok", True))
    evolve: dict[str, dict[int, bool]] = {m: {} for m in MEAL_NAMES}
    frozen: dict[str, dict[int, bool]] = {m: {} for m in MEAL_NAMES}
    evolve_keys: list[tuple[str, int]] = []
    for meal in MEAL_NAMES:
        for mi in MAJOR_CODES:
            s_ok = bool(s_flags.get(meal, {}).get(mi, True))
            if not s_ok:
                ev = True
            elif all_nut_ok:
                ev = False
            elif mi in responsible:
                ev = True
            else:
                ev = False
            evolve[meal][mi] = ev
            frozen[meal][mi] = not ev
            if ev:
                evolve_keys.append((meal, mi))
    return {
        "evolve": evolve,
        "frozen": frozen,
        "evolve_keys": evolve_keys,
        "responsible_majors": sorted(responsible),
        "structure_ok": bool(structure.get("ok")),
        "target_ok": all_nut_ok,
    }


def flags_matrix_bool(flags_by_meal_major: dict[str, dict[int, bool]]) -> np.ndarray:
    """Shape (4, 9) bool, meals × majors 1..9."""
    mat = np.zeros((len(MEAL_NAMES), len(MAJOR_CODES)), dtype=bool)
    for i, meal in enumerate(MEAL_NAMES):
        for j, mi in enumerate(MAJOR_CODES):
            mat[i, j] = bool(flags_by_meal_major.get(meal, {}).get(mi, False))
    return mat

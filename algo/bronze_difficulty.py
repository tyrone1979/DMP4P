"""
Bronze GA difficulty: nutrient target + personal category structure.

Base tier from nutrient keys (macros / Na+K / micros); structure pressure can bump
the tier when meal×major envelopes conflict with calorie/carb/protein floors.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from algo.structure_features import MEAL_NAMES

_DIFFICULTY_MICROS = frozenset(
    {
        "vitamin_c",
        "vitamin_d",
        "vitamin_b12",
        "calcium",
        "iron",
        "folic_acid",
        "phosphorus",
    }
)
_ORDER = ("easy", "medium", "hard", "extreme")


def classify_target_difficulty(target_dict: dict) -> str:
    """
    Nutrient-only tier (legacy).

    - easy: macros (+fiber/chol) only, no Na/K
    - medium: roomy Na+K, or a few micros without Na/K
    - hard: tight/low-cal Na+K (or Na-only tight)
    - extreme: Na/K stacked with micros, or very wide nutrient sets
    """
    keys = set(target_dict.keys())
    has_na = "sodium" in keys
    has_k = "potassium" in keys
    n_micro = len(keys & _DIFFICULTY_MICROS)
    cal_lo = float(target_dict["calorie"][0]) if "calorie" in target_dict else 0.0
    cal_hi = float(target_dict["calorie"][1]) if "calorie" in target_dict else 1e9
    width = cal_hi - cal_lo

    if (has_na or has_k) and n_micro >= 1:
        return "extreme"
    if n_micro >= 3 or len(keys) >= 12:
        return "extreme"
    if has_na and has_k:
        if cal_lo <= 1700 or width < 1200:
            return "hard"
        return "medium"
    if has_na:
        return "hard" if cal_lo <= 1700 else "medium"
    if n_micro >= 1:
        return "medium"
    return "easy"


def _band_hi_sum(personal: dict[str, Any], major: int) -> float:
    bands = personal.get("bands") or {}
    total = 0.0
    for meal in MEAL_NAMES:
        b = bands.get(f"g_major_{major}_{meal}")
        if b:
            total += float(b.get("hi") or 0.0)
    if total > 0:
        return total
    # legacy day-only prior
    b = bands.get(f"g_major_{major}")
    return float((b or {}).get("hi") or 0.0)


def structure_pressure_points(
    target_dict: dict,
    personal: Optional[dict[str, Any]],
) -> int:
    """
    Extra difficulty points from structure vs nutrient floors (0–3).

    High carb/calorie/protein floors against tight meal×major envelopes bump GA cost.
    """
    if not personal or not (personal.get("bands") or personal.get("groups")):
        return 0

    cal_lo = float(target_dict["calorie"][0]) if "calorie" in target_dict else 0.0
    cal_hi = float(target_dict["calorie"][1]) if "calorie" in target_dict else 1e9
    if cal_hi >= 9e6:
        cal_hi = cal_lo * 2.0 if cal_lo > 0 else 3000.0
    cal_mid = 0.5 * (cal_lo + cal_hi)
    carb_lo = float(target_dict["carb"][0]) if "carb" in target_dict else 0.0
    protein_lo = float(target_dict["protein"][0]) if "protein" in target_dict else 0.0

    grain_hi = _band_hi_sum(personal, 5)
    meat_hi = _band_hi_sum(personal, 2)
    dairy_hi = _band_hi_sum(personal, 1)
    fruit_hi = _band_hi_sum(personal, 6)
    veg_hi = _band_hi_sum(personal, 7)
    # non-grain carb room (fruit+veg+dairy) — rough mass proxy
    other_carb_hi = fruit_hi + veg_hi + dairy_hi * 0.3
    g_scale = float(personal.get("g_scale") or 1.0)
    expected_g = float(personal.get("expected_total_g") or 0.0)

    pts = 0
    # Carb floor vs grain (+ other) envelope
    if carb_lo >= 350 and grain_hi > 0 and grain_hi + other_carb_hi < carb_lo * 1.15:
        pts += 1
    if carb_lo >= 550 and grain_hi > 0 and grain_hi < carb_lo * 0.95:
        pts += 1
    # High energy demand vs structure mass scale
    if cal_mid >= 3200 and g_scale >= 1.25:
        pts += 1
    if cal_lo >= 2500 or cal_mid >= 4000:
        pts += 1
    # Protein floor vs meat/dairy mass room (~3–4 g food / g protein rough)
    protein_room = meat_hi + dairy_hi
    if protein_lo >= 120 and protein_room > 0 and protein_room < protein_lo * 2.5:
        pts += 1
    # Low expected mass but high calorie mid → density fight
    if expected_g > 0 and cal_mid >= 2800 and (cal_mid / expected_g) >= 1.15:
        pts += 1

    return int(min(3, pts))


def bump_difficulty(base: str, points: int) -> str:
    base = base if base in _ORDER else "easy"
    idx = _ORDER.index(base)
    return _ORDER[min(len(_ORDER) - 1, idx + max(0, int(points)))]


def classify_bronze_difficulty(
    target_dict: dict,
    *,
    personal: Optional[dict[str, Any]] = None,
    sex=None,
    bmi: float | None = None,
    age=None,
    resolve_structure: bool = True,
) -> dict[str, Any]:
    """
    Return ``{difficulty, base, structure_points, stratum, g_scale}``.

    If ``personal`` is None and demo is given, resolve category structure once.
    """
    base = classify_target_difficulty(target_dict)
    pers = personal
    if pers is None and resolve_structure and (bmi is not None or sex is not None or age is not None):
        try:
            from algo.structure_category import select_personal_category_for_bronze

            pers = select_personal_category_for_bronze(
                {"sex": sex, "bmi": bmi, "age": age},
                target_dict,
            )
        except Exception:
            pers = None
    pts = structure_pressure_points(target_dict, pers)
    diff = bump_difficulty(base, pts)
    return {
        "difficulty": diff,
        "base_difficulty": base,
        "structure_points": pts,
        "stratum": (pers or {}).get("stratum"),
        "g_scale": (pers or {}).get("g_scale"),
        "expected_total_g": (pers or {}).get("expected_total_g"),
    }


def classify_bronze_difficulty_label(
    target_dict: dict,
    *,
    personal: Optional[dict[str, Any]] = None,
    sex=None,
    bmi: float | None = None,
    age=None,
) -> str:
    return str(
        classify_bronze_difficulty(
            target_dict,
            personal=personal,
            sex=sex,
            bmi=bmi,
            age=age,
        )["difficulty"]
    )


def difficulty_counts_labels(labels: list[str]) -> dict[str, int]:
    return dict(Counter(labels))

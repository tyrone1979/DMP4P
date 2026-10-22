"""
Deterministic meal-placement flags for candidate rows.

Built once at snapshot time so all algorithms share the same allowed_row mask.
Semantics chosen to mirror production archive (ga_gpu_v2._update_archive):
  - empty/missing food_desc → do NOT reject (production ``if not name: continue``)
  - breakfast_only outside breakfast → disallowed
  - breakfast_snack_only on lunch/dinner → disallowed

This is flag-equivalent to production given the same exclusion YAML + food_code
table at build time. We do **not** claim perfect equivalence if those artifacts
drift after the snapshot was frozen.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd


def attach_placement_flags(
    pool: pd.DataFrame,
    *,
    food_desc: Optional[dict[int, str]] = None,
    food_code_path: str | Path = "data/nhanes/data/food_code.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from algo.bronze_food_filter import is_breakfast_only, is_breakfast_snack_only

    df = pool.copy()
    if "meal" not in df.columns:
        if "meal_id" in df.columns:
            from algo.diet_structure_priors import ID_TO_MEAL

            df["meal"] = df["meal_id"].map(ID_TO_MEAL)
        elif "eating_type" in df.columns:
            from algo.diet_structure_priors import meal_from_eating_type

            df["meal"] = df["eating_type"].map(meal_from_eating_type)
        else:
            df["meal"] = "snack"
    meal = df["meal"].astype(str).str.lower()

    if food_desc is None:
        food_desc = {}
        p = Path(food_code_path)
        if p.exists():
            try:
                fc = pd.read_csv(p, usecols=["food_id", "food_desc"])
                food_desc = dict(zip(fc.food_id.astype(int), fc.food_desc.astype(str)))
            except Exception:
                food_desc = {}

    fids = pd.to_numeric(df["food_id"], errors="coerce").fillna(-1).astype(int)
    descs = fids.map(lambda x: food_desc.get(int(x), "") if int(x) > 0 else "")
    missing = descs.astype(str).str.len() == 0
    allowed = []
    reasons = []
    for m, d, miss in zip(meal.tolist(), descs.tolist(), missing.tolist()):
        if miss:
            allowed.append(True)
            reasons.append("desc_missing_treat_allowed")
            continue
        if is_breakfast_only(d) and m != "breakfast":
            allowed.append(False)
            reasons.append("breakfast_only_wrong_meal")
            continue
        if m in ("lunch", "dinner") and is_breakfast_snack_only(d):
            allowed.append(False)
            reasons.append("breakfast_snack_on_ld")
            continue
        allowed.append(True)
        reasons.append("ok")

    df["bench_placement_allowed"] = allowed
    df["bench_desc_missing"] = missing.to_numpy()
    df["bench_placement_reason"] = reasons
    meta = {
        "n_rows": int(len(df)),
        "n_desc_missing": int(missing.sum()),
        "n_disallowed": int((~pd.Series(allowed)).sum()),
        "n_allowed": int(sum(allowed)),
        "food_desc_source": str(food_code_path),
        "equivalence_note": (
            "Flags mirror production archive skip-on-empty-desc and "
            "is_breakfast_only / is_breakfast_snack_only at snapshot build time. "
            "Not re-resolved at search time."
        ),
    }
    return df, meta

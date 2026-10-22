"""
Shared operable (化零为整) food_user store.

Built once by scripts/recalc_actual_diet_nutrients.py (or review pipeline),
then reused by keep-mains Golden and Bronze pool loading — never re-snap
NHANES raw grams on every day.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Canonical location (symlink or copy of the review artifact)
CANONICAL = ROOT / "data" / "nhanes" / "food_user_operable_units.csv"
# Existing one-shot build from phase ①
REVIEW_ARTIFACT = ROOT / "outputs" / "actual_diet_review" / "actual_food_user_units_nutrients.csv"
META_NAME = "food_user_operable_units.meta.json"


def resolve_operable_food_user_path(path: str | Path | None = None) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    if CANONICAL.exists():
        return CANONICAL
    if REVIEW_ARTIFACT.exists():
        return REVIEW_ARTIFACT
    raise FileNotFoundError(
        f"operable food_user missing; expected {CANONICAL} or {REVIEW_ARTIFACT}. "
        "Run scripts/recalc_actual_diet_nutrients.py first."
    )


def ensure_canonical_symlink() -> Path:
    """Point data/nhanes/food_user_operable_units.csv at the review artifact if needed."""
    if CANONICAL.exists() or CANONICAL.is_symlink():
        return CANONICAL
    if not REVIEW_ARTIFACT.exists():
        raise FileNotFoundError(REVIEW_ARTIFACT)
    CANONICAL.parent.mkdir(parents=True, exist_ok=True)
    try:
        CANONICAL.symlink_to(REVIEW_ARTIFACT.resolve())
    except OSError:
        # fallback: leave canonical missing, callers use REVIEW_ARTIFACT
        return REVIEW_ARTIFACT
    meta = {
        "source": str(REVIEW_ARTIFACT),
        "note": "symlink to one-shot operable unit + nutrient recalc",
    }
    (CANONICAL.parent / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return CANONICAL


def is_operable_frame(df: pd.DataFrame) -> bool:
    """Operable if display units present and mass is grams (+ optional grams_raw)."""
    base = {"food_id", "grams", "display_unit"}
    if not base.issubset(set(df.columns)):
        return False
    # Legacy artifacts carry grams_snapped; bronze contract is grams_raw + grams.
    return "grams_snapped" in df.columns or "grams_raw" in df.columns


@lru_cache(maxsize=2)
def _load_cached(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    return pd.read_csv(path)


def load_operable_food_user(
    path: str | Path | None = None,
    *,
    ensure_link: bool = True,
) -> pd.DataFrame:
    """Load precomputed operable food_user (units + nutrients)."""
    if ensure_link:
        try:
            ensure_canonical_symlink()
        except FileNotFoundError:
            pass
    p = resolve_operable_food_user_path(path)
    df = _load_cached(str(p.resolve()))
    # normalize key dtypes
    out = df.copy()
    out["user_id"] = pd.to_numeric(out["user_id"], errors="coerce")
    out["day"] = pd.to_numeric(out["day"], errors="coerce")
    out["food_id"] = pd.to_numeric(out["food_id"], errors="coerce").fillna(0).astype(int)
    out["grams"] = pd.to_numeric(out["grams"], errors="coerce").fillna(0.0)
    if "eating_type" in out.columns:
        out["eating_type"] = pd.to_numeric(out["eating_type"], errors="coerce")
    return out


def clear_operable_cache() -> None:
    _load_cached.cache_clear()

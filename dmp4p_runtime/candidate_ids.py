"""Stable candidate row identities independent of DataFrame index."""
from __future__ import annotations

import hashlib
from typing import Iterable, Optional

import pandas as pd


def row_candidate_id(
    food_id,
    meal,
    grams,
    nutrients: dict,
) -> str:
    parts = [
        str(int(float(food_id))),
        str(meal).strip().lower(),
        f"{float(grams):.8g}",
    ]
    for k in sorted(nutrients.keys()):
        parts.append(f"{k}={float(nutrients[k]):.8g}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def assign_candidate_ids(pool: pd.DataFrame, nutrient_cols: Iterable[str]) -> pd.DataFrame:
    df = pool.copy()
    cols = [c for c in nutrient_cols if c in df.columns]
    ids = []
    for _, r in df.iterrows():
        nuts = {c: float(pd.to_numeric(r.get(c, 0), errors="coerce") or 0.0) for c in cols}
        meal = r.get("meal", "")
        ids.append(
            row_candidate_id(r.get("food_id"), meal, r.get("grams", 0.0), nuts)
        )
    df["candidate_id"] = ids
    return df


def candidate_id_set_hash(candidate_ids: Iterable[str]) -> str:
    payload = ",".join(sorted(str(x) for x in candidate_ids))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def pool_candidate_set_hash(pool: pd.DataFrame) -> str:
    if "candidate_id" not in pool.columns:
        raise ValueError("pool missing candidate_id")
    return candidate_id_set_hash(pool["candidate_id"].astype(str).tolist())


def assert_same_candidate_content(
    a: pd.DataFrame,
    b: pd.DataFrame,
    nutrient_cols: Optional[list[str]] = None,
) -> None:
    """Same candidate_id set; grams/meal/nutrients match per id (order may differ)."""
    if "candidate_id" not in a.columns or "candidate_id" not in b.columns:
        raise AssertionError("candidate_id required")
    sa = set(a["candidate_id"].astype(str))
    sb = set(b["candidate_id"].astype(str))
    if sa != sb:
        raise AssertionError(
            f"candidate_id set mismatch: only_a={len(sa-sb)} only_b={len(sb-sa)}"
        )
    cols = ["grams", "meal", "food_id"] + list(nutrient_cols or [])
    cols = [c for c in cols if c in a.columns and c in b.columns]
    a_i = a.set_index("candidate_id")
    b_i = b.set_index("candidate_id")
    for cid in sa:
        for c in cols:
            va, vb = a_i.at[cid, c], b_i.at[cid, c]
            if c in ("meal",):
                if str(va).lower() != str(vb).lower():
                    raise AssertionError(f"{cid} meal mismatch {va} vs {vb}")
            else:
                if abs(float(va) - float(vb)) > 0:
                    raise AssertionError(f"{cid} {c} mismatch {va} vs {vb}")

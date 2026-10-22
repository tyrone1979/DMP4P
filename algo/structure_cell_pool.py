"""
Partition a Bronze candidate pool into meal × FNDDS-major (4×9) contiguous slices.

Cells with personal band hi≤0 are hard-locked: no rows enter the GA pool.
"""
from __future__ import annotations

import secrets
from typing import Any, Optional

import pandas as pd

from algo.structure_features import MAJOR_CODES, MEAL_NAMES


def _major_from_food_id(fid) -> int:
    try:
        s = str(int(float(fid))).zfill(8)
        return int(s[0]) if s[0].isdigit() else 0
    except (TypeError, ValueError):
        return 0


def annotate_meal_major(pool: pd.DataFrame) -> pd.DataFrame:
    df = pool.copy()
    if "meal" not in df.columns and "eating_type" in df.columns:
        from algo.diet_structure_priors import meal_from_eating_type

        df["meal"] = df["eating_type"].map(meal_from_eating_type)
    df["meal"] = df["meal"].astype(str).str.lower()
    if "food_group_code" in df.columns:
        maj = pd.to_numeric(df["food_group_code"], errors="coerce").fillna(0).astype(int)
        # food_group_code is often 1-digit major already
        df["major"] = maj.where(maj.between(1, 9), df["food_id"].map(_major_from_food_id))
    else:
        df["major"] = df["food_id"].map(_major_from_food_id)
    return df


def forbidden_cells_from_personal(personal: Optional[dict[str, Any]]) -> set[tuple[str, int]]:
    """meal×major with hi≤0 — must never appear in the candidate pool."""
    out: set[tuple[str, int]] = set()
    bands = (personal or {}).get("bands") or {}
    for meal in MEAL_NAMES:
        for mi in MAJOR_CODES:
            b = bands.get(f"g_major_{mi}_{meal}") or {}
            hi = b.get("hi")
            if hi is not None and float(hi) <= 0:
                out.add((meal, int(mi)))
    return out


def drop_forbidden_structure_cells(
    pool: pd.DataFrame,
    personal: Optional[dict[str, Any]],
) -> tuple[pd.DataFrame, int, set[tuple[str, int]]]:
    """
    Remove every row whose (meal, major) has band hi≤0.

    Returns (filtered_pool, n_dropped, forbidden_keys).
    """
    forbidden = forbidden_cells_from_personal(personal)
    if not forbidden or pool is None or len(pool) == 0:
        return pool, 0, forbidden
    df = annotate_meal_major(pool)
    mask = pd.Series(False, index=df.index)
    for meal, mi in forbidden:
        mask = mask | ((df["meal"] == meal) & (df["major"] == int(mi)))
    n_drop = int(mask.sum())
    if n_drop == 0:
        return pool.reset_index(drop=True) if hasattr(pool, "reset_index") else pool, 0, forbidden
    kept = df.loc[~mask].reset_index(drop=True)
    return kept, n_drop, forbidden


def build_structure_cell_pool(
    pool: pd.DataFrame,
    *,
    personal: Optional[dict[str, Any]] = None,
    per_cell_quota: int = 80,
) -> tuple[pd.DataFrame, dict[str, dict[int, tuple[int, int]]], dict[str, Any]]:
    """
    Reorder ``pool`` into contiguous cell blocks.

    hi≤0 cells are hard-locked: empty range, zero candidate rows (not in output).

    Returns
    -------
    ordered_pool, cell_ranges, meta
      meta: forbidden_cells, n_dropped_forbidden, n_nonempty_cells
    """
    df = annotate_meal_major(pool)
    bands = (personal or {}).get("bands") or {}
    forbidden = forbidden_cells_from_personal(personal)
    parts: list[pd.DataFrame] = []
    ranges: dict[str, dict[int, tuple[int, int]]] = {m: {} for m in MEAL_NAMES}
    cursor = 0
    n_dropped = 0
    for meal in MEAL_NAMES:
        for mi in MAJOR_CODES:
            sub = df[(df["meal"] == meal) & (df["major"] == mi)]
            key = f"g_major_{mi}_{meal}"
            b = bands.get(key) or {}
            hi = b.get("hi")
            if hi is not None and float(hi) <= 0:
                # Hard lock: do not put any of these rows in the GA pool
                n_dropped += int(len(sub))
                ranges[meal][mi] = (cursor, cursor)
                continue
            if sub.empty:
                ranges[meal][mi] = (cursor, cursor)
                continue
            # Capability top + random explore (easy feasibility without locking one dish)
            q = int(per_cell_quota)
            rs = secrets.randbits(31)
            explore = 0.4
            if "_prio" in sub.columns and len(sub) > q:
                ranked = sub.sort_values("_prio", ascending=False)
                n_top = max(1, int(q * (1.0 - explore)))
                top = ranked.head(n_top)
                rest = ranked.iloc[n_top:]
                n_rand = min(len(rest), q - len(top))
                rnd = (
                    rest.sample(n=n_rand, random_state=rs)
                    if n_rand > 0
                    else rest.iloc[0:0]
                )
                take = pd.concat([top, rnd]).drop_duplicates()
            elif len(sub) > q:
                take = sub.sample(n=q, random_state=rs)
            else:
                take = sub.sample(frac=1.0, random_state=rs)
            n = len(take)
            parts.append(take)
            ranges[meal][mi] = (cursor, cursor + n)
            cursor += n
    meta = {
        "forbidden_cells": sorted(forbidden),
        "n_dropped_forbidden": int(n_dropped),
        "n_nonempty_cells": sum(
            1 for m in ranges for _mi, (a, b) in ranges[m].items() if b > a
        ),
    }
    if not parts:
        empty = df.iloc[0:0].reset_index(drop=True)
        return empty, ranges, meta
    out = pd.concat(parts, ignore_index=True)
    # Sanity: no forbidden cell rows survived
    check = annotate_meal_major(out)
    for meal, mi in forbidden:
        n_bad = int(((check["meal"] == meal) & (check["major"] == int(mi))).sum())
        if n_bad:
            raise RuntimeError(
                f"forbidden cell {meal}×{mi} still has {n_bad} rows after cell pool build"
            )
    return out, ranges, meta


def cell_key(meal: str, major: int) -> str:
    return f"{meal}:{int(major)}"

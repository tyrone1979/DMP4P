"""Lightweight FPED coded-id set — no EquivFoodReplacer / nutrient index."""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _coded_cfg() -> dict:
    p = ROOT / "config" / "equiv_food_replace.yaml"
    with open(p, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("coded_set") or {}


def load_discontinued_maps(cfg: Optional[dict] = None) -> list[dict[int, Optional[int]]]:
    """Excel maps, pickled after first read."""
    from algo.equiv_food_replace import _load_discontinued_maps

    cfg = cfg or _coded_cfg()
    ddir = ROOT / str(cfg.get("discontinued_dir", "data/fndds_discontinued"))
    files = list(cfg.get("discontinued_files") or [])
    cache = ddir / "_disc_maps.pkl"
    stamp = tuple((f, (ddir / f).stat().st_mtime if (ddir / f).exists() else 0) for f in files)
    if cache.exists():
        try:
            payload = pickle.loads(cache.read_bytes())
            if payload.get("stamp") == stamp:
                return payload["maps"]
        except Exception:
            pass
    maps = _load_discontinued_maps(ddir, files)
    try:
        cache.write_bytes(pickle.dumps({"stamp": stamp, "maps": maps}))
    except Exception:
        pass
    return maps


@lru_cache(maxsize=1)
def fped_coded_ids() -> frozenset[int]:
    cfg = _coded_cfg()
    coded: set[int] = set()
    for rel in cfg.get("fped_csvs") or []:
        p = ROOT / rel
        if not p.exists():
            continue
        df = pd.read_csv(p, usecols=["FOODCODE"])
        coded |= set(pd.to_numeric(df["FOODCODE"], errors="coerce").dropna().astype(int))
    return frozenset(coded)


@lru_cache(maxsize=1)
def fped_ok_ids() -> frozenset[int]:
    """FPED codes plus any discontinued code that official-maps into FPED."""
    from algo.equiv_food_replace import chain_official_successor

    coded = set(fped_coded_ids())
    cfg = _coded_cfg()
    ok = set(coded)
    if cfg.get("use_official_discontinued_maps", True):
        maps = load_discontinued_maps(cfg)
        starts: set[int] = set()
        for mp in maps:
            starts.update(mp.keys())
        for fid in starts:
            if fid in ok:
                continue
            succ, _ = chain_official_successor(fid, coded, maps)
            if succ is not None and int(succ) in coded:
                ok.add(int(fid))
    return frozenset(ok)

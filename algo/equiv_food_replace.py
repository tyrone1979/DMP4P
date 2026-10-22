"""
Equivalent food replacement (reusable).

Two modes share the same nutrient-density index:

1. **code_repair** — foods lacking modern FPED codes → coded foods
   (same-category bias + USDA discontinued maps).
2. **avoidance** — user忌口 / allergy: replace forbidden foods with
   nutritionally similar foods from *other* categories (e.g. cow's milk → almond milk).

Public API
----------
- ``EquivFoodReplacer`` — ``suggest`` / ``remap_day`` / ``remap_frame``
- ``AvoidanceSpec`` / ``load_avoidance_preset`` — 忌口定义
- ``RemapResult`` — day-level outcome + audit rows

Typical use
-----------
>>> repl = EquivFoodReplacer()
>>> # code repair
>>> items, summary, audit = repl.remap_frame(bronze_df)
>>> # dairy avoidance (cross-category)
>>> avoid = load_avoidance_preset("dairy")
>>> items, summary, audit = repl.remap_frame(bronze_df, avoidance=avoid)
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
import yaml

from algo.diet_structure_priors import food_group_code, food_id_fingerprint, fingerprint_hash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "equiv_food_replace.yaml"

NUTRIENT_DEFAULTS = [
    "calorie",
    "protein",
    "carb",
    "sugar",
    "fiber",
    "saturated_fat",
    "cholesterol",
    "folic_acid",
    "vitamin_b12",
    "vitamin_c",
    "vitamin_d",
    "calcium",
    "phosphorus",
    "potassium",
    "iron",
    "sodium",
]


def load_equiv_config(path: Optional[Path] = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_target(t) -> dict:
    if isinstance(t, dict):
        return t
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return {}
    return ast.literal_eval(str(t))


def subgroup2(food_id) -> int:
    """First 2 digits of 8-digit FNDDS code."""
    if pd.isna(food_id):
        return 0
    s = str(int(float(food_id))).zfill(8)
    try:
        return int(s[:2])
    except ValueError:
        return 0


def nutrients_in_target(
    day: pd.DataFrame,
    target: dict,
    features: Iterable[str],
    tol_rel: float = 0.01,
    tol_abs: float = 1.0,
    infinite_upper: float = 1.0e7,
) -> tuple[bool, list[str]]:
    fails: list[str] = []
    for nut, bounds in target.items():
        if nut not in features and nut not in day.columns:
            continue
        if nut not in day.columns:
            continue
        lo, hi = float(bounds[0]), float(bounds[1])
        val = float(pd.to_numeric(day[nut], errors="coerce").fillna(0).sum())
        slack_lo = max(tol_abs, abs(lo) * tol_rel) if tol_rel or tol_abs else 0.0
        if hi >= infinite_upper:
            if val < lo - slack_lo:
                fails.append(f"{nut}={val:.2f} not in [{lo},{hi}]")
            continue
        slack_hi = max(tol_abs, abs(hi) * tol_rel) if tol_rel or tol_abs else 0.0
        if val < lo - slack_lo or val > hi + slack_hi:
            fails.append(f"{nut}={val:.2f} not in [{lo},{hi}]")
    return len(fails) == 0, fails


def _load_discontinued_maps(dir_path: Path, files: list[str]) -> list[dict[int, Optional[int]]]:
    """Return ordered list of old→new maps (new may be None if dropped)."""
    maps: list[dict[int, Optional[int]]] = []
    if not dir_path.exists():
        return maps
    for name in files:
        path = dir_path / name
        if not path.exists():
            continue
        raw = pd.read_excel(path, sheet_name=0, header=None)
        hdr = 1
        for i in range(min(5, len(raw))):
            row = " ".join(str(x).lower() for x in raw.iloc[i].tolist())
            if "food code" in row and "discontinu" in row:
                hdr = i
                break
        df = pd.read_excel(path, sheet_name=0, header=hdr)
        cols = list(df.columns)
        if len(cols) < 4:
            continue
        old = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        new = pd.to_numeric(df.iloc[:, 3], errors="coerce")
        m: dict[int, Optional[int]] = {}
        for o, n in zip(old, new):
            if pd.isna(o):
                continue
            m[int(o)] = int(n) if pd.notna(n) else None
        maps.append(m)
    return maps


def chain_official_successor(
    food_id: int,
    coded: set[int],
    maps: list[dict[int, Optional[int]]],
) -> tuple[Optional[int], str]:
    """Follow USDA discontinued maps until coded or dead-end."""
    cur = int(food_id)
    if cur in coded:
        return cur, "already_coded"
    hops = []
    for mp in maps:
        if cur in coded:
            return cur, "official:" + ">".join(hops) if hops else "already_coded"
        if cur not in mp:
            continue
        nxt = mp[cur]
        hops.append(f"{cur}->{nxt}")
        if nxt is None:
            return None, "official_dropped:" + ">".join(hops)
        cur = int(nxt)
        if cur in coded:
            return cur, "official:" + ">".join(hops)
    if cur in coded:
        return cur, "official:" + ">".join(hops) if hops else "already_coded"
    return None, "no_official_successor"


@dataclass
class AvoidanceSpec:
    """
    User 忌口 / allergy definition.

    A food is forbidden if it matches any forbid rule AND does not match
    an allow keyword exception (e.g. almond milk under a dairy avoid).
    """

    name: str = "custom"
    description: str = ""
    forbid_food_ids: set[int] = field(default_factory=set)
    forbid_major_groups: set[int] = field(default_factory=set)
    forbid_subgroup2: set[int] = field(default_factory=set)
    forbid_keywords: list[str] = field(default_factory=list)
    allow_keywords: list[str] = field(default_factory=list)
    preferred_keywords: list[str] = field(default_factory=list)
    preferred_major_groups: set[int] = field(default_factory=set)
    cross_category: bool = True

    def is_forbidden(self, food_id: int, desc: str = "") -> bool:
        fid = int(food_id)
        d = (desc or "").lower()
        if self._keywords_hit(d, self.allow_keywords):
            return False
        if fid in self.forbid_food_ids:
            return True
        if food_group_code(fid) in self.forbid_major_groups:
            return True
        if subgroup2(fid) in self.forbid_subgroup2:
            return True
        if d and self._keywords_hit(d, self.forbid_keywords):
            return True
        return False

    @staticmethod
    def _keywords_hit(desc_lower: str, keywords: list[str]) -> bool:
        """Substring match with a few safe-guards (e.g. 'no butter')."""
        d = desc_lower
        for kw in keywords:
            if not kw:
                continue
            k = kw.lower()
            if k == "butter" and ("no butter" in d or "without butter" in d):
                continue
            if k == "milk" and ("non-dairy" in d or "nondairy" in d or "almond milk" in d
                                or "soy milk" in d or "oat milk" in d or "rice milk" in d
                                or "coconut milk" in d or "cashew milk" in d):
                # plant / non-dairy handled by allow list; skip bare milk here too
                # if those phrases present — still let allow_keywords decide first
                pass
            if k in d:
                # avoid matching 'butter' inside 'no butter' already handled
                if k == "butter" and d.find(k) >= 0:
                    # skip if only occurrence is in 'no butter' / 'without butter'
                    tmp = d.replace("no butter", " ").replace("without butter", " ")
                    if "butter" not in tmp:
                        continue
                return True
        return False


def load_avoidance_preset(
    name: str,
    config: Optional[Any] = None,
) -> AvoidanceSpec:
    """Load a named preset from config (dairy / peanut / shellfish / …)."""
    if isinstance(config, (str, Path)):
        cfg = load_equiv_config(Path(config))
    elif isinstance(config, dict):
        cfg = config
    else:
        cfg = load_equiv_config()
    av = cfg.get("avoidance") or {}
    presets = av.get("presets") or {}
    if name not in presets:
        raise KeyError(f"unknown avoidance preset {name!r}; have {sorted(presets)}")
    p = presets[name] or {}
    return AvoidanceSpec(
        name=name,
        description=str(p.get("description") or ""),
        forbid_food_ids={int(x) for x in (p.get("forbid_food_ids") or [])},
        forbid_major_groups={int(x) for x in (p.get("forbid_major_groups") or [])},
        forbid_subgroup2={int(x) for x in (p.get("forbid_subgroup2") or [])},
        forbid_keywords=[str(x) for x in (p.get("forbid_keywords") or [])],
        allow_keywords=[str(x) for x in (p.get("allow_keywords") or [])],
        preferred_keywords=[str(x) for x in (p.get("preferred_keywords") or [])],
        preferred_major_groups={int(x) for x in (p.get("preferred_major_groups") or [])},
        cross_category=bool(av.get("cross_category_default", True)),
    )


def merge_avoidances(*specs: AvoidanceSpec) -> AvoidanceSpec:
    """Union of several avoidance specs (e.g. dairy + peanut)."""
    if not specs:
        return AvoidanceSpec(name="empty")
    out = AvoidanceSpec(
        name="+".join(s.name for s in specs),
        description="; ".join(s.description for s in specs if s.description),
        cross_category=any(s.cross_category for s in specs),
    )
    for s in specs:
        out.forbid_food_ids |= set(s.forbid_food_ids)
        out.forbid_major_groups |= set(s.forbid_major_groups)
        out.forbid_subgroup2 |= set(s.forbid_subgroup2)
        out.forbid_keywords.extend(s.forbid_keywords)
        out.allow_keywords.extend(s.allow_keywords)
        out.preferred_keywords.extend(s.preferred_keywords)
        out.preferred_major_groups |= set(s.preferred_major_groups)
    return out


@dataclass
class ReplacementCandidate:
    food_id: int
    distance: float
    method: str
    major_group: int
    subgroup2: int
    dens: dict[str, float] = field(default_factory=dict)


@dataclass
class RemapResult:
    day: pd.DataFrame
    ok: bool
    target_ok: bool
    n_replaced: int
    fail_reasons: list[str] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)


class EquivFoodReplacer:
    """
    Build nutrient-density index over coded foods; replace uncoded items.

    Parameters
    ----------
    config : dict or path, optional
    coded_predicate : optional callable(food_id)->bool
        Override default "in FPED / official-map successor" coded check for
        *sources that need replacement*. Candidates are always drawn from the
        coded FPED set unless ``candidate_ids`` is passed.
    candidate_ids : optional set of food_ids allowed as replacements.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        coded_predicate: Optional[Callable[[int], bool]] = None,
        candidate_ids: Optional[set[int]] = None,
    ):
        if isinstance(config, (str, Path)):
            self.cfg = load_equiv_config(Path(config))
        elif isinstance(config, dict):
            self.cfg = config
        else:
            self.cfg = load_equiv_config()

        self.features: list[str] = list(
            self.cfg.get("nutrition", {}).get("features") or NUTRIENT_DEFAULTS
        )
        self._coded_predicate = coded_predicate
        self._build_indexes(candidate_ids=candidate_ids)

    # ------------------------------------------------------------------ build
    def _build_indexes(self, candidate_ids: Optional[set[int]] = None) -> None:
        nut_cfg = self.cfg["nutrition"]
        coded_cfg = self.cfg["coded_set"]
        sim_cfg = self.cfg["similarity"]

        # FPED coded universe
        coded: set[int] = set()
        for rel in coded_cfg.get("fped_csvs") or []:
            p = ROOT / rel
            if not p.exists():
                continue
            df = pd.read_csv(p, usecols=["FOODCODE"])
            coded |= set(pd.to_numeric(df["FOODCODE"], errors="coerce").dropna().astype(int))
        self.fped_coded = set(coded)

        self.disc_maps: list[dict[int, Optional[int]]] = []
        if coded_cfg.get("use_official_discontinued_maps", True):
            self.disc_maps = _load_discontinued_maps(
                ROOT / coded_cfg.get("discontinued_dir", "data/fndds_discontinued"),
                list(coded_cfg.get("discontinued_files") or []),
            )

        # Expand coded set with anything that official-maps into FPED
        # (sources stay uncoded until remapped; candidates = FPED)
        self.coded_ids = set(self.fped_coded)
        if candidate_ids is not None:
            self.candidate_ids = {int(x) for x in candidate_ids} & self.fped_coded
        else:
            self.candidate_ids = set(self.fped_coded)

        # Nutrient densities per 100 g
        fn_path = ROOT / nut_cfg["food_nutrition_csv"]
        fn = pd.read_csv(fn_path)
        min_g = float(nut_cfg.get("min_grams_for_density", 1.0))
        fn["food_id"] = pd.to_numeric(fn["food_id"], errors="coerce")
        fn["grams"] = pd.to_numeric(fn["grams"], errors="coerce")
        fn = fn.dropna(subset=["food_id", "grams"])
        fn = fn[fn["grams"] >= min_g]
        for c in self.features:
            if c not in fn.columns:
                fn[c] = 0.0
            fn[c] = pd.to_numeric(fn[c], errors="coerce").fillna(0.0)

        # Average density across portion rows
        dens_rows = []
        for fid, g in fn.groupby("food_id", sort=False):
            w = g["grams"].to_numpy(dtype=float)
            row = {"food_id": int(fid)}
            for c in self.features:
                # nutrient amount / grams * 100 → per 100 g; weight rows by grams
                amt = g[c].to_numpy(dtype=float)
                dens = amt / w * 100.0
                row[c] = float(np.average(dens, weights=w))
            dens_rows.append(row)
        dens_df = pd.DataFrame(dens_rows).set_index("food_id")
        self.density = dens_df

        # Descriptions (optional)
        self.desc: dict[int, str] = {}
        fc_path = ROOT / nut_cfg.get("food_code_csv", "")
        if fc_path.exists():
            fc = pd.read_csv(fc_path)
            if "food_id" in fc.columns and "food_desc" in fc.columns:
                for fid, d in zip(
                    pd.to_numeric(fc["food_id"], errors="coerce"),
                    fc["food_desc"].astype(str),
                ):
                    if pd.notna(fid):
                        self.desc[int(fid)] = d

        # Candidate matrix
        cand = [i for i in sorted(self.candidate_ids) if i in dens_df.index]
        self.cand_ids = np.asarray(cand, dtype=int)
        raw = dens_df.loc[cand, self.features].to_numpy(dtype=float)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        transform = str(nut_cfg.get("transform", "log1p_zscore"))
        if transform == "log1p_zscore":
            mat = np.log1p(np.clip(raw, 0, None))
            mu = mat.mean(axis=0)
            sd = mat.std(axis=0)
            sd[sd < 1e-8] = 1.0
            mat = (mat - mu) / sd
            self._mu, self._sd = mu, sd
        else:
            mat = raw
            self._mu = np.zeros(mat.shape[1])
            self._sd = np.ones(mat.shape[1])
        # L2-normalize rows for cosine
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        self.cand_mat = mat / norms
        self.cand_raw = raw  # per 100g original scale
        self.cand_major = np.asarray([food_group_code(i) for i in self.cand_ids], dtype=int)
        self.cand_sub2 = np.asarray([subgroup2(i) for i in self.cand_ids], dtype=int)

        self.metric = str(sim_cfg.get("metric", "cosine"))
        self.top_k = int(sim_cfg.get("top_k", 8))
        self.max_distance = float(sim_cfg.get("max_distance", 0.85))
        self.prefer_major = bool(sim_cfg.get("prefer_same_major_group", True))
        self.prefer_sub2 = bool(sim_cfg.get("prefer_same_subgroup2", True))
        self.pen_major = float(sim_cfg.get("major_group_penalty", 0.35))
        self.pen_sub2 = float(sim_cfg.get("subgroup2_penalty", 0.10))

    def needs_replace(self, food_id: int) -> bool:
        fid = int(food_id)
        if self._coded_predicate is not None:
            return not bool(self._coded_predicate(fid))
        if fid in self.fped_coded:
            return False
        succ, _ = chain_official_successor(fid, self.fped_coded, self.disc_maps)
        return succ is None

    def _embed(self, food_id: int) -> Optional[np.ndarray]:
        if food_id not in self.density.index:
            return None
        raw = self.density.loc[food_id, self.features].to_numpy(dtype=float)
        raw = np.nan_to_num(raw, nan=0.0)
        mat = np.log1p(np.clip(raw, 0, None))
        mat = (mat - self._mu) / self._sd
        n = np.linalg.norm(mat)
        if n < 1e-12:
            return mat
        return mat / n

    def _raw_dens(self, food_id: int) -> dict[str, float]:
        if food_id not in self.density.index:
            return {c: 0.0 for c in self.features}
        row = self.density.loc[food_id, self.features]
        return {c: float(row[c]) for c in self.features}

    def is_avoided(self, food_id: int, avoidance: AvoidanceSpec) -> bool:
        return avoidance.is_forbidden(int(food_id), self.desc.get(int(food_id), ""))

    def suggest(
        self,
        food_id: int,
        k: Optional[int] = None,
        exclude: Optional[set[int]] = None,
        mode: str = "code_repair",
        avoidance: Optional[AvoidanceSpec] = None,
    ) -> list[ReplacementCandidate]:
        """
        Ranked replacement candidates for one food_id.

        mode:
          - ``code_repair``: official map + same-category nutrient NN
          - ``avoidance`` / ``cross_category``: nutrient NN excluding AvoidanceSpec
            foods; no same-group penalty; preferred keyword/group bonuses
        """
        fid = int(food_id)
        exclude = set(exclude or set())
        out: list[ReplacementCandidate] = []
        cross = mode in ("avoidance", "cross_category") or (
            avoidance is not None and avoidance.cross_category
        )
        av_cfg = self.cfg.get("avoidance") or {}
        if cross:
            k = int(k or av_cfg.get("top_k") or self.top_k)
            max_dist = float(av_cfg.get("max_distance", self.max_distance))
            prefer_major = False
            prefer_sub2 = False
            use_official = False
        else:
            k = int(k or self.top_k)
            max_dist = self.max_distance
            prefer_major = self.prefer_major
            prefer_sub2 = self.prefer_sub2
            use_official = True

        # Build exclude set from avoidance
        if avoidance is not None:
            # Exclude all currently forbidden candidate ids (prefilter by rules)
            # Full scan once per call is OK for ~8k candidates
            for cid in self.cand_ids:
                if self.is_avoided(int(cid), avoidance):
                    exclude.add(int(cid))
            exclude.add(fid)

        # Tier A: official successor (code_repair only; skip if avoided)
        if use_official:
            succ, method = chain_official_successor(fid, self.fped_coded, self.disc_maps)
            if (
                succ is not None
                and succ in self.candidate_ids
                and succ not in exclude
                and succ in self.density.index
            ):
                if avoidance is None or not self.is_avoided(succ, avoidance):
                    out.append(
                        ReplacementCandidate(
                            food_id=succ,
                            distance=0.0,
                            method=method,
                            major_group=food_group_code(succ),
                            subgroup2=subgroup2(succ),
                            dens=self._raw_dens(succ),
                        )
                    )

        # Tier B: nutrient NN
        vec = self._embed(fid)
        if vec is None or len(self.cand_ids) == 0:
            return out[:k]

        if self.metric == "euclidean":
            dist = np.linalg.norm(self.cand_mat - vec, axis=1)
        else:
            sim = self.cand_mat @ vec
            dist = 1.0 - sim

        maj = food_group_code(fid)
        sub = subgroup2(fid)
        if prefer_major:
            dist = dist + self.pen_major * (self.cand_major != maj).astype(float)
        if prefer_sub2:
            dist = dist + self.pen_sub2 * (self.cand_sub2 != sub).astype(float)

        # Cross-category: boost preferred substitutes
        if cross and avoidance is not None:
            kw_bonus = float(av_cfg.get("preferred_keyword_bonus", -0.12))
            maj_bonus = float(av_cfg.get("preferred_major_bonus", -0.05))
            pref_kw = [x.lower() for x in avoidance.preferred_keywords if x]
            pref_maj = set(avoidance.preferred_major_groups)
            src_desc = self.desc.get(fid, "").lower()
            require_pref = [
                str(x).lower()
                for x in (av_cfg.get("require_preferred_for_keywords") or [])
                if x
            ]
            force_pref = bool(require_pref) and any(rk in src_desc for rk in require_pref)
            if pref_kw or pref_maj:
                for i, cid in enumerate(self.cand_ids):
                    ddesc = self.desc.get(int(cid), "").lower()
                    hit_pref = pref_kw and any(kw in ddesc for kw in pref_kw)
                    if hit_pref:
                        dist[i] = dist[i] + kw_bonus
                    elif force_pref and pref_kw:
                        # push non-preferred far away when replacing milk-like items
                        dist[i] = dist[i] + 0.45
                    if pref_maj and int(self.cand_major[i]) in pref_maj:
                        dist[i] = dist[i] + maj_bonus

        order = np.argsort(dist)
        seen = {c.food_id for c in out} | exclude | {fid}
        method_tag = "cross_category_nn" if cross else "nutrient_nn"
        for idx in order:
            cid = int(self.cand_ids[idx])
            if cid in seen:
                continue
            if avoidance is not None and self.is_avoided(cid, avoidance):
                continue
            d = float(dist[idx])
            if d > max_dist and not out:
                if d > max_dist * 1.25:
                    break
            if d > max_dist and out:
                break
            out.append(
                ReplacementCandidate(
                    food_id=cid,
                    distance=d,
                    method=method_tag,
                    major_group=int(self.cand_major[idx]),
                    subgroup2=int(self.cand_sub2[idx]),
                    dens=self._raw_dens(cid),
                )
            )
            seen.add(cid)
            if len(out) >= k:
                break
        return out[:k]

    def _scale_grams(self, old_id: int, new_id: int, grams: float) -> float:
        rep = self.cfg["replacement"]
        scale_by = str(rep.get("scale_by", "calorie"))
        g_lo = float(rep.get("grams_min", 5.0))
        g_hi = float(rep.get("grams_max", 800.0))
        old_d = self._raw_dens(old_id)
        new_d = self._raw_dens(new_id)
        g = float(grams)
        old_v = float(old_d.get(scale_by, 0.0))
        new_v = float(new_d.get(scale_by, 0.0))
        if old_v > 1e-9 and new_v > 1e-9:
            # dens is per 100 g → amount ∝ dens * grams
            g = g * (old_v / new_v)
        return float(np.clip(g, g_lo, g_hi))

    def _apply_item_nutrients(self, row: pd.Series, new_id: int, new_grams: float) -> pd.Series:
        """Fill nutrient columns from density × grams/100."""
        out = row.copy()
        dens = self._raw_dens(new_id)
        out["food_id"] = int(new_id)
        out["grams"] = float(new_grams)
        for c in self.features:
            if c in out.index:
                out[c] = dens[c] * float(new_grams) / 100.0
        if "food_group_code" in out.index:
            out["food_group_code"] = food_group_code(new_id)
        return out

    def _target_tol(self) -> tuple[float, float, float]:
        rep = self.cfg.get("replacement") or {}
        hard = bool(rep.get("hard_target", True))
        if hard:
            return (
                float(rep.get("nutrient_tol_rel", 0.0)),
                float(rep.get("nutrient_tol_abs", 1e-6)),
                float(rep.get("infinite_upper", 1e7)),
            )
        return (
            float(rep.get("nutrient_tol_rel", 0.01)),
            float(rep.get("nutrient_tol_abs", 1.0)),
            float(rep.get("infinite_upper", 1e7)),
        )

    def day_in_target(self, day: pd.DataFrame, target: dict) -> tuple[bool, list[str]]:
        tol_rel, tol_abs, inf_u = self._target_tol()
        return nutrients_in_target(
            day, target, self.features, tol_rel=tol_rel, tol_abs=tol_abs, infinite_upper=inf_u
        )

    def _try_row_swap(
        self,
        df: pd.DataFrame,
        row_i: Any,
        new_id: int,
        new_grams: float,
        target: dict,
    ) -> Optional[pd.DataFrame]:
        """Return a copy with row replaced if day stays in target; else None."""
        if not target:
            trial = df.copy()
            trial.loc[row_i] = self._apply_item_nutrients(trial.loc[row_i], int(new_id), float(new_grams))
            return trial
        trial = df.copy()
        trial.loc[row_i] = self._apply_item_nutrients(trial.loc[row_i], int(new_id), float(new_grams))
        ok, _ = self.day_in_target(trial, target)
        return trial if ok else None

    def _pick_feasible_replacement(
        self,
        df: pd.DataFrame,
        row_i: Any,
        old_id: int,
        old_grams: float,
        cands: list[ReplacementCandidate],
        target: dict,
    ) -> Optional[tuple[pd.DataFrame, ReplacementCandidate, float]]:
        """
        Try candidates × gram factors; accept first that keeps ALL nutrients in target.
        Hard constraint: never commit an out-of-bounds day.
        """
        rep = self.cfg.get("replacement") or {}
        factors = list(rep.get("feasible_gram_factors") or [1.0, 0.9, 0.8, 0.7, 1.1, 1.2])
        max_tries = int(rep.get("feasible_candidate_tries", 20))
        g_lo = float(rep.get("grams_min", 5.0))
        g_hi = float(rep.get("grams_max", 800.0))

        # If no target, first candidate with calorie scale
        if not target:
            if not cands:
                return None
            pick = cands[0]
            new_g = float(np.clip(self._scale_grams(old_id, pick.food_id, old_grams), g_lo, g_hi))
            trial = df.copy()
            trial.loc[row_i] = self._apply_item_nutrients(trial.loc[row_i], pick.food_id, new_g)
            return trial, pick, new_g

        for pick in cands[:max_tries]:
            base_g = self._scale_grams(old_id, pick.food_id, old_grams)
            for fac in factors:
                new_g = float(np.clip(base_g * float(fac), g_lo, g_hi))
                trial = self._try_row_swap(df, row_i, pick.food_id, new_g, target)
                if trial is not None:
                    return trial, pick, new_g
        return None

    def enforce_target_hard(
        self,
        day: pd.DataFrame,
        target: Optional[dict] = None,
        avoidance: Optional[AvoidanceSpec] = None,
        max_rounds: Optional[int] = None,
        protected_food_ids: Optional[set[int]] = None,
    ) -> RemapResult:
        """
        Force day into target [L,U] for every nutrient (hard constraint).

        Strategies: shrink/grow top contributors; swap high-impact foods via suggest.
        Rows whose food_id is in ``protected_food_ids`` are never shrunk/grown/swapped.
        """
        df = day.copy().reset_index(drop=True)
        if target is None and "target" in df.columns:
            try:
                target = _parse_target(df["target"].iloc[0])
            except Exception:
                target = {}
        target = target or {}
        audit: list[dict[str, Any]] = []
        fail_reasons: list[str] = []
        if not target:
            return RemapResult(day=df, ok=True, target_ok=True, n_replaced=0, fail_reasons=[], audit=[])

        rep = self.cfg.get("replacement") or {}
        rounds = int(max_rounds or rep.get("repair_max_rounds", 24))
        g_lo = float(rep.get("grams_min", 5.0))
        g_hi = float(rep.get("grams_max", 800.0))
        inf_u = float(rep.get("infinite_upper", 1e7))
        n_replaced = 0
        protected = {int(x) for x in (protected_food_ids or set())}

        def _is_prot(i: Any) -> bool:
            if not protected:
                return False
            try:
                return int(pd.to_numeric(df.at[i, "food_id"], errors="coerce")) in protected
            except (TypeError, ValueError):
                return False

        def _editable_idx() -> list:
            return [i for i in df.index if not _is_prot(i)]

        ok, fails = self.day_in_target(df, target)
        if ok:
            return RemapResult(day=df, ok=True, target_ok=True, n_replaced=0, fail_reasons=[], audit=[])

        for rnd in range(rounds):
            ok, fails = self.day_in_target(df, target)
            if ok:
                break

            # classify violations
            high_nuts, low_nuts = [], []
            for nut, bounds in target.items():
                if nut not in df.columns:
                    continue
                lo_b, hi_b = float(bounds[0]), float(bounds[1])
                val = float(pd.to_numeric(df[nut], errors="coerce").fillna(0).sum())
                if val < lo_b:
                    low_nuts.append((nut, lo_b, hi_b, val))
                if hi_b < inf_u and val > hi_b:
                    high_nuts.append((nut, lo_b, hi_b, val))

            changed = False
            editable = _editable_idx()
            if not editable:
                break

            # A) Shrink top contributors for over-upper nutrients
            for nut, lo_b, hi_b, val in sorted(high_nuts, key=lambda x: -(x[3] - x[2])):
                if nut not in df.columns:
                    continue
                # repeatedly shrink the largest contributor
                for _ in range(8):
                    val = float(pd.to_numeric(df[nut], errors="coerce").fillna(0).sum())
                    if val <= hi_b:
                        break
                    editable = _editable_idx()
                    if not editable:
                        break
                    i = max(editable, key=lambda j: float(pd.to_numeric(df.at[j, nut], errors="coerce") or 0.0))
                    g = float(pd.to_numeric(df.at[i, "grams"], errors="coerce") or 0.0)
                    if g <= g_lo + 1e-9:
                        # swap this item to a lower-density candidate
                        fid = int(df.at[i, "food_id"])
                        cands = self.suggest(
                            fid, k=24, mode="avoidance" if avoidance else "code_repair", avoidance=avoidance
                        )
                        # prefer lower amount of this nutrient per calorie
                        scored = []
                        for c in cands:
                            dens = self._raw_dens(c.food_id)
                            scored.append((float(dens.get(nut, 0.0)), c))
                        scored.sort(key=lambda x: x[0])
                        for _, c in scored[:12]:
                            new_g = float(np.clip(self._scale_grams(fid, c.food_id, g), g_lo, g_hi))
                            # try smaller grams for high nutrients
                            for fac in [0.7, 0.5, 1.0, 0.85]:
                                trial = self._try_row_swap(df, i, c.food_id, new_g * fac, target)
                                if trial is None:
                                    # even if full target not met, accept if this nutrient improved and no new worse? 
                                    # HARD: only accept if full day ok OR at least closer on this nut without creating new highs
                                    t2 = df.copy()
                                    t2.loc[i] = self._apply_item_nutrients(t2.loc[i], c.food_id, float(np.clip(new_g * fac, g_lo, g_hi)))
                                    v2 = float(t2[nut].sum())
                                    if v2 < val:
                                        # check no nutrient newly exceeds upper more
                                        ok_partial, _ = self.day_in_target(t2, target)
                                        if ok_partial:
                                            df = t2
                                            changed = True
                                            n_replaced += 1
                                            audit.append(
                                                {
                                                    "row": int(i),
                                                    "old_food_id": fid,
                                                    "new_food_id": c.food_id,
                                                    "method": "hard_enforce_swap",
                                                    "nutrient": nut,
                                                }
                                            )
                                            break
                                else:
                                    df = trial
                                    changed = True
                                    n_replaced += 1
                                    audit.append(
                                        {
                                            "row": int(i),
                                            "old_food_id": fid,
                                            "new_food_id": c.food_id,
                                            "method": "hard_enforce_swap",
                                            "nutrient": nut,
                                        }
                                    )
                                    break
                            if changed:
                                break
                        break
                    new_g = float(np.clip(g * 0.85, g_lo, g_hi))
                    fid = int(df.at[i, "food_id"])
                    df.loc[i] = self._apply_item_nutrients(df.loc[i], fid, new_g)
                    changed = True
                    audit.append(
                        {
                            "row": int(i),
                            "method": "hard_enforce_shrink",
                            "nutrient": nut,
                            "grams_new": new_g,
                        }
                    )

            # B) Grow / swap for under-lower nutrients
            for nut, lo_b, hi_b, val in sorted(low_nuts, key=lambda x: (x[1] - x[3]), reverse=True):
                if nut not in df.columns:
                    continue
                for _ in range(8):
                    val = float(pd.to_numeric(df[nut], errors="coerce").fillna(0).sum())
                    if val >= lo_b:
                        break
                    # grow the row with highest density of this nutrient (skip protected)
                    best_i = None
                    best_dens = -1.0
                    for i in _editable_idx():
                        fid = int(df.at[i, "food_id"])
                        dens = float(self._raw_dens(fid).get(nut, 0.0))
                        if dens > best_dens:
                            best_dens = dens
                            best_i = i
                    if best_i is None:
                        break
                    g = float(pd.to_numeric(df.at[best_i, "grams"], errors="coerce") or 0.0)
                    fid = int(df.at[best_i, "food_id"])
                    if g < g_hi - 1e-9 and best_dens > 0:
                        new_g = float(np.clip(g * 1.12, g_lo, g_hi))
                        trial = self._try_row_swap(df, best_i, fid, new_g, target)
                        if trial is not None:
                            df = trial
                            changed = True
                            audit.append(
                                {
                                    "row": int(best_i),
                                    "method": "hard_enforce_grow",
                                    "nutrient": nut,
                                    "grams_new": new_g,
                                }
                            )
                            continue
                        # grow even if not fully feasible yet, if improves nut and doesn't create new upper violations worse
                        t2 = df.copy()
                        t2.loc[best_i] = self._apply_item_nutrients(t2.loc[best_i], fid, new_g)
                        ok_t, fails_t = self.day_in_target(t2, target)
                        # reject if any upper-bound violation newly appears or worsens sugar-like
                        old_high = {f.split("=")[0] for f in fails if "not in" in f}
                        # simpler: only accept grow if no NEW upper breaches
                        upper_ok = True
                        for n2, b2 in target.items():
                            if n2 not in t2.columns:
                                continue
                            hi2 = float(b2[1])
                            if hi2 >= inf_u:
                                continue
                            if float(t2[n2].sum()) > hi2 + 1e-6:
                                upper_ok = False
                                break
                        if upper_ok and float(t2[nut].sum()) > val:
                            df = t2
                            changed = True
                            audit.append(
                                {
                                    "row": int(best_i),
                                    "method": "hard_enforce_grow_partial",
                                    "nutrient": nut,
                                    "grams_new": new_g,
                                }
                            )
                            continue
                    # swap to higher density food
                    cands = self.suggest(
                        fid, k=24, mode="avoidance" if avoidance else "code_repair", avoidance=avoidance
                    )
                    scored = sorted(
                        ((float(self._raw_dens(c.food_id).get(nut, 0.0)), c) for c in cands),
                        key=lambda x: -x[0],
                    )
                    swapped = False
                    for dens_v, c in scored[:12]:
                        if dens_v <= best_dens:
                            continue
                        new_g = float(np.clip(self._scale_grams(fid, c.food_id, g), g_lo, g_hi))
                        trial = self._try_row_swap(df, best_i, c.food_id, new_g, target)
                        if trial is not None:
                            df = trial
                            changed = True
                            n_replaced += 1
                            swapped = True
                            audit.append(
                                {
                                    "row": int(best_i),
                                    "old_food_id": fid,
                                    "new_food_id": c.food_id,
                                    "method": "hard_enforce_boost_swap",
                                    "nutrient": nut,
                                }
                            )
                            break
                    if not swapped:
                        break

            if not changed:
                break

        ok, fails = self.day_in_target(df, target)
        if not ok:
            fail_reasons.extend(fails[:12])
        return RemapResult(
            day=df,
            ok=ok,
            target_ok=ok,
            n_replaced=n_replaced,
            fail_reasons=fail_reasons,
            audit=audit,
        )

    def remap_day(
        self,
        day: pd.DataFrame,
        target: Optional[dict] = None,
        avoidance: Optional[AvoidanceSpec] = None,
        mode: Optional[str] = None,
    ) -> RemapResult:
        """
        Remap a single day.

        HARD CONSTRAINT (``replacement.hard_target``): every nutrient in ``target``
        must remain within [L, U] after each accepted replacement; final day is
        enforced via ``enforce_target_hard``.
        """
        if mode is None:
            mode = "avoidance" if avoidance is not None else "code_repair"
        rep_cfg = self.cfg["replacement"]
        audit_cfg = self.cfg.get("audit") or {}
        hard = bool(rep_cfg.get("hard_target", True))
        tol_rel, tol_abs, inf_u = self._target_tol()
        orig_col = audit_cfg.get("keep_original_food_id_col", "food_id_original")
        method_col = audit_cfg.get("keep_method_col", "replace_method")
        dist_col = audit_cfg.get("keep_distance_col", "replace_distance")

        df = day.copy().reset_index(drop=True)
        if target is None and "target" in df.columns:
            try:
                target = _parse_target(df["target"].iloc[0])
            except Exception:
                target = {}
        target = target or {}

        if orig_col not in df.columns:
            df[orig_col] = pd.to_numeric(df["food_id"], errors="coerce")
        if method_col not in df.columns:
            df[method_col] = ""
        if dist_col not in df.columns:
            df[dist_col] = np.nan

        audit: list[dict[str, Any]] = []
        fail_reasons: list[str] = []
        n_replaced = 0
        suggest_mode = "avoidance" if mode in ("avoidance", "cross_category") else "code_repair"

        # Snapshot if already in target (for revert)
        baseline_ok, _ = self.day_in_target(df, target) if target else (True, [])
        baseline = df.copy() if baseline_ok else None

        for i in df.index:
            fid = int(pd.to_numeric(df.at[i, "food_id"], errors="coerce"))
            grams = float(pd.to_numeric(df.at[i, "grams"], errors="coerce") or 0.0)

            if suggest_mode == "avoidance":
                assert avoidance is not None
                if not self.is_avoided(fid, avoidance):
                    continue
                cands = self.suggest(fid, k=None, mode="avoidance", avoidance=avoidance)
                if not cands:
                    fail_reasons.append(f"no_avoidance_candidate_for_{fid}")
                    continue
                picked = self._pick_feasible_replacement(df, i, fid, grams, cands, target)
                if picked is None:
                    fail_reasons.append(f"no_feasible_avoidance_for_{fid}_hard_target")
                    continue
                trial, pick, new_g = picked
                df = trial
                df.at[i, method_col] = pick.method
                df.at[i, dist_col] = pick.distance
                n_replaced += 1
                audit.append(
                    {
                        "row": int(i),
                        "old_food_id": fid,
                        "new_food_id": pick.food_id,
                        "method": pick.method,
                        "distance": pick.distance,
                        "grams_old": grams,
                        "grams_new": new_g,
                        "old_desc": self.desc.get(fid, ""),
                        "new_desc": self.desc.get(pick.food_id, ""),
                        "avoidance": avoidance.name,
                        "hard_target": True,
                    }
                )
                continue

            # ---- code_repair path ----
            if not self.needs_replace(fid):
                succ, method = chain_official_successor(fid, self.fped_coded, self.disc_maps)
                if succ is not None and succ != fid and succ in self.candidate_ids:
                    new_g = self._scale_grams(fid, succ, grams)
                    trial = self._try_row_swap(df, i, succ, new_g, target) if hard else None
                    if trial is None and hard:
                        # try factors
                        cands = [
                            ReplacementCandidate(
                                food_id=succ, distance=0.0, method=method,
                                major_group=food_group_code(succ), subgroup2=subgroup2(succ),
                            )
                        ]
                        picked = self._pick_feasible_replacement(df, i, fid, grams, cands, target)
                        if picked is None:
                            fail_reasons.append(f"official_successor_infeasible_{fid}->{succ}")
                            continue
                        trial, pick, new_g = picked
                        method = pick.method
                    elif trial is None:
                        trial = df.copy()
                        trial.loc[i] = self._apply_item_nutrients(trial.loc[i], succ, new_g)
                    df = trial
                    df.at[i, method_col] = method
                    df.at[i, dist_col] = 0.0
                    n_replaced += 1
                    audit.append(
                        {
                            "row": int(i),
                            "old_food_id": fid,
                            "new_food_id": int(df.at[i, "food_id"]),
                            "method": method,
                            "distance": 0.0,
                            "grams_old": grams,
                            "grams_new": float(df.at[i, "grams"]),
                            "old_desc": self.desc.get(fid, ""),
                            "new_desc": self.desc.get(int(df.at[i, "food_id"]), ""),
                            "hard_target": hard,
                        }
                    )
                continue

            cands = self.suggest(fid, k=self.top_k, mode="code_repair")
            if not cands:
                fail_reasons.append(f"no_candidate_for_{fid}")
                continue
            picked = self._pick_feasible_replacement(df, i, fid, grams, cands, target)
            if picked is None:
                fail_reasons.append(f"no_feasible_candidate_for_{fid}_hard_target")
                continue
            trial, pick, new_g = picked
            df = trial
            df.at[i, method_col] = pick.method
            df.at[i, dist_col] = pick.distance
            n_replaced += 1
            audit.append(
                {
                    "row": int(i),
                    "old_food_id": fid,
                    "new_food_id": pick.food_id,
                    "method": pick.method,
                    "distance": pick.distance,
                    "grams_old": grams,
                    "grams_new": new_g,
                    "old_desc": self.desc.get(fid, ""),
                    "new_desc": self.desc.get(pick.food_id, ""),
                    "hard_target": hard,
                }
            )

        target_ok, nut_fails = self.day_in_target(df, target) if target else (True, [])

        if target and (not target_ok) and bool(rep_cfg.get("repair_target", True)):
            enforced = self.enforce_target_hard(df, target=target, avoidance=avoidance)
            df = enforced.day
            audit.extend(enforced.audit)
            n_replaced += enforced.n_replaced
            target_ok, nut_fails = enforced.target_ok, enforced.fail_reasons

        # Final hard gate: if still violated and we had a feasible baseline, revert
        if target and hard:
            target_ok, nut_fails = self.day_in_target(df, target)
            if not target_ok and baseline is not None:
                df = baseline
                fail_reasons.append("reverted_to_in_bounds_baseline")
                target_ok, nut_fails = self.day_in_target(df, target)
            elif not target_ok:
                fail_reasons.extend(nut_fails[:8])

        if "fingerprint_hash" in df.columns:
            fp = food_id_fingerprint(df["food_id"])
            df["fingerprint_hash"] = fingerprint_hash(fp)

        ok = bool(target_ok) and not any(
            r.startswith("no_candidate") or r.startswith("no_avoidance") or r.startswith("no_feasible")
            for r in fail_reasons
        )
        return RemapResult(
            day=df,
            ok=ok,
            target_ok=bool(target_ok),
            n_replaced=n_replaced,
            fail_reasons=fail_reasons,
            audit=audit,
        )

    def remap_frame(
        self,
        df: pd.DataFrame,
        group_cols: Optional[list[str]] = None,
        target_col: str = "target",
        avoidance: Optional[AvoidanceSpec] = None,
        mode: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Remap all days in a Bronze-like dataframe.

        Pass ``avoidance=load_avoidance_preset("dairy")`` for 忌口 cross-category
        replacement.

        Returns
        -------
        remapped_items, day_summary, audit_table
        """
        group_cols = group_cols or ["user_id", "day"]
        work = df.copy()
        for c in group_cols:
            if c not in work.columns:
                raise ValueError(f"missing group col {c}")

        item_parts: list[pd.DataFrame] = []
        summaries: list[dict] = []
        audits: list[dict] = []

        for keys, g in work.groupby(group_cols, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_dict = dict(zip(group_cols, keys))
            tgt = {}
            if target_col in g.columns:
                try:
                    tgt = _parse_target(g[target_col].iloc[0])
                except Exception:
                    tgt = {}
            result = self.remap_day(g, target=tgt, avoidance=avoidance, mode=mode)
            item_parts.append(result.day)
            summaries.append(
                {
                    **key_dict,
                    "ok": result.ok,
                    "target_ok": result.target_ok,
                    "n_replaced": result.n_replaced,
                    "n_items": len(result.day),
                    "fail_reasons": "|".join(result.fail_reasons),
                    "mode": mode
                    or ("avoidance" if avoidance is not None else "code_repair"),
                    "avoidance": avoidance.name if avoidance is not None else "",
                }
            )
            for a in result.audit:
                audits.append({**key_dict, **a})

        remapped = pd.concat(item_parts, ignore_index=True) if item_parts else work.iloc[0:0]
        summary = pd.DataFrame(summaries)
        audit = pd.DataFrame(audits)
        return remapped, summary, audit

"""
Documented hard-constraint set for the Bronze search benchmark.

Source of truth is production archive gate in ``FoodCombinerV2._update_archive``
driven by ``find_bronze_multiobj`` (``require_bld=False``).

Hard (unified evaluator / all algorithms must satisfy)
------------------------------------------------------
1. Nutrient totals in personal [lo, hi] for every key in target_dict.
2. Unique food_id cardinality in [min_foods, max_combination_size].
3. Daily total grams ≤ structure hard max (NHANES p75 / fallback).
4. Condiment/oil/sweet-beverage dominance (``check_condiment_oil_hard``).
5. At most one row per food_id per day (duplicates collapsed then re-checked).
6. Category presence: meal×major cells with band hi≤0 must be empty
   (``category_presence_hard=true``).
7. Breakfast placement via frozen ``bench_placement_allowed`` flags
   (missing desc → allowed; see placement_flags.py / constraint_catalog.yaml).

Soft (reported, never feasibility gates)
----------------------------------------
- Category gram [lo, hi] band deviations (``category_grams_hard=false``).
- Layer-B scalar score.
- B/L/D minimum grams when ``require_bld=False`` (production Bronze driver).

Archive vs fitness mismatch (documented, not silently "fixed")
--------------------------------------------------------------
- Fitness ``fitness==1`` with ``require_bld=False`` only gates card + weight + nutrients.
- Archive additionally rejects condiment hard, breakfast placement, and hi≤0 presence.
- Condiment fitness proxy uses meal grams≥80; archive uses meal kcal≥200.
The unified evaluator matches the **archive** gate (authoritative accept path).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ConstraintSpec:
    """Immutable-ish knobs for the unified evaluator."""

    min_foods: int = 5
    max_combination_size: int = 15
    require_bld: bool = False  # production find_bronze_multiobj default
    hard_daily_weight: bool = True
    max_daily_g: float = 2505.0
    category_presence_hard: bool = True
    category_grams_hard: bool = False  # must stay soft for Bronze
    check_breakfast_placement: bool = True
    check_condiment: bool = True
    # Notes for reports
    notes: list[str] = field(
        default_factory=lambda: [
            "require_bld=False matches find_bronze_multiobj production driver",
            "category_grams_hard=False (soft Layer-B / deviation only)",
            "evaluator mirrors archive hard gates, not fitness==1 alone",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_ga_config(
        cls,
        cfg: dict,
        *,
        structure_prior: Optional[dict] = None,
        require_bld: Optional[bool] = None,
    ) -> "ConstraintSpec":
        from algo.diet_structure_priors import daily_weight_hard_max

        st = cfg.get("structure") or {}
        max_g = daily_weight_hard_max(structure_prior, cfg)
        return cls(
            min_foods=int(cfg.get("min_foods", 5)),
            max_combination_size=int(cfg.get("max_combination_size", 15)),
            require_bld=bool(False if require_bld is None else require_bld),
            hard_daily_weight=bool(st.get("hard_daily_weight", True)),
            max_daily_g=float(max_g),
            category_presence_hard=bool(st.get("category_presence_hard", True)),
            category_grams_hard=bool(st.get("category_grams_hard", False)),
            check_breakfast_placement=True,
            check_condiment=True,
        )


# Scales for normalizing hard-violation magnitudes in GA penalties / reporting.
VIOLATION_SCALES: dict[str, float] = {
    "nutrient": 100.0,  # grams / mg-ish sum of abs overshoot
    "cardinality": 1.0,
    "daily_weight": 100.0,  # grams over max
    "bld": 50.0,  # grams shortfall
    "condiment": 1.0,  # binary-ish fail count
    "duplicate_food": 1.0,
    "category_presence": 1.0,
    "breakfast_placement": 1.0,
}

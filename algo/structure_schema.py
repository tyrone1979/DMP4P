"""
Declarative schema for learning diet *structures* from arbitrary features.

Default instance today: person = sex × bmi_class × age_band;
cell axes = meal × FNDDS major (grams).

Extension points (not wired to data yet — keep columns / dims reserved):
  - person: race / ethnicity → e.g. Latin-American strata
  - food cell: cuisine (Mexican, …) → e.g. g_cuisine_mexican_dinner

Any new dim is: (1) list it in YAML ``structure_schema``, (2) provide a
labeler, (3) emit matching feature columns when building day frames.
Fit / resolve / score stay generic over ``feature_names`` + ``stratum_key``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from algo.structure_features import (
    MAJOR_CODES,
    MEAL_MAJOR_GRAM_FEATURES,
    MEAL_NAMES,
    age_band_label,
    bmi_class_label,
    sex_label_stratum,
)

Labeler = Callable[[Any], str]


@dataclass(frozen=True)
class PersonDim:
    """One demographic (or preference) axis used to stratify training days."""

    name: str
    labeler: Labeler
    # Columns consulted on a feature/demo row (first present wins for raw value).
    source_cols: tuple[str, ...] = ()
    # Reserved / not used in default fit until data + labeler are ready.
    optional: bool = False
    notes: str = ""


@dataclass(frozen=True)
class CellAxis:
    """
    One food-side axis of the structure grid (crossed with meal by default).

    ``feature_template`` uses ``{meal}`` and ``{code}`` placeholders.
    Example majors: ``g_major_{code}_{meal}``
    Future cuisine: ``g_cuisine_{code}_{meal}``
    """

    name: str
    codes: tuple[Any, ...]
    feature_template: str
    optional: bool = False
    notes: str = ""


@dataclass
class StructureSchema:
    """Full recipe for fit → personal resolve → Hard/soft score."""

    name: str
    version: str
    person_dims: list[PersonDim]
    cell_axes: list[CellAxis]
    meals: tuple[str, ...] = MEAL_NAMES
    # Day-level mass refs used for g_scale (not Hard cells).
    mass_features: tuple[str, ...] = ("total_g", "total_kcal")
    # Also emit day-aggregate features for each cell axis (e.g. g_major_5).
    emit_day_aggregates: bool = True
    notes: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def active_person_dims(self) -> list[PersonDim]:
        return [d for d in self.person_dims if not d.optional]

    def active_cell_axes(self) -> list[CellAxis]:
        return [a for a in self.cell_axes if not a.optional]

    def band_feature_names(self) -> list[str]:
        """Quantile axes: meal×cell (+ optional day aggregates) + mass refs."""
        names: list[str] = []
        for ax in self.active_cell_axes():
            for meal in self.meals:
                for code in ax.codes:
                    names.append(ax.feature_template.format(meal=meal, code=code))
            if self.emit_day_aggregates:
                # day aggregate: strip _{meal} suffix pattern → g_major_{code}
                for code in ax.codes:
                    # templates like g_major_{code}_{meal} → g_major_{code}
                    day_t = ax.feature_template.replace("_{meal}", "").replace("{meal}", "")
                    names.append(day_t.format(code=code))
        for m in self.mass_features:
            if m not in names:
                names.append(m)
        # de-dupe preserve order
        seen: set[str] = set()
        out: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def hard_cell_features(self) -> list[str]:
        """Meal×cell features used as personal Hard / presence axes."""
        names: list[str] = []
        for ax in self.active_cell_axes():
            for meal in self.meals:
                for code in ax.codes:
                    names.append(ax.feature_template.format(meal=meal, code=code))
        return names

    def stratum_key_from_row(self, row: pd.Series | dict) -> str:
        get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
        parts: list[str] = []
        for dim in self.active_person_dims():
            raw = None
            for c in dim.source_cols or (dim.name,):
                v = get(c, None)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    raw = v
                    break
            # Prefer precomputed band/class columns when present
            if dim.name == "bmi_class":
                pre = get("bmi_class", None)
                if pre is not None and str(pre) not in ("", "NA", "nan"):
                    parts.append(_normalize_bmi_class(str(pre)))
                    continue
            if dim.name == "age_band":
                pre = get("age_band", None)
                if pre is not None and str(pre) not in ("", "NA", "nan"):
                    parts.append(_normalize_age_band(str(pre)))
                    continue
            parts.append(str(dim.labeler(raw)))
        return "|".join(parts)

    def to_meta(self) -> dict[str, Any]:
        """JSON-serializable schema stub stored inside prior.json."""
        return {
            "name": self.name,
            "version": self.version,
            "person_dims": [
                {
                    "name": d.name,
                    "source_cols": list(d.source_cols),
                    "optional": d.optional,
                    "notes": d.notes,
                }
                for d in self.person_dims
            ],
            "cell_axes": [
                {
                    "name": a.name,
                    "codes": list(a.codes),
                    "feature_template": a.feature_template,
                    "optional": a.optional,
                    "notes": a.notes,
                }
                for a in self.cell_axes
            ],
            "meals": list(self.meals),
            "mass_features": list(self.mass_features),
            "band_feature_names": self.band_feature_names(),
            "hard_cell_features": self.hard_cell_features(),
            "notes": self.notes,
        }


def _normalize_bmi_class(bmi_c: str) -> str:
    alias = {
        "under": "underweight (<18.5)",
        "normal": "normal (18.5–24.9)",
        "over": "overweight (25–29.9)",
        "obese": "obesity (≥30)",
        "obesity": "obesity (≥30)",
        "overweight": "overweight (25–29.9)",
        "underweight": "underweight (<18.5)",
    }
    return alias.get(str(bmi_c).strip().lower(), str(bmi_c))


def _normalize_age_band(age_b: str) -> str:
    alias_age = {
        "19-29": "19–30",
        "30-49": "31–50",
        "50-64": "51–70",
        "65+": "71+",
    }
    return alias_age.get(str(age_b).strip(), str(age_b))


def _label_sex(v) -> str:
    return sex_label_stratum(v)


def _label_bmi_class(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NA"
    # raw BMI float vs already a class string
    try:
        return bmi_class_label(float(v))
    except (TypeError, ValueError):
        return _normalize_bmi_class(str(v))


def _label_age_band(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NA"
    try:
        return age_band_label(float(v))
    except (TypeError, ValueError):
        return _normalize_age_band(str(v))


def _label_passthrough(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NA"
    s = str(v).strip()
    return s or "NA"


def default_category_schema() -> StructureSchema:
    """Current production schema: sex×BMI×age × (meal×FNDDS major grams)."""
    return StructureSchema(
        name="category_grams_meal",
        version="category_grams_meal_v1",
        person_dims=[
            PersonDim("sex", _label_sex, source_cols=("sex", "gender")),
            PersonDim("bmi_class", _label_bmi_class, source_cols=("bmi", "bmi_class")),
            PersonDim("age_band", _label_age_band, source_cols=("age", "age_band")),
            # Future person axes — optional until cohort columns exist
            PersonDim(
                "race",
                _label_passthrough,
                source_cols=("race", "ethnicity", "RIDRETH3"),
                optional=True,
                notes="e.g. Latin-American / Hispanic strata when NHANES race wired",
            ),
        ],
        cell_axes=[
            CellAxis(
                name="major",
                codes=tuple(MAJOR_CODES),
                feature_template="g_major_{code}_{meal}",
                notes="FNDDS one-digit food group 1–9",
            ),
            CellAxis(
                name="cuisine",
                codes=(),  # populate when cuisine tags exist
                feature_template="g_cuisine_{code}_{meal}",
                optional=True,
                notes="e.g. mexican under Latin-American person strata — not fit yet",
            ),
        ],
        meals=MEAL_NAMES,
        notes=(
            "Default Bronze structure. Add person race + food cuisine axes in YAML "
            "and drop optional=False when feature columns are available."
        ),
    )


_ACTIVE: Optional[StructureSchema] = None


def get_structure_schema(schema: StructureSchema | None = None) -> StructureSchema:
    global _ACTIVE
    if schema is not None:
        _ACTIVE = schema
        return _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = default_category_schema()
    return _ACTIVE


def stratum_key_generic(row: pd.Series | dict, schema: StructureSchema | None = None) -> str:
    return get_structure_schema(schema).stratum_key_from_row(row)


def pick_stratum_envelope(
    strata: dict[str, Any],
    row_or_key: pd.Series | dict | str,
    *,
    schema: StructureSchema | None = None,
    overall: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Match full stratum key; else drop optional/trailing person dims from the end
    (age → bmi → sex) picking the largest-n partial match; else overall.
    """
    sch = get_structure_schema(schema)
    if isinstance(row_or_key, str):
        key = row_or_key
    else:
        key = sch.stratum_key_from_row(row_or_key)
    if key in strata:
        return strata[key], key
    parts = key.split("|")
    # Progressive drop from the last active dim
    n_dims = len(sch.active_person_dims())
    for drop in range(1, n_dims):
        prefix = "|".join(parts[: n_dims - drop])
        if not prefix:
            break
        candidates = [k for k in strata if k.startswith(prefix + "|") or k == prefix]
        if not candidates:
            # also allow infix match on remaining middle dims (legacy BMI-class)
            continue

        def _n(k: str) -> int:
            band = strata[k]
            tg = (band.get("total_g") or {}) if isinstance(band, dict) else {}
            return int(tg.get("n") or 0)

        best = max(candidates, key=_n)
        return strata[best], best
    if overall is not None:
        return overall, "overall"
    return {}, "overall"


def assert_features_cover_schema(
    columns: Sequence[str],
    schema: StructureSchema | None = None,
) -> list[str]:
    """Return missing band feature names (empty if ok)."""
    need = set(get_structure_schema(schema).band_feature_names())
    have = set(columns)
    return sorted(need - have)


# Back-compat: meal×major list equals default hard cells
assert list(MEAL_MAJOR_GRAM_FEATURES) == default_category_schema().hard_cell_features()

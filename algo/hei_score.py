"""
HEI-2020 day score (component standards identical to HEI-2015).

Status: stub — returns None until FPED aggregation + NCI scoring is implemented.
Silver Soft calls ``score_day_hei``; missing scores only set soft_flags, never reject.

HEI-2025: not released by USDA/NCI yet; bump ``version`` when official macros ship.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def score_day_hei(
    day: pd.DataFrame,
    version: str = "HEI-2020",
    energy_kcal: Optional[float] = None,
) -> Optional[float]:
    """
    Return HEI total score in [0, 100], or None if unscored.

    Planned implementation:
      1) Map food_id → FPED equivalents (g/100g) with discontinued-code chain
      2) Sum day components + energy
      3) Apply HEI-2015/2020 density standards (NCI hei2015.score.macro.sas)
    """
    _ = (day, version, energy_kcal)
    return None

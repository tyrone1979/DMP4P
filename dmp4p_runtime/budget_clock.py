"""
Shared wall-clock for E1a: one deadline covers init + search + decode + validate.

Does not exclude init/sync/eval to “meet” the budget.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BudgetClock:
    wall_s: float
    t_start: float = field(default_factory=time.perf_counter)
    first_verified_feasible: Optional[float] = None  # absolute perf_counter
    last_search_end: Optional[float] = None
    finalization_end: Optional[float] = None

    def __post_init__(self) -> None:
        self.wall_s = float(self.wall_s)
        self.t_start = float(self.t_start)

    @property
    def deadline(self) -> float:
        return self.t_start + self.wall_s

    def now(self) -> float:
        return time.perf_counter()

    def elapsed(self) -> float:
        return self.now() - self.t_start

    def in_budget(self, at: Optional[float] = None) -> bool:
        t = self.now() if at is None else float(at)
        return t <= self.deadline + 1e-12

    def mark_verified_feasible(self, at: Optional[float] = None) -> float:
        t = self.now() if at is None else float(at)
        if self.first_verified_feasible is None:
            self.first_verified_feasible = t
        return t

    def mark_search_end(self, at: Optional[float] = None) -> float:
        t = self.now() if at is None else float(at)
        self.last_search_end = t
        return t

    def mark_finalization_end(self, at: Optional[float] = None) -> float:
        t = self.now() if at is None else float(at)
        self.finalization_end = t
        return t

    def budget_overflow_s(self) -> float:
        end = self.finalization_end if self.finalization_end is not None else self.now()
        return max(0.0, float(end) - self.deadline)

    def rel(self, absolute: Optional[float]) -> Optional[float]:
        if absolute is None:
            return None
        return float(absolute) - self.t_start

    def search_after_first_verified_s(self) -> Optional[float]:
        """Wall span from first in-budget verified feasible to last search end (may include overflow)."""
        if self.first_verified_feasible is None or self.last_search_end is None:
            return None
        return max(0.0, float(self.last_search_end) - float(self.first_verified_feasible))

    def search_after_first_verified_within_deadline_s(self) -> Optional[float]:
        """
        Wall span from first in-budget verified feasible to min(last_search_end, deadline).
        Not a precise exclusive CPU/GPU phase timer — wall-clock span only.
        """
        if self.first_verified_feasible is None:
            return None
        end = self.last_search_end if self.last_search_end is not None else self.now()
        capped = min(float(end), float(self.deadline))
        return max(0.0, capped - float(self.first_verified_feasible))

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_s": self.wall_s,
            "t_start": self.t_start,
            "deadline": self.deadline,
            "first_verified_feasible": self.first_verified_feasible,
            "first_verified_feasible_rel_s": self.rel(self.first_verified_feasible),
            "last_search_end": self.last_search_end,
            "last_search_end_rel_s": self.rel(self.last_search_end),
            "finalization_end": self.finalization_end,
            "finalization_end_rel_s": self.rel(self.finalization_end),
            "budget_overflow_s": self.budget_overflow_s(),
            "search_after_first_verified_s": self.search_after_first_verified_s(),
            "search_after_first_verified_within_deadline_s": (
                self.search_after_first_verified_within_deadline_s()
            ),
            "accounting": (
                "single_deadline_from_solve_start_covers_init_search_decode_validate; "
                "no exclusion of init/sync/eval to meet budget; "
                "time spans are wall-clock intervals, not exclusive CPU/GPU phase timers"
            ),
        }

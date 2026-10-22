"""
Pluggable N+1 pool-ban for sequential Bronze search.

Plugins decide which food_ids to remove after solution N before searching N+1.
Default production plugin is ``fndds`` (see ``config/ga_multiobj_weights.yaml``).
Use ``none`` for coverage ablation (no pool shrink).
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol, runtime_checkable

import pandas as pd

PLUGIN_NONE = "none"
PLUGIN_FNDDS = "fndds"
KNOWN_PLUGINS = (PLUGIN_NONE, PLUGIN_FNDDS)


@runtime_checkable
class SequentialBanPlugin(Protocol):
    name: str
    enabled: bool

    def ban_ids_after(
        self,
        day: pd.DataFrame,
        *,
        pool_food_ids: Optional[Iterable[int]] = None,
    ) -> set[int]:
        """Return food_ids to exclude from the pool for the next solution."""
        ...


class NoOpBanPlugin:
    """Ablation baseline: never shrink the pool."""

    name = PLUGIN_NONE
    enabled = False

    def ban_ids_after(
        self,
        day: pd.DataFrame,
        *,
        pool_food_ids: Optional[Iterable[int]] = None,
    ) -> set[int]:
        return set()


class FnddsDayBanPlugin:
    """
    Ban food_ids from solution N (FNDDS policy in config/fndds_food_code_policy.yaml).

    Default policy: exact day food_ids (+ exact-ban majors); optional family expand.
    """

    name = PLUGIN_FNDDS
    enabled = True

    def __init__(self, policy_cfg: Optional[dict] = None):
        self.policy_cfg = policy_cfg

    def ban_ids_after(
        self,
        day: pd.DataFrame,
        *,
        pool_food_ids: Optional[Iterable[int]] = None,
    ) -> set[int]:
        from algo.fndds_code_policy import collect_sequential_ban_food_ids

        return collect_sequential_ban_food_ids(
            day,
            pool_food_ids=pool_food_ids,
            cfg=self.policy_cfg,
            include_mains=True,
        )


def _seq_cfg(config: Optional[dict]) -> dict[str, Any]:
    return dict((config or {}).get("sequential_ban") or {})


def resolve_ban_plugin_name(config: Optional[dict] = None) -> str:
    """
    Resolve plugin id from GA / run config.

    Priority:
      1. sequential_ban.plugin
      2. if sequential_ban.enabled or sequential_ban_mains → fndds (legacy)
      3. else → none
    """
    cfg = config or {}
    seq = _seq_cfg(cfg)
    raw = seq.get("plugin")
    if raw is not None and str(raw).strip():
        name = str(raw).strip().lower()
        if name in ("off", "disabled", "noop", "null"):
            return PLUGIN_NONE
        return name
    if bool(seq.get("enabled")) or bool(cfg.get("sequential_ban_mains")):
        return PLUGIN_FNDDS
    return PLUGIN_NONE


def build_sequential_ban_plugin(config: Optional[dict] = None) -> SequentialBanPlugin:
    """Factory: ``fndds`` (production) or ``none`` (ablation)."""
    name = resolve_ban_plugin_name(config)
    if name == PLUGIN_NONE:
        return NoOpBanPlugin()
    if name == PLUGIN_FNDDS:
        return FnddsDayBanPlugin()
    raise ValueError(
        f"unknown sequential_ban.plugin={name!r}; expected one of {KNOWN_PLUGINS}"
    )


def sequential_search_enabled(config: Optional[dict] = None) -> bool:
    """True when the one-by-one sequential optimizer should run (ban plugin optional)."""
    cfg = config or {}
    seq = _seq_cfg(cfg)
    if "enabled" in seq:
        return bool(seq.get("enabled"))
    if bool(cfg.get("sequential_ban_mains")):
        return True
    # Explicit plugin other than none implies sequential search for ablation runs
    plugin = resolve_ban_plugin_name(cfg)
    return plugin != PLUGIN_NONE

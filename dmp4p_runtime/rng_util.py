"""Seed helpers — document what is and is not controlled."""
from __future__ import annotations

import random
from typing import Any

import numpy as np


def seed_all(seed: int, *, device: str = "cpu") -> dict[str, Any]:
    """
    Seed Python, NumPy, and PyTorch RNGs.

    Notes (recorded for reproducibility claims):
    - PyTorch CUDA ops are not fully deterministic unless
      ``torch.use_deterministic_algorithms(True)`` and cuBLAS workspace is set;
      that path can break some kernels used by production V2, so we only seed.
    - ``secrets`` / OS entropy used inside production cell-pool explore is
      bypassed when snapshots pass ``random_state``.
    - OR-Tools CP-SAT uses ``parameters.random_seed`` separately.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    notes = [
        "python.random seeded",
        "numpy.random seeded",
        "torch seeded if available; CUDA not forced deterministic",
    ]
    try:
        import torch

        torch.manual_seed(seed)
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        notes.append(f"torch.manual_seed({seed})")
    except Exception as e:
        notes.append(f"torch_seed_skip:{e}")
    return {"seed": seed, "notes": notes}


def make_numpy_generator(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

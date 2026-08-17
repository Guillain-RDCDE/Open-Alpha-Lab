"""Shared fixtures — deterministic synthetic tapes for Study 954 (High Yield in Disguise).

Two fixtures, both offline and deterministic (fixed seed 954; no network). In both, the
synthetic credit fund is genuinely ``w_true * equity + (1 - w_true) * duration`` plus an
idiosyncratic credit shock of the same size — only the *price* of that shock differs:

- ``uncompensated`` — ``signal_strength=1``: the shock carries a 3%/yr give-up, so the
  held-out replication must win the vol-matched race.
- ``fairly_paid`` — ``signal_strength=0``: the shock earns exactly the premium that keeps
  the fund's Sharpe level with the replication's (the null — the race must come out flat).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hy_replication import data  # noqa: E402


@pytest.fixture
def uncompensated():
    """A credit fund whose idiosyncratic risk is paid nothing — the planted effect."""
    return data.synthetic_panel(signal_strength=1.0, seed=954)


@pytest.fixture
def fairly_paid():
    """A credit fund whose idiosyncratic risk is fairly paid — the null."""
    return data.synthetic_panel(signal_strength=0.0, seed=954)

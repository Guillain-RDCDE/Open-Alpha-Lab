"""Shared fixtures for Study 1027 — deterministic, offline, no network.

Three synthetic worlds from one GJR generator with one knob moved: ``signal_strength = +1``
plants the **equity** leverage effect, ``-1`` plants the **inverse** (oil-claim) effect at the
same size, ``0`` is the symmetric null. Unconditional volatility, persistence and tails are
identical across the three, so any estimator that separates them is reading the asymmetry and
nothing else. A test that passes on the planted world and *also* on the null tests nothing.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mirrorlev import data  # noqa: E402

N_YEARS = 12


def _returns(k):
    px, truth = data.synthetic_gjr(n_years=N_YEARS, signal_strength=k, seed=1027)
    return px, data.log_returns(px), truth


@pytest.fixture(scope="session")
def equity_world():
    """Planted equity-type leverage (down shocks raise variance more)."""
    return _returns(1.0)


@pytest.fixture(scope="session")
def mirror_world():
    """Planted inverse leverage — the oil claim, at the same size."""
    return _returns(-1.0)


@pytest.fixture(scope="session")
def null_world():
    """The matching symmetric null."""
    return _returns(0.0)

"""Shared fixtures — deterministic synthetic daily price tapes for Study 110.

Two fixtures:

- ``flat_vol`` — a single-regime tape (``signal_strength=0``): the SMA rule has nothing
  to read, so any advantage it shows is noise.
- ``two_regime`` — a two-state bull/bear tape (``signal_strength=1``): genuine regime
  separation the 200-day SMA *can* detect. The cross beats random on this tape.

Both fixtures are offline and deterministic (fixed seed 110). No network is touched.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from faber_timing import data  # noqa: E402


@pytest.fixture
def flat_vol():
    """A single-regime tape (no timing signal to read) — the null control."""
    prices, truth = data.synthetic_daily(n_years=20, signal_strength=0.0, seed=110)
    return prices, truth


@pytest.fixture
def two_regime():
    """A two-state bull/bear tape with genuine regime structure the SMA can exploit."""
    prices, truth = data.synthetic_daily(n_years=30, signal_strength=1.0, seed=110)
    return prices, truth

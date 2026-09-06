"""Shared fixtures for Study 963 — deterministic, offline, no network.

Two synthetic worlds, both built from ``data.synthetic_ohlc`` and then run through
``strategy.plant_half_days``: one where a known bump *and* a volume drought are planted on
a known set of dates (the world where the claim is true), one where only the volume drought
is planted and the return bump is zero (the null — thin sessions that are otherwise ordinary).

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from half_day import data  # noqa: E402

N_YEARS = 12


@pytest.fixture
def planted():
    """Bars with clustered volatility (signal_strength=1) — the interesting world."""
    return data.synthetic_ohlc(n_years=N_YEARS, signal_strength=1.0, seed=963)


@pytest.fixture
def null_bars():
    """The matching null: the same machinery with volatility held constant."""
    return data.synthetic_ohlc(n_years=N_YEARS, signal_strength=0.0, seed=963)


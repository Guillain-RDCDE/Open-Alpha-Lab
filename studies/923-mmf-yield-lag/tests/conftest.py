"""Shared fixtures — deterministic synthetic panels for Study 923 (The Cash Lag).

Three fixtures, all offline and deterministic (fixed seed 923; no network). Two
independent things are planted — a **WAM ladder** (the measurement effect) and a
**trending rate path** (the tradability effect) — so the fixtures switch them on
and off separately:

- ``laddered`` — a real WAM ladder *and* a trending rate (``signal_strength=1``).
  Both halves of the study must fire: durations order with WAM, and the
  rate-direction switch rule earns a positive gross excess.
- ``random_walk_ladder`` — the same real ladder but ``trend_phi=0``, so rate
  changes are unpredictable. The measurement must still be exact (this is the
  cleanest recovery test, with no autocorrelated regressor to bias OLS) while the
  switch rule must find nothing. It is the synthetic twin of what the real tape
  turns out to look like.
- ``flat_panel`` — the null (``signal_strength=0``): every vehicle collapses to the
  same WAM (no duration ordering left to find) and the rate path loses its trend.
  Each vehicle still carries its own small NAV noise, so this is a genuine noise
  world, not a degenerate one in which the series are literally identical.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_lag import data  # noqa: E402


@pytest.fixture
def laddered():
    """A panel with a real WAM ladder and a trending rate — both effects planted."""
    return data.synthetic_panel(signal_strength=1.0, seed=923)


@pytest.fixture
def random_walk_ladder():
    """A real WAM ladder under an unpredictable rate path: measurable, untradable."""
    return data.synthetic_panel(signal_strength=1.0, trend_phi=0.0, seed=923)


@pytest.fixture
def flat_panel():
    """The null: one shared WAM, an unpredictable rate path, vehicle noise only."""
    return data.synthetic_panel(signal_strength=0.0, seed=923)

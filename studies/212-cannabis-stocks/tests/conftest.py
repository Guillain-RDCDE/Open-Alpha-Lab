"""Shared fixtures — deterministic synthetic SPY/cannabis price pairs for offline testing.

Two fixtures cover the study's two hypotheses:

- ``wealth_shredder`` — a tape where cannabis has near-market beta and strongly negative alpha.
  Tests the beta/alpha estimation and the core wealth-destruction thesis.
- ``null_tape`` — a tape where cannabis has zero beta (leverage=0, no signal).
  The timing rule must not show a reliable edge here.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cannabis_stocks import data  # noqa: E402


@pytest.fixture
def wealth_shredder():
    """A tape with beta=1.0, negative alpha=-0.30, high idio vol — the cannabis thesis."""
    prices, truth = data.synthetic_daily(
        n_years=8,
        beta=1.0,
        alpha_ann=-0.30,
        idio_vol_ann=0.50,
        seed=212,
    )
    return prices, truth


@pytest.fixture
def null_tape():
    """A tape where cannabis is uncorrelated with SPY (beta=0) — the null for timing."""
    prices, truth = data.synthetic_daily(
        n_years=8,
        beta=0.0,
        alpha_ann=0.0,
        idio_vol_ann=0.30,
        seed=999,
    )
    return prices, truth


@pytest.fixture
def leveraged():
    """A tape with planted beta=1.5, positive alpha — a hypothetical positive scenario."""
    prices, truth = data.synthetic_daily(
        n_years=8,
        beta=1.5,
        alpha_ann=0.10,
        idio_vol_ann=0.30,
        seed=42,
    )
    return prices, truth

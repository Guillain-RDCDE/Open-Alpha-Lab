"""Shared fixtures — deterministic synthetic GLD/GDX price pairs for offline testing.

Two fixtures cover the study's two hypotheses:

- ``leveraged`` — a tape where GDX has a planted beta > 1 to GLD and negative alpha.
  Tests the beta/alpha estimation and the asymmetry claim.
- ``null_tape`` — a tape where GDX has zero beta (leverage=0, no signal in the ratio).
  The timing rule must not show a reliable edge here.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gold_miners import data  # noqa: E402


@pytest.fixture
def leveraged():
    """A tape with planted beta=1.5, asymmetry=0.3, negative alpha — the realistic scenario."""
    prices, truth = data.synthetic_daily(
        n_years=15,
        leverage=1.5,
        asymmetry=0.3,
        alpha_ann=-0.04,
        seed=208,
    )
    return prices, truth


@pytest.fixture
def null_tape():
    """A tape where GDX is uncorrelated with GLD (leverage=0) — the null for the timing rule."""
    prices, truth = data.synthetic_daily(
        n_years=15,
        leverage=0.0,
        asymmetry=0.0,
        alpha_ann=0.0,
        seed=999,
    )
    return prices, truth


@pytest.fixture
def symmetric():
    """A tape where GDX is leveraged but symmetric (no extra downside beta)."""
    prices, truth = data.synthetic_daily(
        n_years=15,
        leverage=1.5,
        asymmetry=0.0,
        alpha_ann=0.0,
        seed=42,
    )
    return prices, truth

"""Shared fixtures — deterministic synthetic panels for Study 936 (Tolerance Bands).

Two fixtures, both offline and deterministic (fixed seed 936; no network):

- ``reverting`` — two equal-vol sleeves whose *dispersion* mean-reverts with a 60-day
  half-life (``signal_strength=1``). Selling the winner and buying the loser is
  genuinely rewarded, so a dispersion-triggered (band) rule should beat a calendar
  rule: the planted world.
- ``random_walk`` — the same two sleeves with no reversion at all
  (``signal_strength=0``). Dispersion is permanent, nothing is harvestable, and every
  rebalancing schedule should look alike gross and slightly worse net: the null.

Both are short by design (12 years) so the suite stays fast; the seed-robustness
checks that need the full horizon build their own panels.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rebal_bands import data  # noqa: E402


@pytest.fixture(scope="session")
def reverting():
    """Planted world: dispersion mean-reverts, so band rebalancing has something to eat."""
    return data.synthetic_panel(n_years=12, signal_strength=1.0, seed=936)


@pytest.fixture(scope="session")
def random_walk():
    """Null world: dispersion is a random walk — no rebalancing schedule can win."""
    return data.synthetic_panel(n_years=12, signal_strength=0.0, seed=936)

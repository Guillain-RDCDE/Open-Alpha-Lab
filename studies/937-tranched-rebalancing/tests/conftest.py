"""Shared fixtures — deterministic synthetic tapes for Study 937 (Tranches).

Both fixtures are offline and deterministic (fixed seed 937; no network, no cache):

- ``regime`` — a two-state tape with a long, deep bear regime (``signal_strength=1``)
  that a monthly 200-day trend sleeve can genuinely duck: the planted effect.
- ``flat`` — a single-regime tape (``signal_strength=0``): the sleeve has nothing to
  read, so it must not beat an exposure-matched random control. The *timing-luck cone*
  must nevertheless still be there — it is an artefact of the arbitrary rebalance date,
  not a symptom of edge.

Thirty-two synthetic years are used so the planted effect is not swamped by regime-draw
noise while the suite still runs quickly.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tranching import data  # noqa: E402

N_YEARS = 32


@pytest.fixture
def regime():
    """A tape with a real, trend-followable bear regime (the planted effect)."""
    return data.synthetic_daily(n_years=N_YEARS, signal_strength=1.0, seed=937)


@pytest.fixture
def flat():
    """A single-regime null tape — no edge for the sleeve to find."""
    return data.synthetic_daily(n_years=N_YEARS, signal_strength=0.0, seed=937)

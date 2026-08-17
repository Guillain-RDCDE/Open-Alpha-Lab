"""Shared fixtures — deterministic synthetic panels for Study 957 (Holdco Discount).

Both fixtures are offline and deterministic (fixed seed 957; no network, no cache):

- ``ou_panel`` — seven synthetic (stake, holdco) pairs whose discount is a genuine
  mean-reverting Ornstein-Uhlenbeck process (``signal_strength=1``). The planted world: the
  gap really does close, so both tests must fire.
- ``rw_panel`` — the same seven pairs with the pull switched off (``signal_strength=0``), so
  the discount is a martingale random walk. The null: neither test may fire.

A small ``n_years`` keeps the whole suite fast while leaving enough history for the 504-day
trailing standardisation plus the 126-day forward window.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from holdco_nav import data  # noqa: E402

N_YEARS = 12
N_NAMES = 7


@pytest.fixture(scope="session")
def ou_panel():
    """Planted: a genuinely mean-reverting discount the rule should be able to harvest."""
    return data.synthetic_panel(n_names=N_NAMES, signal_strength=1.0, seed=957,
                                n_years=N_YEARS)


@pytest.fixture(scope="session")
def rw_panel():
    """The null: a martingale discount with nothing to harvest."""
    return data.synthetic_panel(n_names=N_NAMES, signal_strength=0.0, seed=957,
                                n_years=N_YEARS)


@pytest.fixture(scope="session")
def one_pair():
    """A single planted pair, for the per-series unit tests."""
    return data.synthetic_daily(signal_strength=1.0, seed=957, n_years=N_YEARS)

"""Shared fixtures — deterministic synthetic panels for Study 940 (The Turnover Budget).

Two fixtures, both offline and deterministic (fixed seed 940; no network):

- ``planted`` — a cross-sectional panel whose expected returns carry a persistent,
  slowly-decaying component (``signal_strength=1``), so a 12-1 momentum rank reads
  something real and a *faster* rebalance tracks it better. This is the world in which
  a turnover budget is a meaningful question.
- ``null_panel`` — the same machinery with the persistent component switched off
  (``signal_strength=0``): a market factor plus pure idiosyncratic noise. No rebalance
  speed may earn anything here.

Both are small (12 years, 8 names) so the whole suite stays fast.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from turnover_budget import data  # noqa: E402

N_YEARS = 12
N_ASSETS = 8


@pytest.fixture
def planted():
    """A panel with genuine, persistent cross-sectional momentum to be harvested."""
    prices, cash, truth = data.synthetic_panel(
        n_assets=N_ASSETS, n_years=N_YEARS, signal_strength=1.0, seed=940
    )
    return prices, cash, truth


@pytest.fixture
def null_panel():
    """A no-alpha panel — market factor plus idiosyncratic noise only (the null)."""
    prices, cash, truth = data.synthetic_panel(
        n_assets=N_ASSETS, n_years=N_YEARS, signal_strength=0.0, seed=940
    )
    return prices, cash, truth

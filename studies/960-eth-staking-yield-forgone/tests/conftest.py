"""Shared fixtures — deterministic synthetic panels for Study 960 (The Unstaked ETF).

Both fixtures are offline and deterministic (fixed seed 960; no network, no cache):

- ``planted`` — one coin, a mis-timed observed coin close, and two wrappers separated
  by a planted 2.25%/yr drag spread (``signal_strength=1``). The fund-vs-fund estimator
  must recover that spread; the fund-vs-coin estimator must be far too wide to see it.
- ``null_panel`` — the same world with ``signal_strength=0``: the two wrappers carry
  identical drags, so the fund-vs-fund estimate must collapse to ~0.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from unstaked import data  # noqa: E402


@pytest.fixture
def planted():
    """Two wrappers separated by a known drag spread, observed through a mis-timed coin."""
    prices, truth = data.synthetic_panel(signal_strength=1.0, seed=960)
    return prices, truth


@pytest.fixture
def null_panel():
    """Identical wrappers — the fund-vs-fund estimator must read ~0 (the null)."""
    prices, truth = data.synthetic_panel(signal_strength=0.0, seed=960)
    return prices, truth

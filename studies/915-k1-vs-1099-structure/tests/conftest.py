"""Shared fixtures — deterministic synthetic wrapper pairs for Study 915.

Both fixtures are offline and deterministic (fixed seed 915; no network, no cache):

- ``planted`` — two wrappers on a shared commodity factor where the 1099 leg carries a
  genuine, known annual drag (``signal_strength=1``). The race must recover it.
- ``null`` — the same construction with the drag switched off (``signal_strength=0``):
  economically identical wrappers separated only by tracking noise. The race must find
  nothing, and must not manufacture a fee or tax story out of the noise.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wrapper_tax import data  # noqa: E402


@pytest.fixture
def planted():
    """A wrapper pair with a real, known annual drag on the 1099 leg."""
    return data.synthetic_daily(signal_strength=1.0, seed=915)


@pytest.fixture
def null():
    """A wrapper pair that differs by tracking noise only (the null)."""
    return data.synthetic_daily(signal_strength=0.0, seed=915)


@pytest.fixture
def year_table(planted):
    """A complete-calendar-year table built off the synthetic pair, for the tax model."""
    from wrapper_tax import strategy as st
    prices, _ = planted
    return st.calendar_year_table(prices, "ric", "k1", "cash")

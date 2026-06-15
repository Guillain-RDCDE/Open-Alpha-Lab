"""Shared fixtures for Study 165 (Chinese-Zodiac) -- deterministic synthetic tapes only.

No network calls. The synthetic tape plants a known Dragon-year bonus (or none), so tests
can assert that the analysis finds the effect only when it is baked in.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from chinese_zodiac import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A daily return series with *no* zodiac seasonality -- the null hypothesis."""
    ret, truth = data.synthetic_daily(n_years=36, dragon_bonus_bp=0.0, seed=165)
    return ret, truth


@pytest.fixture
def dragon_tape():
    """A daily return series with a planted Dragon-year bonus -- something to find."""
    ret, truth = data.synthetic_daily(n_years=36, dragon_bonus_bp=200.0, seed=165)
    return ret, truth

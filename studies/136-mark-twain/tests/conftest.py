"""Shared fixtures for Study 136 (Mark-Twain) — deterministic synthetic tapes only.

No network calls. The synthetic tape plants a known October penalty (or none), so tests
can assert that the analysis finds the effect only when it is baked in.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mark_twain import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A daily return series with *no* October seasonality — the null hypothesis."""
    ret, truth = data.synthetic_daily(n_years=60, october_penalty_bp=0.0, seed=136)
    return ret, truth


@pytest.fixture
def penalised_tape():
    """A daily return series with a planted −100 bps/day October drag — something to find."""
    ret, truth = data.synthetic_daily(n_years=60, october_penalty_bp=-100.0, seed=136)
    return ret, truth

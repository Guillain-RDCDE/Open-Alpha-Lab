"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
tax-day anomaly, so tests never touch the network and the only thing the strategy
can harvest (a planted return differential) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tax_day import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A martingale tape (both effects = 0) — tax windows are just random days."""
    df, truth = data.synthetic_daily(n_years=75, pre_tax_effect=0.0, post_tax_effect=0.0, seed=192)
    return df, truth


@pytest.fixture
def pre_effect_tape():
    """A tape with a planted positive pre-tax effect (+3 sigma days)."""
    df, truth = data.synthetic_daily(n_years=75, pre_tax_effect=3.0, post_tax_effect=0.0, seed=192)
    return df, truth


@pytest.fixture
def post_effect_tape():
    """A tape with a planted negative post-tax effect (−3 sigma days)."""
    df, truth = data.synthetic_daily(n_years=75, pre_tax_effect=0.0, post_tax_effect=-3.0, seed=192)
    return df, truth

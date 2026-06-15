"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
Friday-13th anomaly, so tests never touch the network and the only thing the
strategy can harvest (a planted return differential) is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from friday_13th import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A martingale tape (f13_effect=0) — the 13th is just another Friday."""
    df, truth = data.synthetic_daily(n_years=80, f13_effect=0.0, seed=163)
    return df, truth


@pytest.fixture
def negative_tape():
    """A tape with a planted negative return on Friday-13th (−3 sigma days)."""
    df, truth = data.synthetic_daily(n_years=80, f13_effect=-3.0, seed=163)
    return df, truth

"""Shared fixtures -- deterministic synthetic monthly (sentiment, return) tapes with a
*known* contrarian edge, so tests never touch the network and the only thing the
contrarian rule can harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from aaii_sentiment import data  # noqa: E402


def _to_panel(df):
    """Stamp the forward return at month-end t (spread at t predicts ret at t+1)."""
    p = df.copy()
    p["ret"] = df["ret"].shift(-1)
    return p.dropna()


@pytest.fixture
def null_panel():
    """No contrarian edge: prior spread carries no info about next-month return."""
    df, truth = data.synthetic_monthly(n_months=360, edge=0.0, seed=257)
    return _to_panel(df), truth


@pytest.fixture
def edge_panel():
    """A planted contrarian edge: high spread -> lower next-month return."""
    df, truth = data.synthetic_monthly(n_months=360, edge=0.06, seed=257)
    return _to_panel(df), truth

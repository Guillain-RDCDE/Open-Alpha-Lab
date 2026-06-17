"""Shared fixtures -- deterministic synthetic margin-debt series with a *known*
contrarian forward signal (or none), so tests never touch the network and the
only thing the regression can recover is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from margin_debt import data  # noqa: E402


def _to_panel(df):
    """Turn a synthetic_series frame into a panel with a forward return column."""
    df = df.dropna(subset=["yoy"]).copy()
    df["fwd_ret"] = df["index_ret"].shift(-1)
    df = df.dropna(subset=["fwd_ret"]).reset_index(drop=True)
    df["mkt_up"] = df["fwd_ret"] > 0
    return df


@pytest.fixture
def null_panel():
    """No planted forward signal: margin growth co-moves with the market but
    carries no forecasting power. Regression slope should be ~0 (|t| < 2)."""
    df, truth = data.synthetic_series(n_years=400, predictive_beta=0.0, seed=260)
    return _to_panel(df), truth


@pytest.fixture
def live_panel():
    """A planted contrarian forward signal: high YoY margin growth predicts low
    forward return. Regression slope should be strongly NEGATIVE (t < -2)."""
    df, truth = data.synthetic_series(n_years=400, predictive_beta=0.06, seed=260)
    return _to_panel(df), truth

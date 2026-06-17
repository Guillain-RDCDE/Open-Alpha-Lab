"""Shared fixtures for Study 235 (World-Cup-Effect).

All fixtures are deterministic and offline — they use either the hardcoded
World Cup table or the synthetic generator, never the network or the yfinance
cache (so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from world_cup_effect import data  # noqa: E402


@pytest.fixture
def wc_table():
    """The hardcoded World Cup windows table (19 editions, 1950-2022)."""
    return data.worldcup_table()


@pytest.fixture
def synthetic_null():
    """A synthetic daily-return frame with NO planted effect (the null)."""
    df, truth = data.synthetic_daily(n_days=2000, signal_bps=0.0, seed=235)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic daily-return frame with a strong planted WC effect."""
    df, truth = data.synthetic_daily(n_days=2000, signal_bps=500.0, seed=235)
    return df, truth


@pytest.fixture
def tagged_null():
    """A synthetic tagged DataFrame (null — no planted effect)."""
    df, _ = data.synthetic_daily(n_days=2000, signal_bps=0.0, seed=235)
    df = df.rename(columns={"return": "sp500_return"})
    df = df.set_index("date")
    df["ctrl_window"] = False
    df["edition"] = float("nan")
    # Mark first 50 days as WC, days 200-250 as control (rough approximation)
    df.iloc[:50, df.columns.get_loc("in_wc")] = True
    df.loc[df.index[200:250], "ctrl_window"] = True
    df.loc[df.index[:50], "edition"] = 1.0
    return df

"""Tests for the data layer — synthetic world shape, determinism, the knob wiring,
and the split-calendar fingerprint."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_split_modern import data  # noqa: E402


def test_synthetic_world_shapes(null_world):
    prices, market, splits = null_world
    assert data.MARKET in prices.columns
    assert isinstance(market, pd.Series)
    assert not splits.empty
    assert set(splits.columns) == {"ticker", "date", "ratio"}
    assert (splits["ratio"] >= 1.5).all()


def test_synthetic_world_is_deterministic():
    a = data.synthetic_world(planted_bps=0.0, seed=7)[0]
    b = data.synthetic_world(planted_bps=0.0, seed=7)[0]
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c = data.synthetic_world(planted_bps=0.0, seed=8)[0]
    assert not np.allclose(a.to_numpy()[:, 0], c.to_numpy()[:, 0])


def test_planted_knob_lifts_stock_paths(null_world, planted_world):
    """A positive planted drift must push synthetic stock prices above the null."""
    p_null = null_world[0].drop(columns=[data.MARKET]).iloc[-1].mean()
    p_plant = planted_world[0].drop(columns=[data.MARKET]).iloc[-1].mean()
    assert p_plant > p_null


def test_fingerprint_stable_and_content_sensitive(null_world, planted_world):
    _, _, sp_a = null_world
    fp1 = data.fingerprint_splits(sp_a)
    fp2 = data.fingerprint_splits(sp_a)
    assert fp1 == fp2
    # a shifted ratio changes the fingerprint
    sp_b = sp_a.copy()
    sp_b.loc[sp_b.index[0], "ratio"] = 99.0
    assert data.fingerprint_splits(sp_b) != fp1


def test_market_column_present_and_positive(null_world):
    prices = null_world[0]
    assert (prices[data.MARKET].dropna() > 0).all()

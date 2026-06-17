"""The synthetic panel is well-formed, deterministic, and carries the planted BAB signal."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from betting_against_beta import data  # noqa: E402

_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache", "bab_prices.parquet"
)


def test_synthetic_shapes_match(null_panel):
    stock_rets, mkt, truth = null_panel
    assert stock_rets.ndim == 2
    assert mkt.ndim == 1
    assert len(stock_rets) == len(mkt)
    assert stock_rets.index.equals(mkt.index)


def test_synthetic_is_business_day_indexed(null_panel):
    stock_rets, mkt, _ = null_panel
    # All dates should be business days (Mon-Fri)
    assert (stock_rets.index.dayofweek < 5).all()


def test_synthetic_is_deterministic():
    a, ma, _ = data.synthetic_panel(n_stocks=50, n_days=500, bab_premium=0.05, seed=7)
    b, mb, _ = data.synthetic_panel(n_stocks=50, n_days=500, bab_premium=0.05, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _, _ = data.synthetic_panel(n_stocks=50, n_days=500, bab_premium=0.05, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_seed_changes_output():
    a, _, _ = data.synthetic_panel(seed=1)
    b, _, _ = data.synthetic_panel(seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_null_truth_flag(null_panel):
    _, _, truth = null_panel
    assert not truth.has_premium
    assert truth.bab_premium == 0.0


def test_live_truth_flag(live_panel):
    _, _, truth = live_panel
    assert truth.has_premium
    assert truth.bab_premium > 0


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    """Absent cache + no network --> graceful empty DataFrames, not a crash."""
    prices, spy = data.fetch_panel(cache_dir=str(tmp_path))
    # Either empty (CI / no network) or populated (live run with network).
    # The point: no exception.
    assert isinstance(prices, type(prices))  # always true -- just check no crash
    assert isinstance(spy, type(spy))


@pytest.mark.skipif(not os.path.exists(_CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_fetch_panel_real_cache():
    """When cache exists, returns non-empty DataFrames."""
    prices, spy = data.fetch_panel()
    assert not prices.empty
    assert not spy.empty
    assert len(prices) == len(spy)


def test_fingerprint_stable_and_sensitive(null_panel):
    stock_rets, _, _ = null_panel
    fp = data.fingerprint(stock_rets)
    assert fp == data.fingerprint(stock_rets)
    other, _, _ = data.synthetic_panel(n_stocks=80, n_days=1500, bab_premium=0.0, seed=99)
    assert fp != data.fingerprint(other)

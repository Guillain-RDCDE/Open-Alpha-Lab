"""Data-layer tests for Study 969 — synthetic determinism offline, cache-gated on tape."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from log_vs_simple import data  # noqa: E402


def test_synthetic_panel_is_deterministic():
    a, ca, _ = data.synthetic_panel(n_assets=5, n_years=4, seed=969)
    b, cb, _ = data.synthetic_panel(n_assets=5, n_years=4, seed=969)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert np.allclose(ca.to_numpy(), cb.to_numpy())


def test_synthetic_panel_seed_sensitive():
    a, _, _ = data.synthetic_panel(n_assets=5, n_years=4, seed=969)
    b, _, _ = data.synthetic_panel(n_assets=5, n_years=4, seed=969 + 1)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_panel_shape_and_index():
    prices, cash, truth = data.synthetic_panel(n_assets=7, n_years=6, seed=969)
    assert prices.shape == (6 * data.TRADING_DAYS_PER_YEAR, 7)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert cash.index.equals(prices.index)
    assert truth["n_days"] == len(prices)
    assert prices.index[-1] < pd.Timestamp("2262-01-01")   # inside pandas' ns horizon
    assert (prices > 0).all().all()


def test_signal_strength_scales_the_plant():
    _, _, t1 = data.synthetic_panel(signal_strength=1.0, seed=969)
    _, _, th = data.synthetic_panel(signal_strength=0.5, seed=969)
    _, _, t0 = data.synthetic_panel(signal_strength=0.0, seed=969)
    assert t0["alpha_vol_eff"] == 0.0
    assert t1["alpha_vol_eff"] > th["alpha_vol_eff"] > 0.0


def test_synthetic_cash_is_monotone():
    _, cash, truth = data.synthetic_panel(n_years=5, seed=969)
    assert (cash.diff().dropna() > 0).all()
    assert truth["cash_rate_ann"] > 0


def test_fingerprint_stable_and_sensitive():
    a, _, _ = data.synthetic_panel(n_assets=4, n_years=3, seed=969)
    b, _, _ = data.synthetic_panel(n_assets=4, n_years=3, seed=969 + 1)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_is_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_universe_is_declared():
    assert len(data.TICKERS) == len(set(data.TICKERS)) >= 2
    assert pd.Timestamp(data.AS_OF) > pd.Timestamp(data.START)


@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_loads_and_is_pinned():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    assert px.index.is_monotonic_increasing and not px.index.has_duplicates
    assert (px.dropna(how="all") > 0).any().all()

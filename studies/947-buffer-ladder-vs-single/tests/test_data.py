"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffer_ladder import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline, no cache)
# --------------------------------------------------------------------------- #
def test_synthetic_panel_is_deterministic():
    a, _ = data.synthetic_panel(seed=947)
    b, _ = data.synthetic_panel(seed=947)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_panel_seed_sensitive():
    a, _ = data.synthetic_panel(seed=947)
    b, _ = data.synthetic_panel(seed=948)
    assert not np.allclose(a["ladder"].to_numpy(), b["ladder"].to_numpy())


def test_synthetic_panel_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=5, n_vintages=4, seed=947)
    assert list(prices.columns) == ["ladder", "v1", "v2", "v3", "v4", "market", "cash"]
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 5 * data.TRADING_DAYS_PER_YEAR
    assert truth["vintages"] == ("v1", "v2", "v3", "v4")
    # OOB-safe: the synthetic index must stay far inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_panel_variable_vintage_count():
    prices, truth = data.synthetic_panel(n_vintages=6, seed=947)
    assert len(truth["vintages"]) == 6
    assert set(truth["vintages"]).issubset(prices.columns)


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_panel(seed=947)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_synthetic_daily_shape():
    prices, truth = data.synthetic_daily(n_years=4, seed=947)
    assert list(prices.columns) == ["market", "cash"]
    assert len(prices) == truth["n_days"] == 4 * data.TRADING_DAYS_PER_YEAR


def test_signal_strength_scales_the_planted_gap():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=947)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=947)
    assert t1["expected_gap_ann"] > t0["expected_gap_ann"]
    # At signal_strength = 0 the only planted effect left is the fee layer.
    assert t0["expected_gap_ann"] == pytest.approx(-t0["extra_fee_ann"])


def test_vintages_share_the_market_factor():
    """Vintages must be highly (but not perfectly) correlated — the panel's whole point."""
    prices, truth = data.synthetic_panel(seed=947)
    r = st.to_returns(prices)
    c = r[list(truth["vintages"])].corr().to_numpy()
    iu = np.triu_indices_from(c, k=1)
    assert 0.5 < c[iu].mean() < 0.999


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=947)
    b, _ = data.synthetic_panel(seed=948)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_fee_proxies_are_declared():
    """The non-tape inputs must exist as named, sweepable constants, not magic numbers."""
    assert data.FEE_SINGLE_VINTAGE_PCT > 0
    assert data.FEE_LADDER_EXTRA_PCT in data.FEE_EXTRA_GRID_PCT
    assert 0.0 in data.FEE_EXTRA_GRID_PCT      # the "assume nothing" corner of the sweep


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when _cache/ is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) - synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    assert set(data.TICKERS).issubset(px.columns)
    res = st.race(px, data.LADDER, data.VINTAGES, data.MARKET, data.CASH)
    for key in ("vs_diy_basket", "vs_diy_beta_matched", "vs_beta_mix"):
        assert np.isfinite(res["gaps"][key]["gap_ann_pp"])
        assert np.isfinite(res["gaps"][key]["t_hac"])
    # The buffers damp the market: every buffer arm must be less volatile than SPY.
    assert res["summary"]["market"]["vol_ann"] > res["summary"]["ladder"]["vol_ann"]
    assert res["summary"]["ladder"]["vol_ann"] > res["summary"]["diy_basket"]["vol_ann"]

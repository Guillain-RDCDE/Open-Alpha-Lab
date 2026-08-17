"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rebal_bands import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_years=6, seed=936)
    b, _ = data.synthetic_panel(n_years=6, seed=936)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_seed_changes_the_tape():
    a, _ = data.synthetic_panel(n_years=6, seed=936)
    b, _ = data.synthetic_panel(n_years=6, seed=937)
    assert not np.allclose(a["a0"].to_numpy(), b["a0"].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=8, n_assets=3, seed=936)
    assert list(prices.columns) == ["a0", "a1", "a2", "cash"]
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 8 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay well inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")
    assert prices.notna().all().all()


def test_synthetic_daily_is_the_two_asset_wrapper():
    a, ta = data.synthetic_daily(n_years=5, seed=936)
    b, tb = data.synthetic_panel(n_years=5, n_assets=2, seed=936)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert ta["n_assets"] == tb["n_assets"] == 2


def test_signal_strength_controls_dispersion_persistence():
    """strength 0 -> spread is a random walk (phi ~ 1); strength 1 -> it mean-reverts."""
    _, t1 = data.synthetic_panel(n_years=12, signal_strength=1.0, seed=936)
    _, t0 = data.synthetic_panel(n_years=12, signal_strength=0.0, seed=936)
    assert t0["phi"] == 1.0
    assert t1["phi"] < 0.999
    # The realised AR(1) of the dispersion follows the planted one.
    assert t0["phi_hat_spread"] > t1["phi_hat_spread"]
    assert t0["phi_hat_spread"] > 0.995


def test_synthetic_cash_is_monotone_growing():
    prices, truth = data.synthetic_panel(n_years=6, seed=936)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_daily_returns_shape():
    prices, _ = data.synthetic_panel(n_years=4, seed=936)
    r = data.daily_returns(prices)
    assert len(r) == len(prices) - 1
    assert not r.isna().any().any()


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(n_years=4, seed=936)
    b, _ = data.synthetic_panel(n_years=4, seed=937)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_bad_asset_count_raises():
    with pytest.raises(ValueError):
        data.synthetic_panel(n_years=3, n_assets=4, drift_ann=(0.05, 0.05), vol_ann=(0.1, 0.1))


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    rets = data.daily_returns(px)
    r = st.race(rets[["SPY", "IEF"]], (0.6, 0.4), cash=rets["BIL"], cost_bps=5.0)
    for p in st.POLICIES:
        assert np.isfinite(r["stats"][p]["sharpe"])
    # The do-nothing book never trades; every scheduled policy does.
    assert r["traded_per_year"]["drift"] == 0.0
    assert r["traded_per_year"]["quarterly"] > r["traded_per_year"]["annual"] > 0.0

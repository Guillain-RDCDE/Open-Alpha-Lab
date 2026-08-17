"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stale_nav import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=917)
    b, _ = data.synthetic_panel(seed=917)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=10, n_funds=3, seed=917)
    assert {"SPY", "FUND1", "FUND2", "FUND3", "cash"} == set(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 10 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_daily_is_one_fund():
    prices, truth = data.synthetic_daily(n_years=5, seed=917)
    assert truth["n_funds"] == 1
    assert truth["funds"] == ["FUND1"]
    assert "FUND1" in prices.columns


def test_signal_strength_scales_the_planted_beta():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=917)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=917)
    _, th = data.synthetic_panel(signal_strength=0.5, seed=917)
    assert t1["stale_beta_effective"] == pytest.approx(t1["stale_beta"])
    assert t0["stale_beta_effective"] == 0.0
    assert th["stale_beta_effective"] == pytest.approx(0.5 * th["stale_beta"])


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_panel(seed=917)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_contemporaneous_load_present_in_both_worlds():
    """The null must look just as *correlated* — only the LAGGED link is switched off."""
    out = []
    for ss in (0.0, 1.0):
        prices, truth = data.synthetic_panel(signal_strength=ss, seed=917)
        r = prices.pct_change()
        out.append(float(r["FUND1"].corr(r["SPY"])))
    assert out[0] > 0.4 and out[1] > 0.4


def test_safe_name_strips_index_prefix():
    assert data.safe_name("^IRX") == "IRX"
    assert data.safe_name("EWJ") == "EWJ"


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=917)
    fp1 = data.fingerprint(a)
    fp2 = data.fingerprint(a)
    b, _ = data.synthetic_panel(seed=918)
    assert fp1 == fp2 and len(fp1) == 12
    assert fp1 != data.fingerprint(b)


def test_cash_returns_from_irx_is_lagged_and_positive():
    idx = pd.bdate_range("2020-01-01", periods=50)
    irx = pd.Series(np.full(50, 5.0), index=idx)     # a flat 5% bill rate
    c = data.cash_returns_from_irx(irx)
    assert np.isnan(c.iloc[0])                        # lagged one day: no look-ahead
    daily = (1.05) ** (1 / 252) - 1
    assert c.dropna().iloc[0] == pytest.approx(daily)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_regression_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    rets = px.pct_change()
    tbl = st.per_fund_regressions(rets, data.FUNDS)
    assert set(tbl.index) == set(data.FUNDS)
    assert np.isfinite(tbl["beta_raw"]).all()
    assert np.isfinite(tbl["t_rel"]).all()

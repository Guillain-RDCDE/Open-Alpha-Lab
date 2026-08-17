"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimal_leverage import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=944)
    b, _ = data.synthetic_daily(seed=944)
    assert np.allclose(a["asset"].to_numpy(), b["asset"].to_numpy())
    assert np.allclose(a["cash"].to_numpy(), b["cash"].to_numpy())


def test_synthetic_seed_changes_the_tape():
    a, _ = data.synthetic_daily(seed=944)
    b, _ = data.synthetic_daily(seed=945)
    assert not np.allclose(a["asset"].to_numpy(), b["asset"].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=20, seed=944)
    assert {"asset", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 20 * data.TRADING_DAYS_PER_YEAR
    assert prices.index.is_monotonic_increasing
    # OOB-safe: the synthetic index must stay inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_rejects_an_out_of_range_horizon():
    with pytest.raises(ValueError):
        data.synthetic_daily(n_years=1000)


def test_synthetic_honours_the_planted_vol():
    prices, truth = data.synthetic_daily(n_years=40, vol_ann=0.16, seed=944)
    r = prices["asset"].pct_change().dropna()
    realised = float(r.std(ddof=1) * np.sqrt(data.TRADING_DAYS_PER_YEAR))
    assert abs(realised - truth["vol_ann"]) < 0.02


def test_signal_strength_zero_kills_the_excess_drift():
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=944)
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=944)
    assert t1["kelly_true"] == pytest.approx(2.0)
    assert t0["kelly_true"] == pytest.approx(0.0)
    assert t0["mu_excess_ann"] == pytest.approx(0.0)


def test_synthetic_cash_leg_grows_monotonically():
    prices, truth = data.synthetic_daily(seed=944)
    assert (prices["cash"].diff().dropna() > 0).all()
    years = truth["n_days"] / data.TRADING_DAYS_PER_YEAR
    grew = float(prices["cash"].iloc[-1] / prices["cash"].iloc[0])
    assert grew == pytest.approx((1 + truth["cash_rate_ann"] / 252) ** (years * 252 - 1), rel=1e-6)


# --------------------------------------------------------------------------- #
# The ^IRX -> daily cash-rate conversion
# --------------------------------------------------------------------------- #
def test_cash_rate_daily_uses_act_360_and_the_previous_close():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])  # Thu, Fri, Mon
    irx = pd.Series([3.6, 3.6, 3.6], index=idx)
    r = data.cash_rate_daily(irx)
    assert np.isnan(r.iloc[0])                       # no previous close to finance at
    assert r.iloc[1] == pytest.approx(0.036 / 360)   # one calendar day
    assert r.iloc[2] == pytest.approx(0.036 * 3 / 360)  # the weekend earns three days


def test_cash_rate_daily_is_zero_when_the_rate_is_zero():
    idx = pd.bdate_range("2021-01-04", periods=10)
    r = data.cash_rate_daily(pd.Series(0.0, index=idx))
    assert (r.dropna() == 0.0).all()


# --------------------------------------------------------------------------- #
# Fingerprint & cache contract
# --------------------------------------------------------------------------- #
def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=944)
    b, _ = data.synthetic_daily(seed=945)
    assert data.fingerprint(a) == data.fingerprint(a)
    assert len(data.fingerprint(a)) == 12
    assert data.fingerprint(a) != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_is_false_on_an_empty_cache(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_cache_path_strips_the_index_caret():
    p = data._cache_path("^IRX", "/tmp/x")
    assert os.path.basename(p) == "prices_IRX_1d.parquet"


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared studies/_cache present (CI) — synthetic tests cover the logic")
def test_real_cache_sweep_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    legs = st.prepare_real(px)
    tab = st.sweep(legs, grid=(1.0, 2.0, 3.0))
    assert np.isfinite(tab["cagr"]).all()
    # Leverage scales volatility essentially linearly, and drawdowns deepen with it.
    assert tab.loc[3.0, "vol_ann"] > tab.loc[1.0, "vol_ann"] * 2.5
    assert tab.loc[3.0, "max_drawdown"] < tab.loc[1.0, "max_drawdown"]

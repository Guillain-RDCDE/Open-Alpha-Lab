"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tranching import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=937)
    b, _ = data.synthetic_daily(seed=937)
    for col in ("risky", "safe", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=20, seed=937)
    assert {"risky", "safe", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 20 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_daily(seed=937)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_signal_strength_zero_flattens_regimes():
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=937)
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=937)
    assert abs(t0["vol_bull_eff"] - t0["vol_bear_eff"]) < 1e-9
    assert abs(t0["drift_bull_eff"] - t0["drift_bear_eff"]) < 1e-9
    assert t1["vol_bear_eff"] > t1["vol_bull_eff"]
    assert t1["drift_bear_eff"] < t1["drift_bull_eff"]


def test_synthetic_panel_is_distinct_and_deterministic():
    a = data.synthetic_panel(3)
    b = data.synthetic_panel(3)
    assert len(a) == 3
    assert np.allclose(a[0][0]["risky"].to_numpy(), b[0][0]["risky"].to_numpy())
    assert not np.allclose(a[0][0]["risky"].to_numpy(), a[1][0]["risky"].to_numpy())


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=937)
    b, _ = data.synthetic_daily(seed=938)
    fp1 = data.fingerprint(a)
    assert fp1 == data.fingerprint(a) and len(fp1) == 12
    assert fp1 != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_false_on_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


def test_cash_from_yield_is_lagged_and_positive():
    y = pd.Series([5.0] * 10, index=pd.bdate_range("2020-01-01", periods=10))
    r = st.cash_returns_from_yield(y)
    assert np.isnan(r.iloc[0])                       # one lag: day 0 earns nothing yet
    assert abs(r.iloc[5] - 5.0 / 100 / 252) < 1e-12


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_cone_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    spy, ief = px["SPY"].dropna(), px["IEF"].dropna()
    idx = spy.index.intersection(ief.index)
    state = st.trend_state(spy.loc[idx])
    R, TV = st.offset_return_matrix(spy.loc[idx], ief.loc[idx], state)
    r_cash = st.cash_returns_from_yield(px["IRX"].reindex(idx))
    cn = st.cone(R, TV, r_cash, n_list=(1, 21))
    assert cn[1]["sharpe_sd"] > 0.0
    assert cn[21]["sharpe_sd"] == pytest.approx(0.0, abs=1e-12)

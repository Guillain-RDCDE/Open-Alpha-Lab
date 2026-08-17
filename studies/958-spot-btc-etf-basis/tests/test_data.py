"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etf_basis import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=958)
    b, _ = data.synthetic_panel(seed=958)
    for col in ("spot", "spot_etf", "futures_etf", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_panel(n_years=4, seed=958)
    assert {"spot", "spot_etf", "futures_etf", "cash"} == set(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 4 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay well inside pandas' ns horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_daily_is_the_headline_pair():
    pair, truth = data.synthetic_daily(seed=958)
    panel, truth2 = data.synthetic_panel(seed=958)
    assert list(pair.columns) == ["futures_etf", "spot"]
    assert np.allclose(pair["futures_etf"].to_numpy(), panel["futures_etf"].to_numpy())
    assert truth["drag_fut_pre_pct"] == truth2["drag_fut_pre_pct"]


def test_signal_strength_scales_the_planted_compression():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=958)
    _, t_half = data.synthetic_panel(signal_strength=0.5, seed=958)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=958)
    assert t0["drag_change_pct"] == 0.0
    assert t1["drag_change_pct"] > t_half["drag_change_pct"] > 0.0
    # the *level* of the pre-event drag is planted identically in all three worlds
    assert t0["drag_fut_pre_pct"] == t1["drag_fut_pre_pct"]


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_panel(seed=958)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_synthetic_wrappers_track_spot_but_carry_the_timestamp_offset():
    """A wrapper is the spot path plus a drag and a fat close-timestamp offset.

    The offset is deliberately of the same order as a daily bitcoin move, so the daily
    correlation is high but far from one — exactly the real tape's problem, and the
    reason the endpoint estimator is unusable against a differently-stamped reference.
    """
    prices, _ = data.synthetic_panel(seed=958)
    r_spot = np.log(prices["spot"]).diff().dropna()
    r_fut = np.log(prices["futures_etf"]).diff().dropna()
    assert 0.4 < r_spot.corr(r_fut) < 0.95


def test_fees_are_documented_proxies():
    assert set(data.FEES) == {"BITO", "IBIT", "FBTC"}
    assert data.FEES["BITO"] > data.FEES["IBIT"]
    assert "PROXY" in data.__doc__ or "assumption" in data.__doc__.lower()


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=958)
    b, _ = data.synthetic_panel(seed=959)
    fp1 = data.fingerprint(a)
    assert fp1 == data.fingerprint(a) and len(fp1) == 12
    assert fp1 != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_cash_yield_on_a_known_index():
    idx = pd.bdate_range("2020-01-01", periods=756)
    lvl = pd.Series(100.0 * (1.03 ** (np.arange(756) / 252.0)), index=idx)
    y = data.cash_yield(pd.DataFrame({"BIL": lvl}))
    assert abs(y - 0.03) < 0.01


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_drag_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    res = st.trend_drag(px["BITO"].dropna(), px["BTC-USD"].dropna())
    assert np.isfinite(res["drag_pct"]) and np.isfinite(res["t"])
    assert res["drag_pct"] < 0.0  # the futures wrapper bleeds against spot

"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_close_exec import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=938)
    b, _ = data.synthetic_daily(seed=938)
    for col in ("open", "close", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    frame, truth = data.synthetic_daily(n_years=12, seed=938)
    assert {"open", "close", "cash"}.issubset(frame.columns)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert len(frame) == truth["n_days"] == 12 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert frame.index[-1] < pd.Timestamp("2262-01-01")
    assert (frame[["open", "close", "cash"]] > 0).all().all()


def test_leg_decomposition_is_exact():
    """(1 + overnight)(1 + intraday) must reproduce the close-to-close return exactly."""
    frame, _ = data.synthetic_daily(seed=938)
    legs = st.return_legs(frame[["open", "close"]]).dropna()
    lhs = (1.0 + legs["r_on"]) * (1.0 + legs["r_id"])
    assert np.allclose(lhs.to_numpy(), (1.0 + legs["r_cc"]).to_numpy(), atol=1e-12)


def test_signal_strength_leaves_close_to_close_untouched():
    """The knob only *re-splits* the day; the close tape (and so the rule) is invariant."""
    a, _ = data.synthetic_daily(signal_strength=0.0, seed=938)
    b, _ = data.synthetic_daily(signal_strength=1.0, seed=938)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert not np.allclose(a["open"].to_numpy(), b["open"].to_numpy())


def test_planted_tilt_scales_with_signal_strength():
    _, t0 = data.synthetic_daily(signal_strength=0.0, seed=938)
    _, t1 = data.synthetic_daily(signal_strength=1.0, seed=938)
    assert t0["planted_intraday_tilt_bps"] == 0.0
    assert t1["planted_intraday_tilt_bps"] > 0.0


def test_synthetic_cash_is_monotone_growing():
    frame, _ = data.synthetic_daily(seed=938)
    assert (frame["cash"].diff().dropna() > 0).all()
    assert frame["cash"].iloc[-1] > frame["cash"].iloc[0]


def test_synthetic_panel_shape_and_independence():
    frames, truth = data.synthetic_panel(n_assets=3, seed=938)
    assert set(frames) == {"SYN0", "SYN1", "SYN2"}
    assert truth["n_assets"] == 3
    a, b = frames["SYN0"]["close"].to_numpy(), frames["SYN1"]["close"].to_numpy()
    assert not np.allclose(a, b)


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=938)
    b, _ = data.synthetic_daily(seed=939)
    fp = data.fingerprint({"a": a})
    assert fp == data.fingerprint({"a": a}) and len(fp) == 12
    assert fp != data.fingerprint({"a": b})


def test_load_ohlc_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_ohlc(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared OHLC cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_race_runs():
    frames = data.load_ohlc()
    cash = frames[data.CASH]["close"]
    for tk in data.TICKERS:
        assert frames[tk].index[-1] <= pd.Timestamp(data.AS_OF)
    arms = st.run_tape(frames["SPY"], cash, freq="M", lookback=10, cost_bps=2.0)
    g = st.venue_gap(arms)
    for k in ("gap_ann_bps", "gap_t_hac", "sharpe_open", "sharpe_close"):
        assert np.isfinite(g[k])
    # The two arms hold the same book on every non-trade day, so they must be very close.
    assert abs(g["sharpe_open"] - g["sharpe_close"]) < 0.25

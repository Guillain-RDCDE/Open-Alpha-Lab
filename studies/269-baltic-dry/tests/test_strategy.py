"""Signal construction, predictive regression, timing overlay, summary stats,
and the study's spine: the BDI->equity slope is strongly positive only when a
lead-lag is planted, and ~zero on the null."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from baltic_dry import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def test_signal_shape_and_burnin(null_frame):
    df, _ = null_frame
    sig = st.bdi_signal(df, lookback=3)
    assert len(sig) == len(df)
    assert sig.iloc[:3].isnull().all(), "First lookback rows must be NaN (burn-in)"


def test_signal_deterministic(null_frame):
    df, _ = null_frame
    a = st.bdi_signal(df, lookback=3)
    b = st.bdi_signal(df, lookback=3)
    pd.testing.assert_series_equal(a, b)


def test_forward_return_is_shifted(null_frame):
    df, _ = null_frame
    fwd = st.forward_sp_return(df)
    ret = df["sp500"].pct_change()
    # fwd[t] should equal ret[t+1]
    np.testing.assert_allclose(
        fwd.iloc[:-1].to_numpy(), ret.shift(-1).iloc[:-1].to_numpy(), atol=1e-12
    )
    assert np.isnan(fwd.iloc[-1]), "Last forward return must be NaN (no t+1)"


# ---------------------------------------------------------------------------
# Predictive regression
# ---------------------------------------------------------------------------
def test_regression_keys(null_frame):
    df, _ = null_frame
    sig = st.bdi_signal(df, lookback=3)
    fwd = st.forward_sp_return(df)
    r = st.predictive_regression(sig, fwd)
    assert set(["alpha", "beta", "tstat", "r2", "n"]).issubset(r)


def test_regression_short_series_guard():
    s = pd.Series([0.1, 0.2, 0.3])
    f = pd.Series([0.01, 0.02, 0.03])
    r = st.predictive_regression(s, f)
    assert np.isnan(r["beta"])


def test_null_slope_near_zero(null_frame):
    """With beta=0 planted, the BDI signal carries no forward information."""
    df, _ = null_frame
    sig = st.bdi_signal(df, lookback=3)
    fwd = st.forward_sp_return(df)
    r = st.predictive_regression(sig, fwd)
    assert np.isfinite(r["tstat"])
    assert abs(r["tstat"]) < 3.0, f"Null |t| too large: {r['tstat']:.2f}"


def test_planted_slope_significant(live_frame):
    """With beta=0.6 planted, the slope is strongly positive (t > 4)."""
    df, _ = live_frame
    sig = st.bdi_signal(df, lookback=3)
    fwd = st.forward_sp_return(df)
    r = st.predictive_regression(sig, fwd)
    assert r["beta"] > 0.0, f"Planted slope should be positive, got {r['beta']:.4f}"
    assert r["tstat"] > 4.0, f"Planted slope should be strongly significant, got t={r['tstat']:.2f}"


# ---------------------------------------------------------------------------
# Timing overlay
# ---------------------------------------------------------------------------
def test_timing_columns(null_frame):
    df, _ = null_frame
    sig = st.bdi_signal(df, lookback=3)
    fwd = st.forward_sp_return(df)
    tim = st.timing_returns(sig, fwd)
    assert set(["position", "buy_hold", "timing_gross", "cost", "timing_net"]).issubset(tim.columns)


def test_timing_position_binary(null_frame):
    df, _ = null_frame
    sig = st.bdi_signal(df, lookback=3)
    fwd = st.forward_sp_return(df)
    tim = st.timing_returns(sig, fwd)
    assert set(np.unique(tim["position"])).issubset({0.0, 1.0})


def test_timing_net_le_gross(null_frame):
    """Net return is gross minus a non-negative cost."""
    df, _ = null_frame
    sig = st.bdi_signal(df, lookback=3)
    fwd = st.forward_sp_return(df)
    tim = st.timing_returns(sig, fwd, one_way_bps=5.0)
    assert (tim["cost"] >= 0).all()
    np.testing.assert_allclose(
        tim["timing_net"].to_numpy(),
        (tim["timing_gross"] - tim["cost"]).to_numpy(),
        atol=1e-12,
    )


def test_timing_higher_cost_lower_net(live_frame):
    df, _ = live_frame
    sig = st.bdi_signal(df, lookback=3)
    fwd = st.forward_sp_return(df)
    cheap = st.timing_returns(sig, fwd, one_way_bps=5.0)["timing_net"].mean()
    dear = st.timing_returns(sig, fwd, one_way_bps=50.0)["timing_net"].mean()
    assert dear <= cheap + 1e-12, "Higher costs must not raise net return"


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(60) * 0.04)
    s = st.summarize(r)
    assert set(["mean", "vol", "sharpe", "tstat", "hit_rate", "n"]).issubset(s)


def test_summarize_short_series():
    s = st.summarize(pd.Series([0.01, -0.02]))
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_annualise_monthly():
    s = st.summarize(pd.Series([0.01] * 60))
    out = st.annualise_monthly(s, periods=12)
    assert abs(out["mean_ann"] - 12 * 0.01) < 1e-10


def test_sign_hit_rate_keys(null_frame):
    df, _ = null_frame
    sig = st.bdi_signal(df, lookback=3)
    fwd = st.forward_sp_return(df)
    h = st.sign_hit_rate(sig, fwd)
    assert set(["hit", "base_rate", "n"]).issubset(h)
    assert 0.0 <= h["hit"] <= 1.0

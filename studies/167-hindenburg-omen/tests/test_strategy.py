"""Tests for strategy.py — cluster deduplication, forward returns, crash-rate logic."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hindenburg_omen import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# cluster_signals
# ---------------------------------------------------------------------------
def test_cluster_deduplication_merges_consecutive_days():
    """Two signal days within 30 calendar days collapse to one cluster."""
    idx = pd.bdate_range("2020-01-02", periods=60, name="date")
    breadth = pd.DataFrame({"hindenburg": 0}, index=idx)
    # Days 10 and 15 are within 5 business days — should cluster to day 10 only.
    breadth.iloc[10, 0] = 1
    breadth.iloc[15, 0] = 1
    clusters = st.cluster_signals(breadth, window_days=30)
    assert clusters.sum() == 1
    assert clusters.iloc[10]


def test_cluster_signals_far_apart_are_separate():
    """Two signal days more than 30 calendar days apart are two separate clusters."""
    idx = pd.bdate_range("2020-01-02", periods=120, name="date")
    breadth = pd.DataFrame({"hindenburg": 0}, index=idx)
    breadth.iloc[10, 0] = 1
    breadth.iloc[80, 0] = 1   # ~70 business days later = well over 30 calendar days
    clusters = st.cluster_signals(breadth, window_days=30)
    assert clusters.sum() == 2


def test_cluster_signals_no_signals_returns_all_false():
    idx = pd.bdate_range("2020-01-02", periods=50, name="date")
    breadth = pd.DataFrame({"hindenburg": 0}, index=idx)
    clusters = st.cluster_signals(breadth)
    assert clusters.sum() == 0


# ---------------------------------------------------------------------------
# forward_returns
# ---------------------------------------------------------------------------
def test_forward_returns_shape(null_breadth):
    df, _ = null_breadth
    sig = st.cluster_signals(df)
    fwd = st.forward_returns(df, sig, horizons=(30, 60))
    assert "ret30" in fwd.columns
    assert "ret60" in fwd.columns
    # Each row is a signal cluster date
    assert len(fwd) == int(sig.sum())


def test_forward_returns_compounded_correctly():
    """A constant +1% daily return should compound to ~(1.01^h - 1) over h days."""
    n = 200
    idx = pd.bdate_range("2020-01-02", periods=n, name="date")
    breadth = pd.DataFrame({"spy_ret": 0.01, "hindenburg": 0}, index=idx)
    # Plant one signal at day 5
    breadth.iloc[5, breadth.columns.get_loc("hindenburg")] = 1
    sig = pd.Series(False, index=idx)
    sig.iloc[5] = True

    fwd = st.forward_returns(breadth, sig, horizons=(10,))
    expected = (1.01**10) - 1.0
    assert abs(fwd["ret10"].iloc[0] - expected) < 1e-6


def test_unconditional_base_excludes_signal_days(null_breadth):
    df, _ = null_breadth
    sig_dates = set(df.index[df["hindenburg"] == 1])
    base = st.unconditional_forward_returns(df, horizons=(30,), sample_every=21)
    assert len(set(base.index) & sig_dates) == 0, "base sample includes signal days"


# ---------------------------------------------------------------------------
# crash_rate
# ---------------------------------------------------------------------------
def test_crash_rate_plausible(null_breadth):
    df, _ = null_breadth
    sig = st.cluster_signals(df)
    cr = st.crash_rate(df, sig, horizon=120, drawdown_threshold=-0.05)
    assert 0.0 <= cr["signal_crash_rate"] <= 1.0
    assert 0.0 <= cr["base_crash_rate"] <= 1.0
    assert 0.0 <= cr["false_alarm_rate"] <= 1.0


def test_crash_rate_keys(null_breadth):
    df, _ = null_breadth
    sig = st.cluster_signals(df)
    cr = st.crash_rate(df, sig)
    for key in ("n_signals", "n_crashes_after_signal", "signal_crash_rate",
                "base_crash_rate", "false_alarm_rate", "welch_t"):
        assert key in cr, f"missing key: {key}"


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(null_breadth):
    df, _ = null_breadth
    s = st.summarize(df, horizons=(30, 60))
    for key in ("n_signal_days", "n_clusters", "n_base",
                "sig_mean_30d", "base_mean_30d", "t_hac_30d", "t_welch_30d",
                "signal_crash_rate", "false_alarm_rate"):
        assert key in s, f"missing summarize key: {key}"


def test_planted_crash_lowers_forward_return():
    """With a large planted crash, signal forward returns should be lower than the null."""
    df_null, _ = data.synthetic_breadth(n_stocks=80, n_days=504, crash_signal_bps=0.0, seed=42)
    df_sig, _ = data.synthetic_breadth(n_stocks=80, n_days=504, crash_signal_bps=200.0, seed=42)
    # Both same seed → same signal days; the only difference is the planted post-signal drift.
    sig = st.cluster_signals(df_null)
    if sig.sum() == 0:
        pytest.skip("no Hindenburg signals generated — threshold may not fire")
    fwd_null = st.forward_returns(df_null, sig, horizons=(30,))
    fwd_sig = st.forward_returns(df_sig, sig, horizons=(30,))
    # The planted version should be (on average) worse
    assert fwd_sig["ret30"].mean() < fwd_null["ret30"].mean(), (
        "Planted crash did not lower forward returns — check strategy.py"
    )

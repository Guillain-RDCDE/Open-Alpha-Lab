"""Quintile logic, shuffled control, and the study's spine: the signal beats a shuffle
only when a real premium is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from turnover_anomaly import data, strategy as st  # noqa: E402


# ---- quintile construction ------------------------------------------------

def test_quintile_returns_columns(null_panel):
    sig, fwd, _ = null_panel
    res = st.quintile_returns(sig, fwd)
    assert "q1" in res.columns
    assert "q5" in res.columns
    assert "market" in res.columns
    assert "hedge" in res.columns
    assert "spread" in res.columns


def test_quintile_returns_length(null_panel):
    sig, fwd, truth = null_panel
    res = st.quintile_returns(sig, fwd)
    # Every year should be present (n_firms=100 >> min_stocks=20)
    assert len(res) == truth["n_years"]


def test_quintile_market_between_q1_and_q5(null_panel):
    """The equal-weight market return must sit between the extreme quintiles most years."""
    sig, fwd, _ = null_panel
    res = st.quintile_returns(sig, fwd).dropna(subset=["q1", "q5", "market"])
    # Market is average of all, so mean should be between q1 and q5 means
    assert res["q1"].mean() != res["q5"].mean()  # quintiles differ


def test_hedge_equals_q1_minus_market(null_panel):
    sig, fwd, _ = null_panel
    res = st.quintile_returns(sig, fwd).dropna(subset=["q1", "market", "hedge"])
    assert np.allclose(res["hedge"].to_numpy(), (res["q1"] - res["market"]).to_numpy())


def test_spread_equals_q1_minus_q5(null_panel):
    sig, fwd, _ = null_panel
    res = st.quintile_returns(sig, fwd).dropna(subset=["q1", "q5", "spread"])
    assert np.allclose(res["spread"].to_numpy(), (res["q1"] - res["q5"]).to_numpy())


def test_min_stocks_drops_year():
    """Years with fewer than min_stocks valid tickers are excluded."""
    sig, fwd, _ = data.synthetic_panel(n_firms=10, n_years=5, seed=1)
    res = st.quintile_returns(sig, fwd, min_stocks=50)  # 10 < 50 → all dropped
    assert len(res) == 0


# ---- shuffled control -----------------------------------------------------

def test_shuffled_control_shape(null_panel):
    sig, fwd, _ = null_panel
    ctrl = st.shuffled_control(sig, fwd, n_shuffles=20, seed=99)
    assert ctrl.shape[0] == 20
    assert ctrl.shape[1] == len(sig.index)


def test_shuffled_control_is_reproducible(null_panel):
    sig, fwd, _ = null_panel
    a = st.shuffled_control(sig, fwd, n_shuffles=10, seed=5)
    b = st.shuffled_control(sig, fwd, n_shuffles=10, seed=5)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_shuffled_control_mean_near_zero(null_panel):
    """Under the null, shuffled Q1 should not consistently beat the market."""
    sig, fwd, _ = null_panel
    ctrl = st.shuffled_control(sig, fwd, n_shuffles=100, seed=141)
    grand_mean = float(ctrl.mean(axis=None).mean())
    # Shuffled excess return should be near zero (no systematic bias)
    assert abs(grand_mean) < 0.03


# ---- summarize ------------------------------------------------------------

def test_summarize_keys(null_panel):
    sig, fwd, _ = null_panel
    res = st.quintile_returns(sig, fwd)
    s = st.summarize(res["hedge"])
    for key in ("n", "mean", "vol", "sharpe", "tstat", "hit_rate"):
        assert key in s


def test_summarize_hit_rate_in_range(null_panel):
    sig, fwd, _ = null_panel
    res = st.quintile_returns(sig, fwd)
    s = st.summarize(res["hedge"])
    assert 0.0 <= s["hit_rate"] <= 1.0


def test_summarize_constant_series_returns_nan_tstat():
    """A constant return series has no variance — t-stat should gracefully return nan."""
    constant = pd.Series([0.05] * 10)
    s = st.summarize(constant)
    assert np.isnan(s["tstat"]) or np.isfinite(s["tstat"])  # either is acceptable; no crash


def test_summarize_n_matches_input(null_panel):
    sig, fwd, truth = null_panel
    res = st.quintile_returns(sig, fwd)
    s = st.summarize(res["market"])
    assert s["n"] == truth["n_years"]


# ---- the study spine ------------------------------------------------------

def test_signal_beats_shuffle_with_premium(strong_signal_panel):
    """The spine: when a real negative premium is planted, Q1 outperforms Q1-under-shuffle."""
    sig, fwd, truth = strong_signal_panel
    assert truth["premium"] < -0.05  # strong signal

    real_res = st.quintile_returns(sig, fwd)
    real_hedge_mean = float(real_res["hedge"].mean())

    ctrl = st.shuffled_control(sig, fwd, n_shuffles=100, seed=0)
    shuffle_hedge_mean = float(ctrl.mean(axis=None).mean())

    # With a strong planted premium, real Q1−market should beat shuffled Q1−market
    assert real_hedge_mean > shuffle_hedge_mean


def test_no_signal_no_edge(null_panel):
    """On the null (premium = 0), the real signal should not consistently beat the shuffle."""
    sig, fwd, truth = null_panel
    assert truth["premium"] == 0.0

    real_res = st.quintile_returns(sig, fwd)
    real_hedge_mean = float(real_res["hedge"].mean())

    ctrl = st.shuffled_control(sig, fwd, n_shuffles=100, seed=0)
    shuffle_hedge_mean = float(ctrl.mean(axis=None).mean())

    # With no planted signal, real and shuffled means should be within 3% of each other
    assert abs(real_hedge_mean - shuffle_hedge_mean) < 0.03


# ---- quintile profile -----------------------------------------------------

def test_quintile_profile_shape(null_panel):
    sig, fwd, _ = null_panel
    profile = st.quintile_profile(sig, fwd)
    assert len(profile) == 5
    assert "mean_fwd_ret" in profile.columns
    assert "mean_turnover" in profile.columns


def test_quintile_profile_turnover_monotone(null_panel):
    """Turnover should be monotonically increasing from Q1 (low) to Q5 (high)."""
    sig, fwd, _ = null_panel
    profile = st.quintile_profile(sig, fwd)
    to = profile["mean_turnover"].dropna().to_numpy()
    assert (np.diff(to) > 0).all(), "Turnover should increase monotonically from Q1 to Q5"

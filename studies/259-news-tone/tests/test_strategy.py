"""Signals, timing returns, costs, inference, and the study's spine: the tone signal
forecasts the next day ONLY when next-day predictability is planted; the within-month
placebo and the null both read zero."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from news_tone import data, strategy as st  # noqa: E402

CACHE_PATH = data.GSPC_CACHE


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def test_sign_signal_values(null_panel):
    df, _ = null_panel
    s = st.tone_sign_signal(df)
    assert set(np.unique(s.dropna())).issubset({-1.0, 0.0, 1.0})


def test_tilt_signal_bounded(null_panel):
    df, _ = null_panel
    s = st.tone_tilt_signal(df, clip=2.0)
    assert s.abs().max() <= 1.0 + 1e-12


def test_signal_deterministic(null_panel):
    df, _ = null_panel
    a = st.tone_sign_signal(df)
    b = st.tone_sign_signal(df)
    pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------------------
# Timing returns
# ---------------------------------------------------------------------------
def test_timing_returns_uses_forward(null_panel):
    df, _ = null_panel
    sig = st.tone_sign_signal(df)
    r = st.timing_returns(df, sig)
    # element-wise = sign * fwd_ret
    common = sig.index.intersection(df.index)
    expected = (sig.loc[common] * df.loc[common, "fwd_ret"]).dropna()
    pd.testing.assert_series_equal(r, expected.rename("strat_ret"))


def test_timing_returns_non_empty(null_panel):
    df, _ = null_panel
    r = st.timing_returns(df, st.tone_sign_signal(df))
    assert len(r) > 100


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def test_net_never_above_gross(null_panel):
    df, _ = null_panel
    sig = st.tone_sign_signal(df)
    gross = st.timing_returns(df, sig)
    net = st.net_of_costs(sig, gross, one_way_bps=2.0, borrow_bps_per_day=0.05)
    assert (net <= gross + 1e-12).all(), "Costs must not increase returns"


def test_zero_cost_equals_gross(null_panel):
    df, _ = null_panel
    sig = st.tone_sign_signal(df)
    gross = st.timing_returns(df, sig)
    net = st.net_of_costs(sig, gross, one_way_bps=0.0, borrow_bps_per_day=0.0)
    np.testing.assert_allclose(net.to_numpy(), gross.to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(300) * 0.01)
    s = st.summarize(r)
    assert set(["mean", "vol", "sharpe", "tstat", "hit_rate", "n"]).issubset(s)


def test_summarize_short_series():
    s = st.summarize(pd.Series([0.01, -0.02]))
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_summarize_tstat_direction():
    rng = np.random.default_rng(10)
    s = st.summarize(pd.Series(rng.normal(0.002, 0.005, 500)))
    assert s["tstat"] > 0


# ---------------------------------------------------------------------------
# Predictive regression
# ---------------------------------------------------------------------------
def test_regression_keys(null_panel):
    df, _ = null_panel
    reg = st.predictive_regression(df, target_col="fwd_ret")
    assert set(["beta", "tstat", "r2", "n"]).issubset(reg)


# ---------------------------------------------------------------------------
# The spine: gamma knob switches next-day predictability on/off
# ---------------------------------------------------------------------------
def test_null_panel_no_predictability(null_panel):
    """With gamma=0, tone must NOT forecast tomorrow's return."""
    df, _ = null_panel
    reg = st.predictive_regression(df, target_col="fwd_ret")
    assert abs(reg["tstat"]) < 2.5, f"Null next-day t should be small, got {reg['tstat']:.2f}"
    sig = st.tone_sign_signal(df)
    s = st.summarize(st.timing_returns(df, sig))
    assert abs(s["tstat"]) < 2.5, f"Null sign-timing t should be small, got {s['tstat']:.2f}"


def test_live_panel_has_predictability(live_panel):
    """With planted gamma, tone strongly forecasts tomorrow's return."""
    df, _ = live_panel
    reg = st.predictive_regression(df, target_col="fwd_ret")
    assert reg["beta"] > 0.0
    assert reg["tstat"] > 2.0, f"Planted next-day t should clear 2, got {reg['tstat']:.2f}"
    sig = st.tone_sign_signal(df)
    s = st.summarize(st.timing_returns(df, sig))
    assert s["tstat"] > 2.0, f"Planted sign-timing t should clear 2, got {s['tstat']:.2f}"


def test_synthetic_null_is_truly_null():
    """The null synthetic panel must not be a fluke at the chosen seed."""
    df, _ = data.synthetic_daily(n_days=2000, gamma=0.0, seed=259)
    reg = st.predictive_regression(df, target_col="fwd_ret")
    assert abs(reg["tstat"]) < 2.5


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="gspc_daily.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_within_month_demean_kills_signal():
    """The honesty spine: demeaning within the month must collapse the slope to ~0."""
    df = data.build_real_panel(fetch=False)
    d2 = df.copy()
    d2["ym"] = d2.index.to_period("M")
    d2["fwd_dm"] = d2["fwd_ret"] - d2.groupby("ym")["fwd_ret"].transform("mean")
    dm = st.predictive_regression(
        d2[["tone", "fwd_dm"]].rename(columns={"fwd_dm": "fwd_ret"}), target_col="fwd_ret"
    )
    assert abs(dm["tstat"]) < 0.5, f"Within-month demeaned t must be ~0, got {dm['tstat']:.3f}"


@requires_cache
def test_real_panel_strategy_runs():
    df = data.build_real_panel(fetch=False)
    sig = st.tone_sign_signal(df)
    s = st.summarize(st.timing_returns(df, sig))
    assert np.isfinite(s["tstat"]) and isinstance(s["tstat"], float)

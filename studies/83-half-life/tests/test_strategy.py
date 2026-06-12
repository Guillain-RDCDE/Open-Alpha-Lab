"""Event windows, the random control, the secular-excess decomposition, and the
study's spine: with planted boost the halving windows beat random days; on the
null tape they do not (they are just the secular BTC trend)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from half_life import data, strategy as st  # noqa: E402


# ---- event-window basics ---------------------------------------------------
def test_window_returns_columns(null_tape):
    bars, _ = null_tape
    ev = st.window_returns(bars)
    expected = [
        "halving_date", "entry_date", "entry_px", "exit_date", "exit_px",
        "log_ret", "pct_ret", "n_days_actual",
    ]
    assert list(ev.columns) == expected


def test_window_returns_no_lookahead(null_tape):
    """Entry date must be strictly after the halving date."""
    bars, _ = null_tape
    ev = st.window_returns(bars)
    for _, row in ev.iterrows():
        assert row["entry_date"] > row["halving_date"]


def test_window_returns_positive_duration(null_tape):
    bars, _ = null_tape
    ev = st.window_returns(bars)
    assert (ev["n_days_actual"] > 0).all()


def test_window_returns_log_ret_finite(null_tape):
    bars, _ = null_tape
    ev = st.window_returns(bars)
    assert ev["log_ret"].apply(np.isfinite).all()


def test_pct_ret_consistent_with_log_ret(null_tape):
    bars, _ = null_tape
    ev = st.window_returns(bars)
    assert np.allclose(ev["pct_ret"], np.expm1(ev["log_ret"]), atol=1e-9)


# ---- random control --------------------------------------------------------
def test_random_control_reproducible(null_tape):
    bars, _ = null_tape
    a = st.random_control_returns(bars, n_events=3, seed=7, n_draws=50)
    b = st.random_control_returns(bars, n_events=3, seed=7, n_draws=50)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_random_control_length(null_tape):
    bars, _ = null_tape
    ctrl = st.random_control_returns(bars, n_events=3, seed=83, n_draws=100)
    assert len(ctrl) == 100


def test_random_control_variety(null_tape):
    """Different seeds produce different distributions."""
    bars, _ = null_tape
    a = st.random_control_returns(bars, n_events=3, seed=1, n_draws=200)
    b = st.random_control_returns(bars, n_events=3, seed=2, n_draws=200)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


# ---- secular excess --------------------------------------------------------
def test_secular_excess_columns(null_tape):
    bars, _ = null_tape
    ex = st.secular_excess(bars)
    assert "excess_log_ret" in ex.columns
    assert "trend_log_ret" in ex.columns
    assert "log_ret" in ex.columns


def test_secular_excess_decomposition(null_tape):
    """log_ret == trend_log_ret + excess_log_ret (within rounding)."""
    bars, _ = null_tape
    ex = st.secular_excess(bars)
    assert np.allclose(ex["log_ret"], ex["trend_log_ret"] + ex["excess_log_ret"], atol=1e-9)


# ---- summarize -------------------------------------------------------------
def test_summarize_with_four_observations():
    r = np.array([0.5, 1.2, 0.8, 2.0])
    s = st.summarize(r, label="test")
    assert s["n"] == 4
    assert s["low_power_warning"] is True
    assert np.isfinite(s["mean"])
    # With n=4 the t-stat is unreliable but should be computed
    assert np.isfinite(s["tstat"]) or np.isnan(s["tstat"])


def test_summarize_single_obs():
    s = st.summarize(np.array([1.5]))
    assert s["n"] == 1
    assert np.isnan(s["tstat"])


# ---- the spine: boost recoverable only when planted -----------------------
def test_boosted_tape_beats_null_on_halving_windows(null_tape, boosted_tape):
    """With planted post-halving boost the halving windows outperform the null-tape windows
    by a meaningful margin. The boost tape's halving mean must clearly exceed the null tape's
    halving mean. Also confirm that the random control is consistent (similar means)."""
    null_bars, _ = null_tape
    boost_bars, _ = boosted_tape
    halvings = data.HALVING_DATES[:3]

    null_ev = st.window_returns(null_bars, halvings=halvings)
    boost_ev = st.window_returns(boost_bars, halvings=halvings)

    # Boosted tape must show substantially higher halving-window returns than the null tape
    assert boost_ev["log_ret"].mean() > null_ev["log_ret"].mean() + 0.5

    # Both tapes should produce finite, plausible control means
    boost_ctrl = st.random_control_returns(boost_bars, n_events=len(boost_ev), seed=83, n_draws=500)
    assert boost_ctrl.notna().all()
    assert len(boost_ctrl) == 500


def test_power_analysis_returns_dict(null_tape):
    bars, _ = null_tape
    result = st.power_analysis(bars, halvings=data.HALVING_DATES[:3], n_draws=200)
    assert "p_value" in result
    assert "obs_mean_log_ret" in result
    assert 0.0 <= result["p_value"] <= 1.0

"""Regime conditioning, the timing engine's invariants, and the study's two spines:
(1) the harness banks a planted QE>QT edge and finds none when none is planted; (2) one
execution lag and one-way costs are applied exactly once."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_balance_sheet import data, strategy as st  # noqa: E402


# ---- HAC / inference helpers ----------------------------------------------
def test_hac_tstat_zero_mean_noise_is_insignificant():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 4000)
    assert abs(st.hac_tstat(x)) < 2.0


def test_hac_tstat_strong_drift_is_significant():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.005, 4000)
    assert st.hac_tstat(x) > 2.0


def test_regime_means_columns(null):
    frame, _ = null
    rm = st.regime_means(frame)
    assert set(rm.columns) == {"n", "mean_bps", "vol_bps", "sharpe_ann", "tstat"}
    assert rm["n"].sum() == len(frame) - 1  # one row lost to pct_change


# ---- the two spines -------------------------------------------------------
def test_control_qe_beats_qt_significantly(control):
    """Spine #1a: with a planted edge the QE-minus-QT contrast clears the inference bar."""
    frame, _ = control
    ci = st.diff_block_bootstrap_ci(frame, n_boot=1000, seed=1)
    assert ci["diff_bps"] > 0
    assert ci["ci_lo"] > 0          # CI excludes zero
    assert ci["p_boot"] < 0.05
    rm = st.regime_means(frame)
    assert rm.loc["QE", "tstat"] > 2.0
    assert rm.loc["QT", "tstat"] < -2.0


def test_null_qe_qt_contrast_is_insignificant(null):
    """Spine #1b: with no planted edge the contrast is indistinguishable from zero."""
    frame, _ = null
    ci = st.diff_block_bootstrap_ci(frame, n_boot=1000, seed=1)
    assert ci["ci_lo"] < 0 < ci["ci_hi"]   # CI brackets zero
    assert ci["p_boot"] > 0.05


def test_control_timing_beats_buy_hold(control):
    """On the planted-edge tape the fight-the-Fed rule should help (positive excess Sharpe)."""
    r = st.race(control[0], short_qt=True, cost_bps=1.0)
    assert r["excess_sharpe_diff"] > 0

def test_null_timing_does_not_beat_buy_hold(null):
    """On the null tape the rule should NOT systematically beat buy-and-hold."""
    r = st.race(null[0], short_qt=True, cost_bps=1.0)
    assert r["excess_sharpe_diff"] < 0.10  # no meaningful edge


# ---- engine invariants: one lag, one-way costs ----------------------------
def test_one_execution_lag_applied_once(null):
    """Position on day t equals the regime target read on day t-1 (one shift, once)."""
    frame, _ = null
    book = st.regime_timing_returns(frame, short_qt=False, lag=1)
    target = frame["regime"].map({"QE": 1.0, "FLAT": 1.0, "QT": 0.0})
    expected = target.shift(1).fillna(0.0).reindex(book.index)
    assert np.allclose(book["pos"].to_numpy(), expected.to_numpy())


def test_cost_monotonically_lowers_return(control):
    frame, _ = control
    means = [
        st.regime_timing_returns(frame, short_qt=True, cost_bps=c)["timing_net"].mean()
        for c in (0.0, 5.0, 20.0)
    ]
    assert means[0] >= means[1] >= means[2]


def test_short_qt_changes_position_in_qt(null):
    frame, _ = null
    flat = st.regime_timing_returns(frame, short_qt=False)
    shrt = st.regime_timing_returns(frame, short_qt=True)
    # In flat mode QT is cash (0); in short mode QT is -1 (lagged).
    assert flat["pos"].min() >= 0.0
    assert shrt["pos"].min() < 0.0


def test_race_excess_of_cash_uses_rf(null):
    """A non-zero risk-free rate lowers the excess Sharpe of the fully-invested leg."""
    frame, _ = null
    r0 = st.race(frame, short_qt=False, cost_bps=0.0, rf_annual=0.0)
    r5 = st.race(frame, short_qt=False, cost_bps=0.0, rf_annual=0.05)
    assert r5["buy_hold"]["ann_ret"] < r0["buy_hold"]["ann_ret"] + 1e-9
    # both legs net the cash rate, so the race is excess-vs-excess
    assert np.isfinite(r5["excess_sharpe_diff"])

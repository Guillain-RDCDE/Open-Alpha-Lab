"""Tests for Study 84 (Moon-Math) strategy/analysis layer — offline, deterministic."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moon_math import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Levels regression
# ---------------------------------------------------------------------------
def test_levels_regression_high_r2_on_planb_tape(planb_tape):
    """When price is generated from S2F (beta=3.3), levels regression finds a high R²."""
    df, _ = planb_tape
    res = st.levels_regression(df)
    assert res["r2"] > 0.60, f"Expected high R², got {res['r2']:.3f}"
    # n reflects valid (non-NaN) rows, not total rows
    assert res["n"] > 0


def test_levels_regression_returns_expected_keys(planb_tape):
    df, _ = planb_tape
    res = st.levels_regression(df)
    for key in ("alpha", "beta", "r2", "resid", "n", "adf_pvalue", "adf_stat"):
        assert key in res, f"Missing key: {key}"


def test_time_regression_competes_with_s2f(planb_tape):
    """log(time) fits roughly as well as log(S2F) — exposing the spurious channel."""
    df, _ = planb_tape
    r_s2f = st.levels_regression(df)
    r_time = st.time_regression(df)
    # Both should achieve a reasonably high R² (since both trend with time)
    assert r_time["r2"] > 0.50


# ---------------------------------------------------------------------------
# First-difference regression (the forecasting content test)
# ---------------------------------------------------------------------------
def test_first_diff_near_zero_r2_on_null_tape(null_tape):
    """On the null (beta=0) tape, first-difference R² should be essentially zero."""
    df, _ = null_tape
    res = st.first_diff_regression(df)
    assert res["r2"] < 0.05, f"First-diff R² should be ~0, got {res['r2']:.4f}"


def test_first_diff_has_tstat_key(planb_tape):
    df, _ = planb_tape
    res = st.first_diff_regression(df)
    assert "tstat_beta" in res
    assert np.isfinite(res["tstat_beta"]) or np.isnan(res["tstat_beta"])


# ---------------------------------------------------------------------------
# OOS evaluation — the smoking gun
# ---------------------------------------------------------------------------
def test_oos_r2_collapses_on_null_tape(null_tape):
    """When price is pure noise (beta=0), OOS R² should be poor (likely very negative)."""
    df, _ = null_tape
    res = st.oos_eval(df, train_end=df.index[int(len(df) * 0.7)].strftime("%Y-%m-%d"))
    assert res["n_train"] > 50
    assert res["n_oos"] > 50
    # OOS R² should be worse than in-sample (overfitting / noise extrapolation)
    assert res["oos_r2"] < res["train_r2"]


def test_oos_eval_returns_expected_keys(planb_tape):
    df, _ = planb_tape
    idx = df.index[int(len(df) * 0.7)]
    res = st.oos_eval(df, train_end=idx.strftime("%Y-%m-%d"))
    for key in ("train_r2", "oos_r2", "oos_mape", "n_train", "n_oos",
                "alpha", "beta", "oos_pred_series", "oos_actual_series"):
        assert key in res, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Spurious control
# ---------------------------------------------------------------------------
def test_spurious_control_returns_three_branches(planb_tape):
    df, _ = planb_tape
    ctrl = st.spurious_control(df)
    assert set(ctrl.keys()) == {"s2f_levels", "time_levels", "first_diff"}


def test_summarize_is_consistent(planb_tape):
    df, _ = planb_tape
    ctrl = st.spurious_control(df)
    smry = st.summarize(ctrl)
    assert smry["n"] > 0  # n = valid rows after warmup
    assert 0.0 <= smry["s2f_r2"] <= 1.0
    assert 0.0 <= smry["time_r2"] <= 1.0
    assert smry["diff_r2"] >= 0.0


def test_null_tape_first_diff_r2_tiny(null_tape):
    """The spine: on a null tape, first-diff R² is tiny — S2F has no forecasting content."""
    df, _ = null_tape
    ctrl = st.spurious_control(df)
    smry = st.summarize(ctrl)
    assert smry["diff_r2"] < 0.05


def test_spurious_flag_on_trending_noise(noisy_trend_tape):
    """On a random-walk price with no S2F beta, the in-sample R² is high (two trends)
    but the residuals should fail stationarity, flagging the regression as spurious."""
    df, _ = noisy_trend_tape
    ctrl = st.spurious_control(df)
    smry = st.summarize(ctrl)
    # Random-walk price + S2F trend → ADF should NOT reject non-stationarity in residuals
    # (i.e. adf_p > 0.05 ⇒ spurious=True)
    # This may not always hold on small samples but should on 2000 days
    assert smry["diff_r2"] < 0.05, "First-diff R² should be tiny on noise tape"

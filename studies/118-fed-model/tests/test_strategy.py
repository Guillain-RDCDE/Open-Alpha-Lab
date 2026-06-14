"""OLS/HAC engine invariants, OOS R², timing strategy, and the study's spine:
the regression finds a slope only when predictability is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_model import strategy as st  # noqa: E402


# ---- OLS/HAC engine -------------------------------------------------------

def test_ols_hac_recovers_known_slope():
    """OLS on a textbook linear relationship should return the true slope."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = 0.7 * x + 0.1 * rng.standard_normal(500)
    res = st.ols_hac(x, y)
    assert abs(res["slope"] - 0.7) < 0.1
    assert res["r2"] > 0.8


def test_ols_hac_null_has_small_tstat():
    """On a null (zero slope) with finite sample the t-stat should be small."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(300)
    y = rng.standard_normal(300)
    res = st.ols_hac(x, y)
    assert abs(res["tstat"]) < 4.0  # could be up to ~3 by chance but not reliably big


def test_ols_hac_returns_finite_values(null_panel):
    from fed_model import data
    panel, _ = null_panel
    sub = panel.dropna(subset=["fed_spread", "fwd10"])
    res = st.ols_hac(sub["fed_spread"].values, sub["fwd10"].values)
    for k in ("slope", "intercept", "r2", "tstat", "n"):
        assert np.isfinite(res[k]), f"key {k} is not finite: {res[k]}"


# ---- OOS R² ---------------------------------------------------------------

def test_oos_r2_signal_panel_positive(signal_panel):
    """With a strong planted signal the OOS R² should be positive."""
    from fed_model import data
    panel, _ = signal_panel
    # signal_panel has 1200 months from 1900; split at 1965 gives ~780 train / ~420 test
    result = st.oos_r2(panel, "fed_spread", "fwd10", train_end_year=1965)
    assert result["oos_r2"] > 0.0


def test_oos_r2_null_panel_near_zero_or_negative(null_panel):
    """With no planted signal the OOS R² should not be meaningfully positive."""
    from fed_model import data
    panel, _ = null_panel
    # null_panel has 600 months from 1900-01; split at 1940 gives ~480 train / ~120 test
    result = st.oos_r2(panel, "fed_spread", "fwd10", train_end_year=1940)
    # nan is returned if there's no test data; treat nan as non-positive
    if result["oos_r2"] != result["oos_r2"]:  # nan check
        return
    assert result["oos_r2"] < 0.3  # could be slightly positive by chance


def test_oos_r2_output_keys():
    from fed_model import data
    panel, _ = data.synthetic_panel(n_months=800, predicts=0.5, seed=5)
    result = st.oos_r2(panel, "fed_spread", "fwd10", train_end_year=1950)
    assert "oos_r2" in result
    assert "n_train" in result
    assert "n_test" in result


# ---- Timing strategy ------------------------------------------------------

def test_timing_returns_columns(signal_panel):
    panel, _ = signal_panel
    sub = panel.dropna(subset=["fed_spread", "fwd1"])
    result = st.timing_returns(sub, "fed_spread", "fwd1")
    for col in ("signal", "position", "ret_strat", "ret_bh", "cum_strat", "cum_bh"):
        assert col in result.columns


def test_timing_position_is_binary(signal_panel):
    panel, _ = signal_panel
    sub = panel.dropna(subset=["fed_spread", "fwd1"])
    result = st.timing_returns(sub, "fed_spread", "fwd1")
    assert set(result["position"].unique()).issubset({0, 1})


def test_timing_cumulative_is_positive(signal_panel):
    panel, _ = signal_panel
    sub = panel.dropna(subset=["fed_spread", "fwd1"])
    result = st.timing_returns(sub, "fed_spread", "fwd1")
    assert (result["cum_strat"] > 0).all()
    assert (result["cum_bh"] > 0).all()


# ---- The spine ------------------------------------------------------------

def test_signal_panel_has_higher_r2_than_null(null_panel, signal_panel):
    """The OLS regression should find a higher R² when predictability is planted."""
    pan_null, _ = null_panel
    pan_sig, _ = signal_panel

    sub_null = pan_null.dropna(subset=["fed_spread", "fwd10"])
    sub_sig = pan_sig.dropna(subset=["fed_spread", "fwd10"])

    res_null = st.ols_hac(sub_null["fed_spread"].values, sub_null["fwd10"].values)
    res_sig = st.ols_hac(sub_sig["fed_spread"].values, sub_sig["fwd10"].values)

    assert res_sig["r2"] > res_null["r2"]
    assert res_sig["tstat"] > res_null["tstat"]

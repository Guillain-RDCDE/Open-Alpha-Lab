"""Regression engine invariants, HAC t-stat, OOS R², and the study's spine:
predictive regression finds a link only when one is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dollar_smile import strategy as st  # noqa: E402


# ---- feature engineering --------------------------------------------------
def test_compute_features_columns(null_tape):
    daily, _ = null_tape
    feat = st.compute_features(daily, horizon="W")
    for col in ["delta_dollar", "ret_spy", "ret_eem", "fwd_spy", "fwd_eem"]:
        assert col in feat.columns


def test_no_lookahead(null_tape):
    """fwd_spy at time t is a *shifted* series — it must not align with ret_spy at t."""
    daily, _ = null_tape
    feat = st.compute_features(daily, horizon="W")
    # The forward return at t must equal the contemporaneous return at t+1.
    fwd = feat["fwd_spy"].dropna()
    ret = feat["ret_spy"].dropna()
    # Align: fwd[i] should equal ret[i+1]
    # Check on the first 10 overlapping periods
    for i in range(10):
        if i + 1 < len(ret):
            t_idx = ret.index[i + 1]
            if t_idx in fwd.index:
                j = fwd.index.get_loc(t_idx) - 1
                if j >= 0 and j < len(fwd):
                    assert abs(fwd.iloc[j] - ret.iloc[i + 1]) < 1e-10


def test_features_no_nan(null_tape):
    daily, _ = null_tape
    feat = st.compute_features(daily, horizon="W")
    assert not feat["delta_dollar"].isna().any()
    assert not feat["fwd_spy"].isna().any()


# ---- regression IS --------------------------------------------------------
def test_regression_is_returns_all_keys(null_tape):
    daily, _ = null_tape
    feat = st.compute_features(daily, horizon="W")
    res = st.regression_is(feat["delta_dollar"], feat["fwd_spy"])
    for key in ("alpha", "beta", "r2", "n", "tstat", "se_hac", "tstat_naive"):
        assert key in res


def test_regression_is_r2_nonneg(null_tape):
    daily, _ = null_tape
    feat = st.compute_features(daily, horizon="W")
    res = st.regression_is(feat["delta_dollar"], feat["fwd_spy"])
    assert res["r2"] >= 0.0
    assert 0 < res["n"]


def test_regression_finds_planted_signal(predicted_tape):
    """When pred_r is strongly negative, the IS regression should recover a negative beta."""
    daily, truth = predicted_tape
    feat = st.compute_features(daily, horizon="W")
    res = st.regression_is(feat["delta_dollar"], feat["fwd_spy"])
    # The beta should be negative (dollar up -> equity down)
    assert res["beta"] < 0.0
    # And the t-stat should be negative (the planted effect has the right sign)
    assert res["tstat"] < 0.0


def test_contemporaneous_finds_link(predicted_tape):
    """A strong contemp_r link should yield a significantly negative contemporaneous t-stat."""
    daily, _ = predicted_tape
    feat = st.compute_features(daily, horizon="W")
    res = st.regression_is(feat["delta_dollar"], feat["ret_spy"])
    assert res["beta"] < 0.0
    assert res["tstat"] < -2.0


# ---- OOS R² ---------------------------------------------------------------
def test_oos_r2_returns_all_keys(null_tape):
    daily, _ = null_tape
    feat = st.compute_features(daily, horizon="W")
    res = st.oos_r2(feat["delta_dollar"], feat["fwd_spy"])
    for key in ("r2_oos", "n_oos", "msfe_model", "msfe_mean", "dm_tstat"):
        assert key in res


def test_oos_r2_null_model_nonpositive(null_tape):
    """With no planted link the OOS R² should be near-zero or negative."""
    daily, _ = null_tape
    feat = st.compute_features(daily, horizon="W")
    res = st.oos_r2(feat["delta_dollar"], feat["fwd_spy"])
    # On a null tape the model does not consistently beat the mean
    assert res["r2_oos"] < 0.05


# ---- signal and summarize -------------------------------------------------
def test_signal_is_binary(null_tape):
    daily, _ = null_tape
    feat = st.compute_features(daily, horizon="W")
    sig = st.dollar_signal(feat)
    assert set(sig.unique()) <= {-1, 1}


def test_trade_returns_columns(null_tape):
    daily, _ = null_tape
    feat = st.compute_features(daily, horizon="W")
    ledger = st.trade_returns(feat, target="fwd_spy", cost_bps=2.0)
    for col in ("signal", "fwd_ret", "ret_gross", "ret_net"):
        assert col in ledger.columns


def test_net_less_than_gross(null_tape):
    daily, _ = null_tape
    feat = st.compute_features(daily, horizon="W")
    ledger = st.trade_returns(feat, cost_bps=2.0)
    # Net should always be below gross by exactly the cost (2 bps = 2e-4)
    diff = ledger["ret_gross"] - ledger["ret_net"]
    assert (diff.abs() - 2e-4).abs().max() < 1e-10


def test_summarize_keys(null_tape):
    daily, _ = null_tape
    feat = st.compute_features(daily, horizon="W")
    ledger = st.trade_returns(feat, cost_bps=2.0)
    s = st.summarize(ledger)
    for key in ("n_periods", "win_rate", "mean_bps", "sharpe", "tstat"):
        assert key in s


def test_run_all_keys(null_tape):
    daily, _ = null_tape
    res = st.run_all(daily, horizon="W")
    for key in ("contemp_spy", "contemp_eem", "is_spy", "is_eem",
                "oos_spy", "oos_eem", "signal_spy", "signal_eem",
                "n_periods", "horizon"):
        assert key in res

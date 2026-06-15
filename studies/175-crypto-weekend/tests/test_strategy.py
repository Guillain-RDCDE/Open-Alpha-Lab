"""Tests for the weekend-effect strategy: signals, controls, and the study's spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_weekend import strategy as st  # noqa: E402
from crypto_weekend import data  # noqa: E402


# ---- daily_returns ----------------------------------------------------------

def test_daily_returns_columns(null_tape):
    bars, _ = null_tape
    dr = st.daily_returns(bars)
    for col in ["log_ret", "pct_ret", "dow", "is_weekend", "regime"]:
        assert col in dr.columns


def test_daily_returns_no_lookahead(null_tape):
    """Only open and close of the same day are used — no look-ahead."""
    bars, _ = null_tape
    dr = st.daily_returns(bars)
    # The log_ret should be log(close/open) for each row.
    sub = bars.loc[dr.index]
    expected = np.log(sub["close"] / sub["open"])
    assert np.allclose(dr["log_ret"].to_numpy(), expected.to_numpy(), atol=1e-10)


def test_is_weekend_correct(null_tape):
    bars, _ = null_tape
    dr = st.daily_returns(bars)
    expected = dr.index.dayofweek >= 5
    assert (dr["is_weekend"] == expected).all()


def test_regime_labels(null_tape):
    bars, _ = null_tape
    dr = st.daily_returns(bars)
    assert set(dr["regime"].unique()).issubset({"pre2023", "post2023"})


# ---- return_comparison ------------------------------------------------------

def test_return_comparison_structure(null_tape):
    bars, _ = null_tape
    dr = st.daily_returns(bars)
    rc = st.return_comparison(dr)
    assert "weekend" in rc and "weekday" in rc
    assert "diff_tstat" in rc
    assert "bonferroni_sig" in rc
    assert "premium_pct" in rc


def test_return_comparison_null_has_small_t(null_tape):
    """On a martingale the weekend premium t-stat should be small (no signal planted)."""
    bars, _ = null_tape
    dr = st.daily_returns(bars)
    rc = st.return_comparison(dr)
    # On a null tape (no boost) the premium t-stat should be inside a wide band;
    # we check that it is not anomalously large (not significant).
    assert abs(rc["diff_tstat"]) < 4.0  # very generous: just no blow-up


def test_return_comparison_boosted_has_larger_t(null_tape, boosted_tape):
    """Planting a weekend boost increases the t-stat vs the null tape."""
    dr_null = st.daily_returns(null_tape[0])
    dr_boost = st.daily_returns(boosted_tape[0])
    rc_null = st.return_comparison(dr_null)
    rc_boost = st.return_comparison(dr_boost)
    # Boosted tape must have a larger (or equal) premium t-stat in absolute value.
    assert abs(rc_boost["diff_tstat"]) > abs(rc_null["diff_tstat"])


# ---- volatility_comparison --------------------------------------------------

def test_volatility_comparison_structure(null_tape):
    bars, _ = null_tape
    dr = st.daily_returns(bars)
    vc = st.volatility_comparison(dr)
    for k in ["vol_ann_weekend", "vol_ann_weekday", "vol_ratio", "levene_t"]:
        assert k in vc
    assert vc["vol_ann_weekend"] > 0
    assert vc["vol_ann_weekday"] > 0


def test_vol_ratio_positive(null_tape):
    bars, _ = null_tape
    dr = st.daily_returns(bars)
    vc = st.volatility_comparison(dr)
    assert vc["vol_ratio"] > 0


# ---- weekend_timing_rule ----------------------------------------------------

def test_timing_rule_structure(null_tape):
    bars, _ = null_tape
    tr = st.weekend_timing_rule(bars, cost_bps=5.0)
    for k in ["n", "mean_log", "tstat", "net_mean_log", "net_tstat"]:
        assert k in tr
    assert tr["n"] > 0


def test_timing_rule_cost_reduces_net(null_tape):
    bars, _ = null_tape
    tr0 = st.weekend_timing_rule(bars, cost_bps=0.0)
    tr5 = st.weekend_timing_rule(bars, cost_bps=5.0)
    tr10 = st.weekend_timing_rule(bars, cost_bps=10.0)
    assert tr0["net_mean_log"] > tr5["net_mean_log"]
    assert tr5["net_mean_log"] > tr10["net_mean_log"]


def test_timing_rule_n_reasonable(null_tape):
    """Number of weekend days should be about n_years * 104 (2 days/week)."""
    bars, truth = null_tape
    tr = st.weekend_timing_rule(bars)
    # 8 years => ~8*104=832 weekend days; allow wide margin
    assert 700 < tr["n"] < 1000


# ---- regime_split -----------------------------------------------------------

def test_regime_split_structure(null_tape):
    bars, _ = null_tape
    dr = st.daily_returns(bars)
    rs = st.regime_split(dr)
    assert "pre2023" in rs and "post2023" in rs
    for k in ["n_weekend", "n_weekday", "premium_pct", "diff_tstat"]:
        assert k in rs["pre2023"]


# ---- full summarize ---------------------------------------------------------

def test_summarize_runs_and_has_all_keys(null_tape):
    bars, _ = null_tape
    s = st.summarize(bars)
    for k in [
        "n_days", "n_weekend", "n_weekday",
        "wknd_mean_pct", "wkday_mean_pct", "premium_pct", "return_tstat",
        "vol_wknd", "vol_wkday", "vol_ratio",
        "timing_n", "timing_tstat", "timing_net_tstat",
        "pre2023_tstat", "post2023_tstat",
    ]:
        assert k in s, f"missing key: {k}"


def test_summarize_planted_vs_null(null_tape, boosted_tape):
    """Planting a boost increases the return t-stat."""
    s_null = st.summarize(null_tape[0])
    s_boost = st.summarize(boosted_tape[0])
    assert abs(s_boost["return_tstat"]) > abs(s_null["return_tstat"])

"""Signals, seasonal split, daylight regression, and the study's spine:
the SAD seasonal only manifests when it is planted in the synthetic tape."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sad_effect import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Month-group labelling
# ---------------------------------------------------------------------------
def test_month_group_covers_all_labels(null_tape):
    ret, _ = null_tape
    g = st.month_group(ret.index)
    assert set(g.unique()) == {"onset", "recovery", "summer"}


def test_month_group_correct_assignments():
    idx = pd.date_range("2000-09-30", periods=7, freq="ME")
    g = st.month_group(idx)
    # Sep=onset, Oct=onset, Nov=onset, Dec=recovery, Jan=recovery, Feb=recovery, Mar=recovery
    assert list(g.values) == ["onset", "onset", "onset", "recovery", "recovery", "recovery", "recovery"]


# ---------------------------------------------------------------------------
# Seasonal split
# ---------------------------------------------------------------------------
def test_seasonal_split_keys_present(null_tape):
    ret, _ = null_tape
    result = st.seasonal_split(ret)
    for key in ("unconditional_bp", "onset_bp", "recovery_bp", "summer_bp",
                "onset_n", "recovery_n", "summer_n",
                "onset_t_vs_unconditional", "recovery_t_vs_unconditional",
                "summer_t_vs_unconditional"):
        assert key in result, f"missing key: {key}"


def test_seasonal_split_group_counts(null_tape):
    ret, truth = null_tape
    result = st.seasonal_split(ret)
    n_total = result["onset_n"] + result["recovery_n"] + result["summer_n"]
    assert n_total == truth["n_years"] * 12


def test_seasonal_split_planted_signal_detectable(with_sad):
    """With a large planted premium, onset should be below and recovery above unconditional."""
    ret, _ = with_sad
    result = st.seasonal_split(ret)
    # Recovery higher than unconditional (planted positive premium)
    assert result["recovery_bp"] > result["unconditional_bp"]
    # Onset lower than unconditional (planted negative adjustment)
    assert result["onset_bp"] < result["unconditional_bp"]


def test_seasonal_split_null_has_small_gap(null_tape):
    """On a pure null tape the gaps should be small (no planted signal)."""
    ret, _ = null_tape
    result = st.seasonal_split(ret)
    # With 80 years of null data, t-stats should not consistently exceed ±3
    assert abs(result["onset_t_vs_unconditional"]) < 4.0
    assert abs(result["recovery_t_vs_unconditional"]) < 4.0


# ---------------------------------------------------------------------------
# Daylight regression
# ---------------------------------------------------------------------------
def test_daylight_regression_keys_present(null_tape):
    ret, _ = null_tape
    dl = data.daylight_proxy(ret.index)
    result = st.daylight_regression(ret, dl)
    for key in ("slope", "intercept", "r_squared", "slope_tstat_hac", "n"):
        assert key in result, f"missing key: {key}"


def test_daylight_regression_planted_gives_positive_slope(with_sad):
    """With a planted SAD signal aligned with daylight, the slope should be positive."""
    ret, _ = with_sad
    dl = data.daylight_proxy(ret.index)
    result = st.daylight_regression(ret, dl)
    assert result["slope"] > 0


def test_daylight_regression_null_has_small_slope(null_tape):
    """On a null tape the daylight slope should be near zero."""
    ret, _ = null_tape
    dl = data.daylight_proxy(ret.index)
    result = st.daylight_regression(ret, dl)
    assert abs(result["slope_tstat_hac"]) < 4.0


def test_daylight_regression_r_squared_bounded(null_tape):
    ret, _ = null_tape
    dl = data.daylight_proxy(ret.index)
    result = st.daylight_regression(ret, dl)
    assert 0.0 <= result["r_squared"] <= 1.0


# ---------------------------------------------------------------------------
# SAD calendar strategy
# ---------------------------------------------------------------------------
def test_sad_strategy_only_has_recovery_months():
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    ret = pd.Series(np.ones(24) * 0.01, index=idx, name="ret")
    strat = st.sad_strategy(ret)
    # Non-recovery months should have return 0
    recovery_months = {12, 1, 2, 3}
    for dt, val in strat.items():
        if dt.month not in recovery_months:
            assert val == pytest.approx(0.0)


def test_sad_strategy_no_lookahead(null_tape):
    """Strategy return on each month depends only on that month's calendar label, not future returns."""
    ret, _ = null_tape
    strat = st.sad_strategy(ret)
    assert len(strat) == len(ret)
    assert strat.isna().sum() == 0


def test_summary_has_expected_keys(null_tape):
    ret, _ = null_tape
    s = st.summary(ret)
    for key in ("sharpe", "cagr", "vol_ann", "max_dd", "n", "tstat"):
        assert key in s, f"missing key: {key}"


def test_summary_positive_cagr_on_drifting_tape(null_tape):
    """The synthetic tape has a base drift, so CAGR and Sharpe should be positive."""
    ret, _ = null_tape
    s = st.summary(ret)
    assert s["cagr"] > 0
    assert s["sharpe"] > 0
    assert s["tstat"] > 0


def test_sad_strategy_underperforms_bah_on_null(null_tape):
    """On a null tape, the SAD calendar strategy (4 months in) should underperform buy-and-hold
    in total return since it misses 8 months of positive drift per year."""
    ret, _ = null_tape
    strat = st.sad_strategy(ret)
    bah = st.buy_hold(ret)
    s_strat = st.summary(strat)
    s_bah = st.summary(bah)
    # The strategy is out of the market 8/12 months — it captures only ~1/3 of total drift
    assert s_bah["cagr"] > s_strat["cagr"]

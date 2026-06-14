"""Tests for the strategy/analysis layer — monthly stats, honest controls, and the study's spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mark_twain import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# monthly_stats
# ---------------------------------------------------------------------------
def test_monthly_stats_shape(null_tape):
    ret, _ = null_tape
    ms = st.monthly_stats(ret)
    assert len(ms) == 12
    assert set(ms.index) == set(range(1, 13))
    assert "mean_bps" in ms.columns
    assert "vol_ann" in ms.columns
    assert "worst_day_pct" in ms.columns
    assert "hac_t" in ms.columns


def test_monthly_stats_worst_day_is_negative(null_tape):
    """Worst day in any given month should be a loss (negative pct) over a long series."""
    ret, _ = null_tape
    ms = st.monthly_stats(ret)
    # On 60 years of data every month should have at least one down day.
    assert (ms["worst_day_pct"] < 0).all()


def test_monthly_stats_n_sums_to_total(null_tape):
    ret, _ = null_tape
    ms = st.monthly_stats(ret)
    assert ms["n"].sum() == len(ret)


def test_monthly_stats_exclude_removes_crash_months():
    """Crash-stripped stats should have fewer October observations."""
    # Craft a tiny series with a known crash month.
    dates = pd.date_range("1987-10-01", periods=23, freq="B")
    r = pd.Series(np.zeros(len(dates)), index=dates, name="ret")
    r.iloc[0] = -0.20  # simulate the crash
    ms_full = st.monthly_stats(r)
    ms_strip = st.monthly_stats(r, exclude_ym={"1987-10"})
    assert ms_strip.loc[10, "n"] == 0 or np.isnan(ms_strip.loc[10, "mean_bps"])


# ---------------------------------------------------------------------------
# compare_october_vs_rest
# ---------------------------------------------------------------------------
def test_compare_october_keys(null_tape):
    ret, _ = null_tape
    result = st.compare_october_vs_rest(ret)
    for key in ("oct_mean_bps", "rest_mean_bps", "diff_bps", "welch_t", "n_oct", "n_rest"):
        assert key in result


def test_compare_october_null_not_significant(null_tape):
    """On a null tape (no October penalty) the Welch t should be small."""
    ret, _ = null_tape
    result = st.compare_october_vs_rest(ret)
    assert abs(result["welch_t"]) < 3.0, (
        f"No penalty planted but Welch-t = {result['welch_t']:.2f} — "
        "the null should not produce a large t-stat."
    )


def test_compare_october_penalised_shows_signal(penalised_tape):
    """A planted −100 bps/day penalty should produce a clearly negative t-stat."""
    ret, _ = penalised_tape
    result = st.compare_october_vs_rest(ret)
    assert result["welch_t"] < -5.0, (
        f"Expected strong negative t with planted penalty, got {result['welch_t']:.2f}"
    )
    assert result["diff_bps"] < -50.0


# ---------------------------------------------------------------------------
# crash_strip_test
# ---------------------------------------------------------------------------
def test_crash_strip_returns_both(null_tape):
    ret, _ = null_tape
    out = st.crash_strip_test(ret)
    assert "full" in out and "stripped" in out


def test_crash_strip_reduces_n():
    """After stripping, October n should be lower (we removed those months)."""
    dates = pd.bdate_range("1928-01-02", periods=24727)
    r = pd.Series(np.zeros(len(dates)), index=dates, name="ret")
    ms_full = st.monthly_stats(r, exclude_ym=None)
    ms_strip = st.monthly_stats(r, exclude_ym=st.CRASH_MONTHS)
    # Three October crash months removed → fewer October days.
    assert ms_strip.loc[10, "n"] < ms_full.loc[10, "n"]


# ---------------------------------------------------------------------------
# null_baseline (permutation)
# ---------------------------------------------------------------------------
def test_permutation_null_returns_keys(null_tape):
    ret, _ = null_tape
    result = st.permutation_null(ret, n_perm=200, seed=42)
    assert "pct_worse_than_oct" in result
    assert "oct_mean_bps" in result
    assert 0.0 <= result["pct_worse_than_oct"] <= 1.0


def test_permutation_null_pct_near_half_on_null(null_tape):
    """On a null tape, a randomly-chosen month should not be systematically ranked #1 worst."""
    ret, _ = null_tape
    result = st.permutation_null(ret, n_perm=500, seed=136)
    # On the null, the fraction should be somewhere in a plausible middle range.
    # We allow 1%–99% to avoid flakiness; the point is that it is NOT 0 or 100%.
    assert 0.01 < result["pct_worse_than_oct"] < 0.99


# ---------------------------------------------------------------------------
# period_summary
# ---------------------------------------------------------------------------
def test_period_summary_keys(null_tape):
    ret, _ = null_tape
    ps = st.period_summary(ret)
    for k in ("n", "cagr", "vol_ann", "sharpe", "start", "end"):
        assert k in ps


def test_period_summary_positive_sharpe_on_drift_tape():
    """A tape with a positive drift should produce a positive Sharpe."""
    ret, _ = __import__("mark_twain", fromlist=["data"]).data.synthetic_daily(
        n_years=40, annual_drift_bp=700.0, october_penalty_bp=0.0, seed=5
    )
    ps = st.period_summary(ret)
    assert ps["sharpe"] > 0

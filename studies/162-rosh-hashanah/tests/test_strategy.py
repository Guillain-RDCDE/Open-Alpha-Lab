"""Strategy invariants, the HAC t, and the study spine: drag is detectable when planted."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rosh_hashanah import data, strategy as st


# ---------------------------------------------------------------------------
# _hac_tstat
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_mean_near_zero():
    """A zero-mean signal should give a t-stat near zero."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    t = st._hac_tstat(x)
    assert abs(t) < 4.0  # within noise even for a single draw


def test_hac_tstat_large_signal_large_t():
    """A strongly non-zero signal with tiny but nonzero variance should give large |t|."""
    rng = np.random.default_rng(0)
    x = 0.01 + rng.standard_normal(80) * 0.0001  # 1% mean, near-zero vol
    t = st._hac_tstat(x)
    assert t > 10.0


def test_hac_tstat_nan_on_tiny_sample():
    assert np.isnan(st._hac_tstat(np.array([])))
    assert np.isnan(st._hac_tstat(np.array([1.0])))


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(null_world):
    df, _ = null_world
    out = st.summarize(df, col="window_ret")
    for k in ("n", "mean_bps", "win_rate", "tstat", "pct_negative", "median_bps"):
        assert k in out


def test_summarize_n_matches_data(null_world):
    df, _ = null_world
    out = st.summarize(df, col="window_ret")
    assert out["n"] == len(df)


def test_summarize_win_rate_and_pct_negative_sum_to_lte_one(null_world):
    df, _ = null_world
    out = st.summarize(df, col="window_ret")
    assert out["win_rate"] + out["pct_negative"] <= 1.0 + 1e-9


def test_summarize_null_tstat_small(null_world):
    """On the null synthetic tape the HAC t should not be extreme."""
    df, _ = null_world
    out = st.summarize(df, col="window_ret")
    # With 80 observations at vol ~1% the t should be well within ±4 under the null
    assert abs(out["tstat"]) < 5.0


# ---------------------------------------------------------------------------
# The spine: drag is detectable only when planted
# ---------------------------------------------------------------------------
def test_drag_detected_by_summarize(holiday_drag_world, null_world):
    """With a large planted drag the mean_bps should be clearly negative."""
    drag_df, _ = holiday_drag_world
    null_df, _ = null_world
    drag_stat = st.summarize(drag_df, col="window_ret")
    null_stat = st.summarize(null_df, col="window_ret")
    assert drag_stat["mean_bps"] < null_stat["mean_bps"]
    assert drag_stat["mean_bps"] < -50.0  # planted -100 bp drag shows up


# ---------------------------------------------------------------------------
# compare_holiday_vs_random
# ---------------------------------------------------------------------------
def test_compare_holiday_vs_random_keys():
    """The comparison function returns the expected keys."""
    hw = pd.DataFrame({"window_ret": [0.01, -0.02, 0.005]}, index=[2020, 2021, 2022])
    rw = pd.DataFrame({"rand_median": [0.02, 0.01, 0.015]}, index=[2020, 2021, 2022])
    out = st.compare_holiday_vs_random(hw, rw)
    for k in ("n", "mean_gap_bps", "tstat_gap", "pct_holiday_worse"):
        assert k in out


def test_compare_holiday_vs_random_gap_direction():
    """When holiday returns are worse, mean_gap_bps should be negative."""
    hw = pd.DataFrame({"window_ret": [-0.01, -0.02, -0.01]}, index=[2020, 2021, 2022])
    rw = pd.DataFrame({"rand_median": [0.01, 0.01, 0.01]}, index=[2020, 2021, 2022])
    out = st.compare_holiday_vs_random(hw, rw)
    assert out["mean_gap_bps"] < 0
    assert out["pct_holiday_worse"] == 1.0


# ---------------------------------------------------------------------------
# timing_pnl
# ---------------------------------------------------------------------------
def _make_hw():
    """Helper: toy holiday-window DataFrame for timing_pnl tests."""
    return pd.DataFrame(
        {
            "window_ret": [0.01, -0.02],
            "entry_date": [pd.Timestamp("2020-09-17"), pd.Timestamp("2021-09-06")],
            "exit_date": [pd.Timestamp("2020-09-28"), pd.Timestamp("2021-09-16")],
            "rosh_hashanah": [pd.Timestamp("2020-09-19"), pd.Timestamp("2021-09-07")],
            "yom_kippur": [pd.Timestamp("2020-09-28"), pd.Timestamp("2021-09-16")],
            "n_trading": [7, 7],
            "entry_close": [3300.0, 4500.0],
            "exit_close": [3333.0, 4410.0],
        },
        index=[2020, 2021],
    )


def _dummy_prices():
    """Minimal prices df to satisfy typing_pnl signature."""
    return pd.DataFrame({"close": [1.0]}, index=[pd.Timestamp("2020-01-01")])


def test_timing_pnl_columns():
    """timing_pnl returns the expected columns."""
    hw = _make_hw()
    pnl = st.timing_pnl(hw, _dummy_prices())
    for col in ("bah_ret", "signal_ret", "cost", "signal_net"):
        assert col in pnl.columns


def test_timing_pnl_signal_is_negative_of_window():
    """Short the window: signal_ret = -window_ret, signal_net = -window_ret - cost."""
    hw = _make_hw()
    pnl = st.timing_pnl(hw, _dummy_prices())
    np.testing.assert_allclose(pnl["signal_ret"].values, -hw["window_ret"].values)
    cost_bps = 5.0
    expected_net = -hw["window_ret"].values - 2 * cost_bps * 1e-4
    np.testing.assert_allclose(pnl["signal_net"].values, expected_net)

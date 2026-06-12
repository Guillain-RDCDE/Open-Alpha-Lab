"""Window detection invariants, the HAC engine, and the study's spine:
the detector finds the Santa drift only when one is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sleigh_ride import strategy as st, data  # noqa: E402


# ---- window labeling ------------------------------------------------------
def test_label_santa_windows_has_correct_count(null_tape):
    bars, truth = null_tape
    mask = st.label_santa_windows(bars)
    # Each full year contributes 7 bars (5 tail + 2 head of next year).
    # Allow ±14 for boundary years.
    expected = truth["n_years"] * 7
    assert abs(mask.sum() - expected) <= 14


def test_label_santa_windows_correct_dates():
    """Check that the last-of-year dates are stamped as Santa, not mid-year dates."""
    bars, _ = data.synthetic_daily(n_years=5, seed=79)
    mask = st.label_santa_windows(bars)
    # No July dates should be in the Santa window.
    july_mask = mask[mask.index.month == 7]
    assert july_mask.sum() == 0
    # December/January dates should have Santa bars.
    dec_mask = mask[mask.index.month == 12]
    jan_mask = mask[mask.index.month == 1]
    assert dec_mask.sum() > 0
    assert jan_mask.sum() > 0


def test_santa_windows_returns_correct_columns(null_tape):
    bars, _ = null_tape
    wins = st.santa_windows(bars)
    expected_cols = {"start", "end", "n_days", "total_ret", "mean_daily_ret", "positive"}
    assert expected_cols.issubset(set(wins.columns))


def test_santa_windows_n_days_is_seven(null_tape):
    bars, _ = null_tape
    wins = st.santa_windows(bars)
    # Most windows should be exactly 7 days (5+2), except boundary years.
    assert (wins["n_days"] == 7).mean() > 0.8


# ---- core analysis --------------------------------------------------------
def test_compare_santa_returns_required_keys(null_tape):
    bars, _ = null_tape
    res = st.compare_santa_vs_baseline(bars)
    for key in ("santa_n", "santa_mean_bps", "santa_tstat", "base_n", "diff_bps", "diff_tstat"):
        assert key in res, f"Missing key: {key}"


def test_planted_drift_shows_higher_santa_mean(planted_tape, null_tape):
    """The window detector finds a planted Santa drift and not a null tape."""
    bars_planted, _ = planted_tape
    bars_null, _ = null_tape
    res_planted = st.compare_santa_vs_baseline(bars_planted)
    res_null = st.compare_santa_vs_baseline(bars_null)
    assert res_planted["santa_mean_bps"] > res_null["santa_mean_bps"] + 5.0


def test_planted_drift_has_higher_tstat(planted_tape, null_tape):
    """With a planted drift the diff_tstat should be clearly higher than on a null tape."""
    bars_planted, _ = planted_tape
    bars_null, _ = null_tape
    t_planted = st.compare_santa_vs_baseline(bars_planted)["diff_tstat"]
    t_null = st.compare_santa_vs_baseline(bars_null)["diff_tstat"]
    # The planted tape should have a substantially larger t-stat.
    assert t_planted > t_null + 1.0


# ---- santa_window_summary -------------------------------------------------
def test_window_summary_keys(null_tape):
    bars, _ = null_tape
    ws = st.santa_window_summary(bars)
    for key in ("n_windows", "mean_window_bps", "win_rate", "tstat_vs_zero"):
        assert key in ws


def test_window_summary_win_rate_in_range(null_tape):
    bars, _ = null_tape
    ws = st.santa_window_summary(bars)
    assert 0.0 <= ws["win_rate"] <= 1.0


# ---- HAC t-stat -----------------------------------------------------------
def test_hac_tstat_known_mean():
    """A series with a known large mean should have a large t-stat."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.001, 100)  # mean 1% well above zero
    t = st._hac_tstat(r)
    assert t > 5.0


def test_hac_tstat_zero_mean():
    """A zero-mean series should have a small t-stat."""
    rng = np.random.default_rng(42)
    r = rng.normal(0.0, 0.01, 200)
    t = st._hac_tstat(r)
    assert abs(t) < 2.5  # not always zero, but should be small


def test_hac_tstat_few_obs_returns_nan():
    r = np.array([0.01, 0.02, 0.01])
    assert np.isnan(st._hac_tstat(r))


# ---- summarize ------------------------------------------------------------
def test_summarize_keys():
    r = np.random.default_rng(1).normal(0.001, 0.01, 50)
    s = st.summarize(r, label="test")
    for key in ("n", "mean_bps", "median_bps", "win_rate", "tstat"):
        assert key in s


def test_summarize_win_rate_in_range():
    r = np.array([0.01, -0.01, 0.02, -0.005, 0.015])
    s = st.summarize(r)
    assert 0.0 <= s["win_rate"] <= 1.0


# ---- positive control: the spine -----------------------------------------
def test_detector_is_a_faithful_drift_finder():
    """The spine: with planted drift the mean is clearly elevated;
    on a martingale it is not (the window is just a random 7 days)."""
    planted, _ = data.synthetic_daily(n_years=75, santa_bps=60.0, seed=79)
    null, _ = data.synthetic_daily(n_years=75, santa_bps=0.0, seed=79)
    ws_planted = st.santa_window_summary(planted)
    ws_null = st.santa_window_summary(null)
    assert ws_planted["mean_window_bps"] > ws_null["mean_window_bps"] + 20.0
    # Planted t-stat should be meaningfully larger.
    assert ws_planted["tstat_vs_zero"] > ws_null["tstat_vs_zero"] + 1.0

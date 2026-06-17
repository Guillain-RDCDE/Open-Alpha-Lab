"""Tests for the strategy layer of Study 280 (Solar-Eclipse).

All tests are offline and deterministic — they use the synthetic generator and
the hardcoded eclipse table only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solar_eclipse import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Event alignment
# ---------------------------------------------------------------------------
def test_event_day_returns_shape():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    ev = st.event_day_returns(df["ret"], ecl)
    assert set(["date", "event_date", "kind", "us_path", "ret"]).issubset(ev.columns)
    # Most eclipses should land inside the synthetic window.
    assert len(ev) >= len(ecl) - 2


def test_event_date_is_on_or_after_eclipse():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    ev = st.event_day_returns(df["ret"], ecl, lag=0)
    assert (ev["event_date"].values >= ev["date"].values).all()


def test_event_window_matrix_shape():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    win = 4
    mat = st.event_window_matrix(df["ret"], ecl, win=win)
    assert list(mat.columns) == list(range(-win, win + 1))
    assert len(mat) >= len(ecl) - 2


# ---------------------------------------------------------------------------
# event_stats: structure and invariants
# ---------------------------------------------------------------------------
def test_event_stats_keys():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    r = st.event_stats(df["ret"], ecl, n_permutations=200, seed=99)
    required = {
        "n_events", "uncond_bps", "mean_event_bps", "mean_abn_bps",
        "se_abn_bps", "win_rate", "t_hac", "t_ord", "p_ord", "perm_p",
    }
    assert required.issubset(set(r.keys()))


def test_event_stats_perm_p_is_probability():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    r = st.event_stats(df["ret"], ecl, n_permutations=500, seed=7)
    assert 0.0 <= r["perm_p"] <= 1.0


def test_event_stats_win_rate_in_range():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    r = st.event_stats(df["ret"], ecl, n_permutations=200, seed=1)
    assert 0.0 <= r["win_rate"] <= 1.0


def test_permutation_deterministic():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    r1 = st.event_stats(df["ret"], ecl, n_permutations=300, seed=42)
    r2 = st.event_stats(df["ret"], ecl, n_permutations=300, seed=42)
    assert r1["perm_p"] == r2["perm_p"]


# ---------------------------------------------------------------------------
# Null vs planted signal: the spine of the study
# ---------------------------------------------------------------------------
def test_null_has_no_signal():
    """On the null tape, the eclipse-day abnormal return is tiny and t is small."""
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    r = st.event_stats(df["ret"], ecl, n_permutations=1000, seed=280)
    assert abs(r["mean_abn_bps"]) < 30.0
    assert abs(r["t_hac"]) < 2.0


def test_planted_drag_detectable():
    """A large planted eclipse drag is detected: negative mean, big |t|, tiny p."""
    df, _ = data.synthetic_panel(signal_bps=-300.0, seed=280)
    ecl = data.eclipse_table()
    r = st.event_stats(df["ret"], ecl, n_permutations=1000, seed=280)
    assert r["mean_abn_bps"] < -150.0
    assert r["t_hac"] < -2.0
    assert r["perm_p"] < 0.05


def test_planted_premium_detectable():
    """A large planted positive premium flips the sign and is detected."""
    df, _ = data.synthetic_panel(signal_bps=250.0, seed=280)
    ecl = data.eclipse_table()
    r = st.event_stats(df["ret"], ecl, n_permutations=1000, seed=280)
    assert r["mean_abn_bps"] > 100.0
    assert r["t_hac"] > 2.0


# ---------------------------------------------------------------------------
# CAR profile
# ---------------------------------------------------------------------------
def test_car_profile_shape():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    win = 5
    car = st.car_profile(df["ret"], ecl, win=win)
    assert list(car.index) == list(range(-win, win + 1))
    assert {"aar_bps", "car_bps"}.issubset(car.columns)


def test_car_is_cumsum_of_aar():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    car = st.car_profile(df["ret"], ecl, win=3)
    assert np.allclose(car["car_bps"].to_numpy(), car["aar_bps"].cumsum().to_numpy())


# ---------------------------------------------------------------------------
# multiple_comparisons_summary
# ---------------------------------------------------------------------------
def test_mc_summary_shape():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    mc = st.multiple_comparisons_summary(df["ret"], ecl, n_permutations=200, seed=1)
    assert "bonferroni_perm_p" in mc.columns
    assert len(mc) >= 3


def test_mc_bonferroni_ge_uncorrected():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    mc = st.multiple_comparisons_summary(df["ret"], ecl, n_permutations=200, seed=1)
    for _, row in mc.iterrows():
        assert row["bonferroni_perm_p"] >= row["perm_p"] - 1e-12


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df, _ = data.synthetic_panel(signal_bps=0.0, seed=280)
    ecl = data.eclipse_table()
    s = st.summarize(df["ret"], ecl, n_permutations=200, seed=1)
    required = {
        "n_all", "n_us", "n_total", "uncond_bps",
        "all_mean_abn_bps", "all_t_hac", "all_perm_p", "all_win_rate_pct",
        "us_mean_abn_bps", "us_t_hac", "car_window_bps",
    }
    assert required.issubset(set(s.keys()))


def test_nw_tstat_matches_ordinary_at_zero_lag():
    """With lags=0 the HAC t equals the ordinary t-stat up to the n/(n-1) SD factor.

    Our HAC variance divides by n (the population convention); scipy's t uses the
    sample SD (n-1). The two t-stats differ only by sqrt((n-1)/n).
    """
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(0.5, 1.0, n)
    t_hac, _ = st._nw_tstat(x, lags=0)
    from scipy import stats as scipy_stats
    t_ord, _ = scipy_stats.ttest_1samp(x, 0.0)
    assert abs(t_hac - t_ord * np.sqrt(n / (n - 1))) < 1e-9

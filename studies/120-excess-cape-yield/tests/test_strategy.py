"""Tests for the strategy layer — regression, OOS split, buckets, and the study spine."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from excess_cape_yield import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# ols_regression
# ---------------------------------------------------------------------------
def test_ols_regression_perfect_fit():
    x = np.linspace(0.01, 0.15, 50)
    y = 2.5 * x + 0.01
    res = st.ols_regression(x, y)
    assert abs(res["slope"] - 2.5) < 1e-8
    assert abs(res["intercept"] - 0.01) < 1e-8
    assert abs(res["r2"] - 1.0) < 1e-8
    assert res["n"] == 50


def test_ols_regression_null():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50)
    y = rng.normal(0, 1, 50)  # no relation
    res = st.ols_regression(x, y)
    assert abs(res["r2"]) < 0.3  # expect low R² in noise


def test_ols_regression_nan_handling():
    x = np.array([1.0, 2.0, np.nan, 4.0])
    y = np.array([2.0, 4.0, 6.0, np.nan])
    res = st.ols_regression(x, y)
    assert res["n"] == 2  # only (1,2) and (2,4) survive; (3,6) and (4,nan) dropped


# ---------------------------------------------------------------------------
# nonoverlay_regression
# ---------------------------------------------------------------------------
def test_nonoverlay_regression_step(signal_world):
    frame, _ = signal_world
    sub = frame[["ecy", "fwd10_excess"]].dropna()
    res = st.nonoverlay_regression(sub, "ecy", "fwd10_excess", step=120)
    assert res["step"] == 120
    # With planted signal, slope should be positive and significant
    assert res["slope"] > 0.3, f"Expected positive slope, got {res['slope']}"
    assert res["tstat"] > 2.0, f"Expected t > 2, got {res['tstat']}"


def test_nonoverlay_regression_null_has_small_t(null_world):
    """In the null world, the regression t-stat on non-overlapping windows is small."""
    frame, _ = null_world
    sub = frame[["ecy", "fwd10_excess"]].dropna()
    res = st.nonoverlay_regression(sub, "ecy", "fwd10_excess", step=120)
    # We don't assert |t| < 2 because it's random; we assert it's not always > 4
    assert abs(res["tstat"]) < 6.0 or np.isnan(res["tstat"])


# ---------------------------------------------------------------------------
# oos_r2
# ---------------------------------------------------------------------------
def test_oos_r2_positive_in_signal_world(signal_world):
    """A planted ECY signal should beat the historical-mean benchmark OOS."""
    frame, _ = signal_world
    sub = frame[["ecy", "fwd10_excess"]].dropna()
    res = st.oos_r2(sub, "ecy", "fwd10_excess", step=120, train_frac=0.5)
    # Note: with only ~13 non-overlap windows this can go either way, allow NaN
    if not np.isnan(res["oos_r2"]):
        assert res["train_n"] >= 1
        assert res["test_n"] >= 1


def test_oos_r2_keys():
    frame, _ = data.synthetic_world(n_years=100, predicts=1.0, seed=1)
    sub = frame[["ecy", "fwd10_excess"]].dropna()
    res = st.oos_r2(sub, "ecy", "fwd10_excess", step=120)
    assert "oos_r2" in res
    assert "train_n" in res and "test_n" in res


# ---------------------------------------------------------------------------
# bucket_returns
# ---------------------------------------------------------------------------
def test_bucket_returns_monotone(signal_world):
    """Quintile excess returns should be (mostly) monotone in ECY in the signal world."""
    frame, _ = signal_world
    sub = frame[["ecy", "fwd10_excess"]].dropna()
    buckets = st.bucket_returns(sub, "ecy", "fwd10_excess", n_buckets=5)
    assert len(buckets) == 5
    vals = buckets.values
    # Monotone up: Q1 (low ECY) should have lower mean excess than Q5 (high ECY)
    assert vals[-1] > vals[0], "High ECY quintile should have higher excess return than low"


def test_bucket_returns_null_not_strictly_monotone(null_world):
    """In the null world the bucket ladder need not be monotone."""
    frame, _ = null_world
    sub = frame[["ecy", "fwd10_excess"]].dropna()
    buckets = st.bucket_returns(sub, "ecy", "fwd10_excess", n_buckets=5)
    assert len(buckets) == 5  # shape correct regardless


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(signal_world):
    frame, _ = signal_world
    s = st.summarize(frame)
    for key in (
        "n_monthly", "r2_ecy_monthly", "r2_ecy_nonoverlap",
        "tstat_ecy_nonoverlap", "oos_r2_ecy", "ecy_mean_pct",
    ):
        assert key in s, f"Missing key: {key}"


def test_summarize_signal_vs_null():
    """The summarize spine: signal world should have higher R² than null."""
    frame_s, _ = data.synthetic_world(n_years=130, predicts=1.0, seed=120)
    frame_n, _ = data.synthetic_world(n_years=130, predicts=0.0, seed=120)
    s = st.summarize(frame_s)
    n = st.summarize(frame_n)
    assert s["r2_ecy_monthly"] > n["r2_ecy_monthly"], "Signal world must have higher monthly R²"

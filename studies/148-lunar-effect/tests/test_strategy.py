"""Strategy: the window-contrast and shuffled-label control, and the study's spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunar_effect import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# summarize() — basic shape and invariants
# ---------------------------------------------------------------------------
def test_summarize_keys(null_frame):
    frame, _ = null_frame
    s = st.summarize(frame)
    expected = {
        "n_new", "n_full", "n_other", "n_total",
        "mean_new_bps", "mean_full_bps", "mean_other_bps", "mean_all_bps",
        "contrast_bps", "t_new", "t_full", "t_contrast",
    }
    assert expected <= set(s.keys())


def test_counts_sum_to_total(null_frame):
    frame, _ = null_frame
    s = st.summarize(frame)
    assert s["n_new"] + s["n_full"] + s["n_other"] == s["n_total"]


def test_contrast_equals_new_minus_full(null_frame):
    frame, _ = null_frame
    s = st.summarize(frame)
    assert abs(s["contrast_bps"] - (s["mean_new_bps"] - s["mean_full_bps"])) < 1e-9


# ---------------------------------------------------------------------------
# The spine: planted signal is detected; null is not
# ---------------------------------------------------------------------------
def test_contrast_detects_planted_effect(signal_frame, null_frame):
    """On the signal tape the contrast exceeds the null tape's contrast.

    The signal tape has full_discount=5e-4 and new_premium=5e-4, so the planted
    contrast is ~10 bps/day.  The null tape has no planted effect.  The signal
    tape's contrast should be materially higher than the null tape's contrast —
    we don't require a specific threshold because random variation on 3000 days
    can be a few bps in either direction.
    """
    sig_frame, _ = signal_frame
    null, _ = null_frame
    s_sig = st.summarize(sig_frame)
    s_null = st.summarize(null)
    assert s_sig["contrast_bps"] > s_null["contrast_bps"] + 5.0


def test_null_contrast_within_noise(null_frame):
    """With no planted effect, the contrast t-stat should be well below 2.

    We test the t-stat rather than the raw bps value because the t-stat
    accounts for the sample noise properly.  On 3000 null-drawn days the
    contrast t should not exceed ±3 (it would happen <1% of the time if
    the seed is unlucky, but seed=148 is a fixed value).
    """
    frame, _ = null_frame
    s = st.summarize(frame)
    assert abs(s["t_contrast"]) < 3.0


# ---------------------------------------------------------------------------
# Shuffled-label control
# ---------------------------------------------------------------------------
def test_shuffled_contrast_distribution(null_frame):
    """Shuffled contrasts should be centred near zero."""
    frame, _ = null_frame
    perms = st.shuffled_contrast(frame, n_perms=200, seed=42)
    assert perms.size == 200
    # Mean of the null distribution near 0
    assert abs(np.nanmean(perms)) < 2.0


def test_shuffled_reproducible(null_frame):
    frame, _ = null_frame
    a = st.shuffled_contrast(frame, n_perms=50, seed=1)
    b = st.shuffled_contrast(frame, n_perms=50, seed=1)
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# by_decade — structure
# ---------------------------------------------------------------------------
def test_by_decade_columns(null_frame):
    frame, _ = null_frame
    dec = st.by_decade(frame)
    assert list(dec.columns) == ["n_new", "n_full", "contrast_bps", "t_contrast"]
    assert len(dec) >= 1


# ---------------------------------------------------------------------------
# lunar_strategy_returns
# ---------------------------------------------------------------------------
def test_strategy_returns_sign(null_frame):
    """On a NEW day the strategy return has the same sign as the underlying return."""
    frame, _ = null_frame
    strat = st.lunar_strategy_returns(frame)
    new_mask = frame["lunar_label"] == "NEW"
    full_mask = frame["lunar_label"] == "FULL"
    # NEW days: strategy_ret = +1 * ret, so sign should match
    assert np.sign(strat[new_mask]).equals(np.sign(frame.loc[new_mask, "ret"]))
    # FULL days: strategy_ret = -1 * ret, so sign should be opposite
    assert np.sign(strat[full_mask]).equals(-np.sign(frame.loc[full_mask, "ret"]))


def test_strategy_other_days_flat(null_frame):
    frame, _ = null_frame
    strat = st.lunar_strategy_returns(frame)
    other_mask = frame["lunar_label"] == "OTHER"
    assert (strat[other_mask] == 0.0).all()

"""Tests for the strategy layer — even/odd split, placebo, era split, the study spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fomc_cycle import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# even_odd_table
# ---------------------------------------------------------------------------
def test_even_odd_table_keys(null_world):
    frame, _ = null_world
    tbl = st.even_odd_table(frame)
    for arm in ("even", "odd", "all"):
        assert arm in tbl
        assert "n" in tbl[arm]
        assert "mean_bps" in tbl[arm]
        assert "tstat" in tbl[arm]
    assert "diff_bps" in tbl
    assert "welch_t" in tbl


def test_even_odd_counts_are_consistent(null_world):
    frame, truth = null_world
    tbl = st.even_odd_table(frame)
    assert tbl["even"]["n"] + tbl["odd"]["n"] == tbl["all"]["n"]
    assert tbl["even"]["n"] == truth["n_even"]
    assert tbl["odd"]["n"] == truth["n_odd"]


def test_even_odd_gap_is_planted_gap(signal_world, null_world):
    """The table correctly measures a planted even-week premium."""
    sig_tbl = st.even_odd_table(signal_world[0])
    null_tbl = st.even_odd_table(null_world[0])
    # Signal world should have a higher even mean than the null world
    assert sig_tbl["even"]["mean_bps"] > null_tbl["even"]["mean_bps"] + 5.0


def test_welch_t_is_float(null_world):
    frame, _ = null_world
    tbl = st.even_odd_table(frame)
    assert np.isfinite(tbl["welch_t"])


# ---------------------------------------------------------------------------
# week_by_week
# ---------------------------------------------------------------------------
def test_week_by_week_shape(null_world):
    frame, _ = null_world
    wbw = st.week_by_week(frame)
    assert list(wbw.index) == [0, 1, 2, 3, 4, 5]
    assert "mean_bps" in wbw.columns
    assert "tstat" in wbw.columns
    assert "is_even" in wbw.columns


def test_week_by_week_even_flag(null_world):
    frame, _ = null_world
    wbw = st.week_by_week(frame)
    assert bool(wbw.loc[0, "is_even"]) is True
    assert bool(wbw.loc[1, "is_even"]) is False
    assert bool(wbw.loc[2, "is_even"]) is True
    assert bool(wbw.loc[4, "is_even"]) is True
    assert bool(wbw.loc[3, "is_even"]) is False


def test_planted_premium_shows_in_even_weeks(signal_world):
    """Even weeks in the signal world should have higher mean than odd weeks."""
    frame, _ = signal_world
    wbw = st.week_by_week(frame)
    even_mean = wbw.loc[wbw["is_even"], "mean_bps"].mean()
    odd_mean = wbw.loc[~wbw["is_even"], "mean_bps"].mean()
    assert even_mean > odd_mean


# ---------------------------------------------------------------------------
# random_week_split (placebo)
# ---------------------------------------------------------------------------
def test_placebo_keys(null_world):
    frame, _ = null_world
    plac = st.random_week_split(frame, n_seeds=20)
    for k in ("real_gap_bps", "placebo_mean_bps", "placebo_std_bps", "p_value", "n_seeds"):
        assert k in plac


def test_placebo_p_value_in_range(null_world):
    frame, _ = null_world
    plac = st.random_week_split(frame, n_seeds=50)
    assert 0.0 <= plac["p_value"] <= 1.0


def test_placebo_centred_near_zero(null_world):
    """The shuffle null distribution should be centred near 0."""
    frame, _ = null_world
    plac = st.random_week_split(frame, n_seeds=100)
    assert abs(plac["placebo_mean_bps"]) < 2.0


# ---------------------------------------------------------------------------
# split_by_era
# ---------------------------------------------------------------------------
def test_split_by_era_keys(null_world):
    frame, _ = null_world
    era = st.split_by_era(frame)
    assert "pre_publication" in era
    assert "post_publication" in era
    for sub in ("pre_publication", "post_publication"):
        assert "diff_bps" in era[sub]
        assert "welch_t" in era[sub]
        assert "n_even" in era[sub]


def test_split_by_era_cuts_the_sample(signal_world):
    frame, _ = signal_world
    # Use a cut within the synthetic date range (starts 1994-02-01, n_days=2000 ~= 8 years)
    era = st.split_by_era(frame, cut="1998-01-01")
    pre = era["pre_publication"]
    post = era["post_publication"]
    assert pre["n_even"] > 0
    assert post["n_even"] > 0
    assert pre["n_even"] + post["n_even"] <= st.even_odd_table(frame)["even"]["n"]


# ---------------------------------------------------------------------------
# cost_sweep
# ---------------------------------------------------------------------------
def test_cost_sweep_shape(null_world):
    frame, _ = null_world
    cs = st.cost_sweep(frame)
    assert "net_ann_pct" in cs.columns
    assert "gross_ann_pct" in cs.columns
    assert len(cs) == 5  # default 5 cost levels


def test_cost_sweep_net_decreases_with_cost(null_world):
    frame, _ = null_world
    cs = st.cost_sweep(frame)
    net = cs["net_ann_pct"].to_numpy()
    assert (np.diff(net) <= 0).all()


# ---------------------------------------------------------------------------
# The spine — signal only appears with planted premium
# ---------------------------------------------------------------------------
def test_spine_null_world_no_significant_gap(null_world):
    """On the null tape the even-odd Welch-t should not be highly significant."""
    frame, _ = null_world
    tbl = st.even_odd_table(frame)
    # With n~2000, random chance t should be well below 3 in absolute value
    assert abs(tbl["welch_t"]) < 3.5


def test_spine_signal_world_detects_premium(signal_world):
    """With a planted 0.1%/day premium on even weeks, the machinery should flag it."""
    frame, _ = signal_world
    tbl = st.even_odd_table(frame)
    # A 10 bps/day lift over ~800 even days should yield a clear positive diff
    assert tbl["diff_bps"] > 3.0

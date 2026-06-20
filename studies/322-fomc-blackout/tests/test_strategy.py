"""Tests for the strategy layer — blackout table, placebo, era split, vol, costs."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fomc_blackout import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# blackout_table
# ---------------------------------------------------------------------------
def test_blackout_table_keys(null_world):
    frame, _ = null_world
    tbl = st.blackout_table(frame)
    for arm in ("blackout", "other", "all"):
        assert arm in tbl
        for k in ("n", "mean_bps", "tstat", "sharpe_ann"):
            assert k in tbl[arm]
    assert "diff_bps" in tbl
    assert "welch_t" in tbl


def test_blackout_counts_consistent(null_world):
    frame, truth = null_world
    tbl = st.blackout_table(frame)
    assert tbl["blackout"]["n"] + tbl["other"]["n"] == tbl["all"]["n"]
    assert tbl["blackout"]["n"] == truth["n_blackout"]


def test_null_diff_is_insignificant(null_world):
    """On the null tape the blackout-minus-other Welch t is near zero."""
    frame, _ = null_world
    tbl = st.blackout_table(frame)
    assert abs(tbl["welch_t"]) < 3.0


def test_signal_diff_detected(signal_world):
    """A planted +15 bps/day blackout excess yields a clear positive, significant diff."""
    frame, _ = signal_world
    tbl = st.blackout_table(frame)
    assert tbl["diff_bps"] > 5.0
    assert tbl["welch_t"] > 2.0


# ---------------------------------------------------------------------------
# placebo_shift
# ---------------------------------------------------------------------------
def test_placebo_shift_shape(null_world):
    frame, _ = null_world
    pl = st.placebo_shift(frame, data.FOMC_DATES, offsets=[-10, 0, 10])
    assert list(pl.index) == [-10, 0, 10]
    assert "diff_bps" in pl.columns
    assert "welch_t" in pl.columns


def test_placebo_peaks_at_zero_for_signal(signal_world):
    """With a real planted effect, the diff at offset 0 dominates the shifted windows."""
    frame, _ = signal_world
    pl = st.placebo_shift(frame, data.FOMC_DATES, offsets=[-20, 0, 20])
    assert pl.loc[0, "diff_bps"] > pl.loc[-20, "diff_bps"]
    assert pl.loc[0, "diff_bps"] > pl.loc[20, "diff_bps"]


# ---------------------------------------------------------------------------
# split_by_era
# ---------------------------------------------------------------------------
def test_split_by_era_keys(null_world):
    frame, _ = null_world
    era = st.split_by_era(frame, cut="2000-01-01")
    for sub in ("pre", "post"):
        assert "diff_bps" in era[sub]
        assert "welch_t" in era[sub]
        assert "n_blackout" in era[sub]


def test_split_by_era_partitions(signal_world):
    frame, _ = signal_world
    era = st.split_by_era(frame, cut="2000-01-01")
    total = st.blackout_table(frame)["blackout"]["n"]
    assert era["pre"]["n_blackout"] + era["post"]["n_blackout"] <= total
    assert era["pre"]["n_blackout"] > 0
    assert era["post"]["n_blackout"] > 0


# ---------------------------------------------------------------------------
# vol_table — the "calm?" check
# ---------------------------------------------------------------------------
def test_vol_table_keys(null_world):
    frame, _ = null_world
    vt = st.vol_table(frame)
    for k in ("blackout_vol_bps", "other_vol_bps", "ratio"):
        assert k in vt


def test_vol_table_calm_detected(calm_world):
    """Planted half-vol in the blackout window gives ratio well below 1."""
    frame, _ = calm_world
    vt = st.vol_table(frame)
    assert vt["ratio"] < 0.8


def test_vol_table_null_near_one(null_world):
    frame, _ = null_world
    vt = st.vol_table(frame)
    assert 0.85 < vt["ratio"] < 1.15


# ---------------------------------------------------------------------------
# cost_sweep & excess_sharpe
# ---------------------------------------------------------------------------
def test_cost_sweep_shape(null_world):
    frame, _ = null_world
    cs = st.cost_sweep(frame)
    for c in ("book_gross_ann_pct", "book_net_ann_pct", "buyhold_ann_pct", "trips_per_year"):
        assert c in cs.columns
    assert len(cs) == 4


def test_cost_sweep_net_decreases_with_cost(null_world):
    frame, _ = null_world
    cs = st.cost_sweep(frame)
    net = cs["book_net_ann_pct"].to_numpy()
    assert (np.diff(net) <= 0).all()


def test_excess_sharpe_keys(null_world):
    frame, _ = null_world
    es = st.excess_sharpe(frame)
    assert "book_excess_sharpe" in es
    assert "buyhold_excess_sharpe" in es


# ---------------------------------------------------------------------------
# HAC t-stat sanity
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000) * 0.01
    assert abs(st.hac_tstat(x)) < 3.0


def test_hac_tstat_detects_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.standard_normal(5000) * 0.01
    assert st.hac_tstat(x) > 3.0

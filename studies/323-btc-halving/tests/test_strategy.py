"""The cycle engine's invariants and the study's two spines: (1) the phase-bucket engine
detects a planted halving cycle and finds nothing in the null, and (2) the timing rule
respects the one-bar fill lag and charges one-way costs honestly."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_halving import data, strategy as st  # noqa: E402


# ---- inference helpers ----------------------------------------------------
def test_hac_tstat_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 5000)
    assert abs(st.hac_tstat(x)) < 3.0


def test_hac_tstat_nan_on_tiny_sample():
    assert np.isnan(st.hac_tstat(np.array([1.0, 2.0, 3.0])))


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.02, 3000)
    lo, hi = st.block_bootstrap_ci(x, block=21, n_boot=500)
    assert lo < x.mean() * 1e4 < hi


# ---- cycle extrema --------------------------------------------------------
def test_cycle_extrema_columns_and_offsets(cycle_tape):
    bars, _ = cycle_tape
    ext = st.cycle_extrema(bars)
    assert list(ext.columns) == [
        "halving", "n_days_in_window", "high_offset_d", "low_offset_d",
        "high_ret_x", "low_ret_x",
    ]
    assert (ext["high_offset_d"] >= 0).all()
    assert (ext["low_offset_d"] >= 0).all()
    # offsets must lie inside the cycle window
    assert (ext["high_offset_d"] <= data.CYCLE_DAYS).all()


# ---- phase-bucket spine ---------------------------------------------------
def test_phase_buckets_partition_the_days(cycle_tape):
    bars, _ = cycle_tape
    pb = st.phase_bucket_returns(bars, n_buckets=4)
    assert len(pb) == 4
    # total bucketed days <= available daily returns (NaN-phase days excluded)
    logret = np.log(bars["close"]).diff()
    phase = data.cycle_phase(bars.index)
    valid = np.isfinite(logret.to_numpy()) & np.isfinite(phase)
    assert pb["n"].sum() == int(valid.sum())


def test_engine_detects_planted_cycle_and_not_the_null(cycle_tape, null_tape):
    """Spine #1: the post-halving bucket clears the bar on the planted tape and does
    not on the pure-GBM null."""
    pb_cyc = st.phase_bucket_returns(cycle_tape[0], 4)
    pb_nul = st.phase_bucket_returns(null_tape[0], 4)
    assert pb_cyc.loc[0, "tstat"] > 2.0      # detected
    assert abs(pb_nul.loc[0, "tstat"]) < 2.0  # null: nothing to find


# ---- timing rule ----------------------------------------------------------
def test_timing_rule_columns_and_equity_positive(cycle_tape):
    bars, _ = cycle_tape
    book = st.timing_rule(bars, 0.0, 0.5, cost_bps=10.0)
    assert list(book.columns) == ["pos", "ret_bh", "ret_strat", "eq_bh", "eq_strat"]
    assert (book["eq_bh"] > 0).all()
    assert (book["eq_strat"] > 0).all()


def test_timing_rule_one_bar_lag(cycle_tape):
    """Spine #2a: the position is shifted one bar (decide at t-1, earn t). The first
    bar must therefore be flat regardless of phase."""
    bars, _ = cycle_tape
    book = st.timing_rule(bars, 0.0, 0.5, cost_bps=0.0)
    assert book["pos"].iloc[0] == 0.0


def test_timing_costs_lower_the_strategy_return(cycle_tape):
    """Spine #2b: positive one-way costs strictly reduce cumulative strategy return."""
    bars, _ = cycle_tape
    free = st.timing_rule(bars, 0.0, 0.5, cost_bps=0.0)["ret_strat"].sum()
    paid = st.timing_rule(bars, 0.0, 0.5, cost_bps=20.0)["ret_strat"].sum()
    assert paid < free


def test_timing_position_only_long_or_flat(cycle_tape):
    bars, _ = cycle_tape
    book = st.timing_rule(bars, 0.0, 0.5, cost_bps=10.0)
    assert set(np.unique(book["pos"].to_numpy())) <= {0.0, 1.0}


def test_summarize_timing_keys(cycle_tape):
    bars, _ = cycle_tape
    s = st.summarize_timing(st.timing_rule(bars, 0.0, 0.5))
    for k in ("cagr_strat", "cagr_bh", "excess_t", "time_in_market", "n_flips"):
        assert k in s

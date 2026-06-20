"""The event engine's invariants, the placebo control, the inference primitives, and the
study's two spines: (1) the event study fires only when a freeze spike is actually planted,
and (2) the reactive (lag=1) trade never earns the un-tradable freeze-day move."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oj_frost import data, strategy as st  # noqa: E402


# ---- returns & window engine ----------------------------------------------
def test_daily_log_returns_length(null_tape):
    frame, _ = null_tape
    r = st.daily_log_returns(frame)
    assert len(r) == len(frame) - 1


def test_window_ledger_columns(null_tape):
    frame, _ = null_tape
    led = st.window_returns(frame, data.freeze_dates(), window=5, lag=1, cost_bps_one_way=2.0)
    assert list(led.columns) == ["event_date", "entry_idx", "ret_gross", "ret_net", "n_obs"]


def test_net_is_gross_minus_round_trip_cost(null_tape):
    frame, _ = null_tape
    led = st.window_returns(frame, data.freeze_dates(), window=5, lag=1, cost_bps_one_way=3.0)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2 * 3.0e-4)


def test_lag_shifts_entry_forward(freeze_tape):
    """lag=1 must enter one session later than lag=0 (the documented execution lag)."""
    frame, truth = freeze_tape
    l0 = st.window_returns(frame, truth["syn_freezes"], window=5, lag=0)
    l1 = st.window_returns(frame, truth["syn_freezes"], window=5, lag=1)
    merged = l0.merge(l1, on="event_date", suffixes=("_0", "_1"))
    assert (merged["entry_idx_1"] == merged["entry_idx_0"] + 1).all()


# ---- spine #1: the event study fires only on a planted spike ---------------
def test_event_study_detects_planted_freeze(freeze_tape):
    frame, truth = freeze_tape
    led = st.window_returns(frame, truth["syn_freezes"], window=5, lag=1)
    ctrl = st.random_control_windows(frame, n_events=len(led), window=5, n_draws=1000)
    s = st.summarize_events(led, "ret_gross", control_means=ctrl)
    assert s["mean_bps"] > 0
    assert s["tstat"] > 2.0           # planted spike clears the inference bar
    assert s["placebo_pct"] > 0.95    # far in the right tail of random placebos


def test_event_study_null_is_quiet(null_tape):
    frame, _ = null_tape
    led = st.window_returns(frame, data.freeze_dates(), window=5, lag=1)
    ctrl = st.random_control_windows(frame, n_events=len(led), window=5, n_draws=1000)
    s = st.summarize_events(led, "ret_gross", control_means=ctrl)
    assert abs(s["tstat"]) < 2.0      # no planted effect -> not significant


# ---- spine #2: the reactive trade can't catch the freeze-day move ----------
def test_lag1_excludes_the_freeze_day_jump():
    """The planted spike is on the sessions AFTER each freeze. lag=1 starts the window one
    session in, so it captures less of the front-loaded spike than lag=0 does — proof the
    reactive trade can't capture the freeze-day reaction it never saw coming."""
    frame, truth = data.synthetic_oj(freeze_jump=0.30, jump_window=5, seed=309)
    m0 = st.window_returns(frame, truth["syn_freezes"], window=2, lag=0)["ret_gross"].mean()
    m1 = st.window_returns(frame, truth["syn_freezes"], window=2, lag=1)["ret_gross"].mean()
    assert m0 > m1   # entering immediately captures more of the front-loaded spike


# ---- winter seasonality ----------------------------------------------------
def test_winter_seasonality_detects_planted_tilt(winter_tape):
    frame, _ = winter_tape
    s = st.winter_seasonality(frame)
    assert s["diff_bps"] > 0
    assert s["tstat"] > 2.0


def test_winter_seasonality_quiet_on_null(null_tape):
    frame, _ = null_tape
    s = st.winter_seasonality(frame)
    assert abs(s["tstat"]) < 2.5  # no planted seasonal -> roughly insignificant


# ---- inference primitives --------------------------------------------------
def test_hac_tstat_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert abs(st.hac_tstat(x)) < 3.0


def test_hac_tstat_strong_mean_is_large():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1, 5000)
    assert st.hac_tstat(x) > 5.0


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.01, 0.02, 4000)
    lo, hi = st.block_bootstrap_ci(x, block=4, n_boot=2000)
    assert lo < x.mean() < hi


def test_random_control_returns_requested_draws(null_tape):
    frame, _ = null_tape
    ctrl = st.random_control_windows(frame, n_events=12, window=5, n_draws=500)
    assert ctrl.shape == (500,)
    assert np.isfinite(ctrl).all()


def test_summarize_empty_ledger_is_safe():
    import pandas as pd
    s = st.summarize_events(pd.DataFrame(columns=["ret_net"]))
    assert s["n_events"] == 0

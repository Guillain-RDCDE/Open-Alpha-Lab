"""Strategy-engine tests for Study 95 (Holiday-Cheer), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from holiday_cheer import data, strategy  # noqa: E402


def test_wilson_interval_brackets_proportion():
    lo, hi = strategy.wilson_interval(60, 100)
    assert lo < 0.60 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_hac_tstat_on_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 5000)
    assert abs(strategy.hac_tstat(x)) < 3.0


def test_split_returns_partition(planted_tape):
    close = planted_tape[0]["close"]
    is_pre = data.pre_holiday_mask(close.index)
    r_pre, r_rest = strategy.split_returns(close, is_pre)
    # Together they cover all but the first (NaN) return.
    assert len(r_pre) + len(r_rest) == len(close) - 1
    assert len(r_pre) > 0 and len(r_rest) > 0


def test_costs_reduce_return(planted_tape):
    close = planted_tape[0]["close"]
    is_pre = data.pre_holiday_mask(close.index)
    free = strategy.pre_holiday_only(close, is_pre, cost_bps=0.0)
    paid = strategy.pre_holiday_only(close, is_pre, cost_bps=20.0)
    assert paid["final"] < free["final"]


def test_pre_only_book_invests_only_on_pre_days(planted_tape):
    close = planted_tape[0]["close"]
    is_pre = data.pre_holiday_mask(close.index)
    book = strategy.pre_holiday_only(close, is_pre)
    assert book["days_invested"] == int(is_pre.sum())
    # It is in cash the large majority of the time.
    assert book["time_in_market"] < 0.10


def test_spine_planted_bump_is_detected(planted_tape):
    """With a planted pre-holiday bump, the harness must detect a large, significant gap."""
    close = planted_tape[0]["close"]
    is_pre = data.pre_holiday_mask(close.index)
    e = strategy.effect_stats(close, is_pre)
    assert e["gap"] > 0
    assert e["t_gap"] > 2.0  # detected


def test_spine_flat_tape_is_not_detected(flat_tape):
    """On a flat i.i.d. tape there is no pre-holiday effect - the harness must not find one."""
    close = flat_tape[0]["close"]
    is_pre = data.pre_holiday_mask(close.index)
    e = strategy.effect_stats(close, is_pre)
    assert abs(e["t_gap"]) < 2.0  # not detected


def test_spine_planted_book_beats_flat_per_day(planted_tape, flat_tape):
    """The per-invested-day mean is much higher on the planted tape than on the flat one."""
    cp = planted_tape[0]["close"]
    cf = flat_tape[0]["close"]
    ip = data.pre_holiday_mask(cp.index)
    iff = data.pre_holiday_mask(cf.index)
    bp = strategy.pre_holiday_only(cp, ip)
    bf = strategy.pre_holiday_only(cf, iff)
    assert bp["mean_invested_day"] > bf["mean_invested_day"]


def test_subperiod_contrast_detects_planted_constant_effect(planted_tape):
    """A constant planted effect across the whole tape should show ~no decay (split mid-tape)."""
    close = planted_tape[0]["close"]
    is_pre = data.pre_holiday_mask(close.index)
    mid = close.index[len(close) // 2]
    c = strategy.subperiod_contrast(close, is_pre, split_date=str(mid.date()), n_boot=500)
    # Both halves carry the bump, so the decay should not be confidently positive.
    assert c["gap_early"] > 0 and c["gap_late"] > 0
    assert 0.05 < c["p_decay_gt0"] < 0.95


def test_subperiod_contrast_handles_empty_half():
    """A split with no pre-1990 data must not crash - it returns NaN for the missing half."""
    idx = pd.bdate_range("2000-01-01", periods=2000)
    close = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    is_pre = pd.Series(False, index=idx)
    is_pre.iloc[::50] = True
    c = strategy.subperiod_contrast(close, is_pre, split_date="1990-01-01", n_boot=200)
    assert np.isnan(c["gap_early"])
    assert np.isnan(c["p_decay_gt0"])

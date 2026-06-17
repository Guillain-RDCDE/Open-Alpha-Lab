"""Strategy-engine tests for Study 247 (Bond-Seasonality), including spine tests."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bond_seasonality import data, strategy  # noqa: E402


def test_wilson_interval_brackets_proportion():
    lo, hi = strategy.wilson_interval(60, 100)
    assert lo < 0.60 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_hac_tstat_on_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 5000)
    assert abs(strategy.hac_tstat(x)) < 3.0


def test_effect_stats_returns_expected_keys(planted_tape):
    frame, _ = planted_tape
    close = frame["close"]
    is_tom = data.tom_mask(close.index)
    e = strategy.effect_stats(close, is_tom)
    for key in ("n_tom", "n_rest", "mu_tom", "mu_rest", "gap", "t_tom", "t_gap",
                "win_tom", "win_rest", "wilson_tom", "wilson_rest"):
        assert key in e, f"Missing key: {key}"


def test_tom_rest_partition(planted_tape):
    frame, _ = planted_tape
    close = frame["close"]
    is_tom = data.tom_mask(close.index)
    e = strategy.effect_stats(close, is_tom)
    assert e["n_tom"] + e["n_rest"] == len(close) - 1  # one NaN return dropped


def test_costs_reduce_return(planted_tape):
    frame, _ = planted_tape
    close = frame["close"]
    is_tom = data.tom_mask(close.index)
    free = strategy.tom_only_book(close, is_tom, cost_bps=0.0)
    paid = strategy.tom_only_book(close, is_tom, cost_bps=20.0)
    assert paid["cagr"] < free["cagr"]


def test_spine_planted_bump_is_detected(planted_tape):
    """With a planted TOM bump, the harness must detect a significant gap."""
    frame, _ = planted_tape
    close = frame["close"]
    is_tom = data.tom_mask(close.index)
    e = strategy.effect_stats(close, is_tom)
    assert e["gap"] > 0
    assert e["t_gap"] > 2.0  # positive control: must be detected


def test_spine_flat_tape_is_not_detected(flat_tape):
    """On a flat i.i.d. tape there is no TOM effect - the harness must not flag it."""
    frame, _ = flat_tape
    close = frame["close"]
    is_tom = data.tom_mask(close.index)
    e = strategy.effect_stats(close, is_tom)
    assert abs(e["t_gap"]) < 3.0  # negative control: must not be detected


def test_spine_planted_book_beats_flat_per_day(planted_tape, flat_tape):
    """The per-invested-day Sharpe is higher on the planted tape than on the flat one."""
    cp = planted_tape[0]["close"]
    cf = flat_tape[0]["close"]
    ip = data.tom_mask(cp.index)
    iff = data.tom_mask(cf.index)
    bp = strategy.tom_only_book(cp, ip)
    bf = strategy.tom_only_book(cf, iff)
    assert bp["sharpe_inv"] > bf["sharpe_inv"]


def test_tom_book_time_in_market(planted_tape):
    """TOM book is in the market roughly 15-25% of the time (4 days / ~21 per month)."""
    frame, _ = planted_tape
    close = frame["close"]
    is_tom = data.tom_mask(close.index)
    book = strategy.tom_only_book(close, is_tom)
    assert 0.10 < book["time_in_market"] < 0.35


def test_buy_and_hold_grows_on_planted_tape(planted_tape):
    frame, _ = planted_tape
    close = frame["close"]
    bh = strategy.buy_and_hold(close)
    assert bh["cagr"] > 0.0

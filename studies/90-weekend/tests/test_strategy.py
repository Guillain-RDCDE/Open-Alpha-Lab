"""Strategy/analysis-engine tests for Study 90 (Weekend), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from weekend import strategy  # noqa: E402


def test_weekday_means_shape(null_tape):
    close = null_tape[0]["close"]
    wm = strategy.weekday_means(close)
    assert list(wm.index) == ["Mon", "Tue", "Wed", "Thu", "Fri"]
    assert list(wm.columns) == ["mean_bps", "n", "hac_t"]
    # The per-weekday counts add up to the number of return bars.
    assert int(wm["n"].sum()) == len(close) - 1


def test_weekday_position_calendar_known(null_tape):
    close = null_tape[0]["close"]
    pos = strategy.tuesday_only_position(close)
    # Position is exactly 1 on Tuesdays (weekday 1), 0 otherwise — no lag.
    expected = (close.index.dayofweek == 1).astype(float)
    assert np.array_equal(pos.to_numpy(), expected)
    assert set(np.unique(pos.to_numpy())) <= {0.0, 1.0}


def test_skip_monday_excludes_only_monday(null_tape):
    close = null_tape[0]["close"]
    pos = strategy.skip_monday_position(close)
    assert (pos[close.index.dayofweek == 0] == 0).all()
    assert (pos[close.index.dayofweek != 0] == 1).all()


def test_costs_reduce_return(planted_tape):
    close = planted_tape[0]["close"]
    pos = strategy.tuesday_only_position(close)
    free = strategy.backtest(close, pos, cost_bps=0.0)
    paid = strategy.backtest(close, pos, cost_bps=20.0)
    assert paid["final"] < free["final"]


def test_hac_tstat_basic():
    # A constant-mean series with tiny noise has a large, finite, positive t.
    rng = np.random.default_rng(0)
    x = 0.001 + rng.normal(0, 1e-5, 5000)
    t = strategy.hac_tstat(x)
    assert np.isfinite(t) and t > 5


def test_contrast_sign_matches_means(null_tape):
    close = null_tape[0]["close"]
    c = strategy.contrast(close, 1)
    wm = strategy.weekday_means(close)
    # The contrast in-group mean equals the per-weekday mean for that day.
    assert abs(c["in_bps"] - wm.loc["Tue", "mean_bps"]) < 1e-6


def test_spine_planted_effect_is_detected(planted_tape):
    """With a planted Mon-down / Tue-up bump, the contrasts must clear the bar."""
    close = planted_tape[0]["close"]
    mon = strategy.contrast(close, 0)
    tue = strategy.contrast(close, 1)
    assert mon["diff_bps"] < 0 and mon["hac_t"] < -2
    assert tue["diff_bps"] > 0 and tue["hac_t"] > 2


def test_spine_null_effect_is_not_detected(null_tape):
    """On an i.i.d. tape there is no weekday structure — no contrast may clear the bar."""
    close = null_tape[0]["close"]
    mon = strategy.contrast(close, 0)
    tue = strategy.contrast(close, 1)
    assert abs(mon["hac_t"]) < 2
    assert abs(tue["hac_t"]) < 2


def test_subperiod_recovers_planted_in_both_halves(planted_tape):
    """The planted effect is stationary, so the *change* across the split is ~0 (|t| small)."""
    close = planted_tape[0]["close"]
    sp = strategy.subperiod_effect(close, 1, cut=str(close.index[len(close) // 2].date()))
    # Effect present in both halves; the change test should not flag a spurious shift.
    assert sp["pre_diff_bps"] > 0 and sp["post_diff_bps"] > 0
    assert abs(sp["hac_t_change"]) < 2

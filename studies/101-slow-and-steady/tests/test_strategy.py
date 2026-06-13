"""Strategy-engine tests for Study 101 (Slow-and-Steady), including both spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from slow_and_steady import strategy  # noqa: E402


def test_lump_sum_equals_buy_and_hold_on_flat_tape():
    # On a perfectly flat tape every purchase price is 1.0, so both arms return exactly 1.
    close = pd.Series(1.0, index=pd.bdate_range("2000-01-03", periods=600))
    out = strategy.window_wealth(close, 0, deploy_months=12)
    assert out is not None
    ls, dca = out
    assert abs(ls - 1.0) < 1e-12
    assert abs(dca - 1.0) < 1e-12


def test_window_returns_none_when_horizon_does_not_fit():
    close = pd.Series(np.linspace(1, 2, 60), index=pd.bdate_range("2000-01-03", periods=60))
    # 60 business days is < 12 months, so the window cannot complete.
    assert strategy.window_wealth(close, 0, deploy_months=12) is None


def test_dca_buys_cheaper_when_price_falls():
    # A steadily falling tape: DCA's later, cheaper tranches must beat the lump bought high.
    close = pd.Series(np.linspace(2.0, 1.0, 600), index=pd.bdate_range("2000-01-03", periods=600))
    ls, dca = strategy.window_wealth(close, 0, deploy_months=12)
    assert dca > ls


def test_dca_loses_when_price_rises():
    # A steadily rising tape: the lump bought low and held beats drip-feeding at higher prices.
    close = pd.Series(np.linspace(1.0, 2.0, 600), index=pd.bdate_range("2000-01-03", periods=600))
    ls, dca = strategy.window_wealth(close, 0, deploy_months=12)
    assert ls > dca


def test_wilson_ci_brackets_point_estimate():
    p, lo, hi = strategy.wilson_ci(800, 1000)
    assert lo < p < hi
    assert 0.0 <= lo and hi <= 1.0


def test_rolling_race_columns_and_no_partial_windows(up_tape):
    close = up_tape[0]["close"]
    race = strategy.rolling_race(close, deploy_months=12, step=10)
    assert list(race.columns) == ["lump_sum", "dca", "ls_minus_dca"]
    assert len(race) > 0
    assert np.allclose(race["ls_minus_dca"], race["lump_sum"] - race["dca"])


def test_dca_dispersion_is_tighter(up_tape):
    # DCA's signature benefit: a lower spread of terminal outcomes than lump-sum.
    close = up_tape[0]["close"]
    race = strategy.rolling_race(close, deploy_months=12, step=5)
    s = strategy.win_stats(race)
    assert s["dispersion_ratio"] < 1.0


def test_spine_positive_drift_lump_sum_wins(up_tape):
    """Equities' empirical case: on an up-drifting tape lump-sum wins the majority of windows."""
    close = up_tape[0]["close"]
    race = strategy.rolling_race(close, deploy_months=12, step=5)
    s = strategy.win_stats(race)
    assert s["ls_win_rate"] > 0.5
    assert s["mean_gap"] > 0  # lump-sum richer on average


def test_spine_negative_drift_dca_wins(down_tape):
    """The case folklore is built on: on a down-drifting tape DCA wins the majority of windows."""
    close = down_tape[0]["close"]
    race = strategy.rolling_race(close, deploy_months=12, step=5)
    s = strategy.win_stats(race)
    assert s["ls_win_rate"] < 0.5
    assert s["mean_gap"] < 0  # DCA richer on average

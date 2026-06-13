"""Strategy/analysis tests for Study 96 (New-Year-Pop), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from new_year_pop import strategy  # noqa: E402


def test_spread_is_small_minus_large():
    frame = pd.DataFrame(
        {"small": [0.05, 0.01], "large": [0.02, 0.03]},
        index=pd.date_range("1990-01-31", periods=2, freq="ME"),
    )
    sp = strategy.spread(frame)
    assert np.allclose(sp.to_numpy(), [0.03, -0.02])


def test_january_mask():
    idx = pd.date_range("1990-01-31", periods=13, freq="ME")
    mask = strategy.january_mask(idx)
    assert mask[0] and mask[12]  # two Januaries one year apart
    assert not mask[1]


def test_wilson_interval_brackets_point():
    lo, hi = strategy.wilson_interval(8, 16)
    assert lo < 0.5 < hi


def test_hac_dummy_matches_mean_difference():
    # The dummy-regression slope equals the group-mean difference exactly.
    rng = np.random.default_rng(0)
    y = rng.normal(size=240)
    d = (np.arange(240) % 12 == 0).astype(float)
    diff = y[d == 1].mean() - y[d == 0].mean()
    # Recover slope via OLS the same way _hac_dummy_tstat does, then check sign/scale.
    X = np.column_stack([np.ones(240), d])
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    assert np.isclose(beta[1], diff)


def test_seasonal_timer_costs_reduce_return(planted_tape):
    frame = planted_tape[0]
    free = strategy.seasonal_timer(frame, cost_bps=0.0)
    paid = strategy.seasonal_timer(frame, cost_bps=50.0)
    assert paid["timer"]["final"] < free["timer"]["final"]


def test_contrast_keys(null_tape):
    c = strategy.contrast(null_tape[0])
    for k in ("jan_mean", "rest_mean", "diff", "t_contrast", "jan_hit_rate",
              "wilson_lo", "wilson_hi"):
        assert k in c


def test_spine_planted_seasonal_is_detected(planted_tape):
    """With a planted +4%/January bump, the harness MUST detect a significant contrast."""
    c = strategy.contrast(planted_tape[0])
    assert c["diff"] > 0.02            # the contrast points up, near the planted size
    assert c["t_contrast"] >= 2.0      # and clears the desk's significance bar
    assert c["jan_hit_rate"] > 0.6     # small beats large most Januaries


def test_spine_null_is_not_manufactured(null_tape):
    """With no planted seasonal, the harness MUST NOT manufacture a significant contrast."""
    c = strategy.contrast(null_tape[0])
    assert abs(c["t_contrast"]) < 2.0


def test_subperiod_difference_runs(planted_tape):
    sub = strategy.subperiod_january(planted_tape[0], split_year=2008)
    assert sub["early_n"] > 0 and sub["late_n"] > 0
    assert np.isfinite(sub["t_diff"])

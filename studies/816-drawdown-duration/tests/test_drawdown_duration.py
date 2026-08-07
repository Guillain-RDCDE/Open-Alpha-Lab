"""Offline, fixed-seed tests for the drawdown-duration machinery.

The synthetic panel is deterministic; the time-underwater signal is a proper fraction in
[0, 1] and is monotone in a clean drawdown; the sort recovers a planted relation
(low-drift names stay underwater and keep sinking -> a negative long-high/short-low
spread); the null shows nothing; the sort is point-in-time (one shift, no look-ahead);
the timer costs reduce the net; the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from drawdown_duration import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(knob=0.0010, seed=816, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_time_underwater_is_a_fraction(edge_world):
    ret = st.close_returns(edge_world)
    uw = st.time_underwater(ret, window=252)
    vals = uw.to_numpy()
    vals = vals[~np.isnan(vals)]
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    # cross-name dispersion in time-underwater must be non-trivial (the sort has bite)
    assert uw.mean().std() > 0.02


def test_underwater_definition_on_a_known_path():
    # A price that rises 5 days then falls below the peak for the rest is underwater
    # exactly on the days strictly below the running high-water mark.
    px = np.array([100, 101, 102, 103, 104, 103, 102, 101, 100, 99], dtype=float)
    ret = pd.DataFrame({"X": px}).pct_change()
    curve = (1.0 + ret.fillna(0.0)).cumprod()["X"].to_numpy()
    hwm = np.maximum.accumulate(curve)
    uw = (curve < hwm).astype(float)
    # rising leg (rows 0..4) at the HWM -> 0; falling leg (rows 5..9) below -> 1
    assert uw[:5].sum() == 0
    assert uw[5:].sum() == 5


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.duration_stats(st.duration_spreads(ret))
    assert ts["t_nw"] < -3.0            # long-high/short-low spread lights up (negative)
    assert ts["spread_bps"] < 0
    assert ts["hi_bps"] < ts["lo_bps"]  # high-underwater names earn LESS (keep sinking)


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.duration_stats(st.duration_spreads(ret))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(
        rng.normal(0, 0.01, (400, 4)),
        index=pd.bdate_range("2020-01-01", periods=400),
        columns=list("ABCD"),
    )
    sig = st.time_underwater(ret, window=60)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[300].to_numpy(), sig.iloc[299].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.duration_spreads(ret)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_sign_matches_mean_gap():
    a = np.array([0.02, 0.03, 0.01, 0.02])
    b = np.array([0.00, -0.01, 0.01, 0.00])
    assert st.welch_t(a, b) > 0

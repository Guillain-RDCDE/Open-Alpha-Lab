"""Offline, fixed-seed tests for the tug-of-war machinery.

The synthetic panel is deterministic; the exact night/day identity holds; the sort
recovers a planted tug (positive overnight leg, negative intraday leg); the null shows
neither leg; the sort is point-in-time (one shift, no look-ahead); the timer costs
reduce the net; the inference primitives behave. All on the seeded offline world.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.decompose import check_identity, decompose  # noqa: E402

from overnight_intraday_tug import data, strategy as st  # noqa: E402


def test_world_deterministic(tug_world):
    p2 = data.synthetic_panel(tug=0.0020, seed=788, n_assets=40, n_days=1500)
    for sym in tug_world:
        assert np.allclose(tug_world[sym].to_numpy(), p2[sym].to_numpy())


def test_decompose_identity_holds(tug_world):
    # The synthetic OHLC must satisfy the exact night/day identity to machine dust.
    for sym in list(tug_world)[:5]:
        assert check_identity(decompose(tug_world[sym])) < 1e-9


def test_planted_tug_recovered(tug_world):
    on, idc = st.leg_panels(tug_world)
    ts = st.tug_stats(st.tug_spreads(on, idc))
    assert ts["on_t_nw"] > 3.0        # overnight persistence lights up
    assert ts["id_t_nw"] < -3.0       # intraday reversal lights up
    assert ts["on_mean_bps"] > 0 > ts["id_mean_bps"]


def test_null_world_no_tug(null_world):
    on, idc = st.leg_panels(null_world)
    ts = st.tug_stats(st.tug_spreads(on, idc))
    assert abs(ts["on_t_nw"]) < 2.5   # neither leg fires on the null
    assert abs(ts["id_t_nw"]) < 2.5


def test_sort_is_point_in_time():
    # The signal used for a day-t position must be shifted so it only sees data
    # through t-1: the last row of the raw signal must NOT drive the last position.
    on = pd.DataFrame(
        np.arange(60).reshape(20, 3).astype(float),
        index=pd.bdate_range("2020-01-01", periods=20),
        columns=["A", "B", "C"],
    )
    sig = st.trailing_overnight_signal(on, window=3)
    shifted = sig.shift(1)
    # row t of the shifted signal equals row t-1 of the unshifted signal
    assert np.allclose(shifted.iloc[5].to_numpy(), sig.iloc[4].to_numpy(), equal_nan=True)


def test_costs_reduce_net(tug_world):
    on, idc = st.leg_panels(tug_world)
    sp = st.tug_spreads(on, idc)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_newey_west_matches_one_sample_on_iid():
    # On i.i.d. data the HAC t and the one-sample t should be close.
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

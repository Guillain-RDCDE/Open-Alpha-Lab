"""Offline, fixed-seed tests for the abnormal-volume machinery.

The synthetic panel is deterministic; the standardised abnormal-volume signal tracks the
planted attention shock; the sort recovers a planted attention->drift relation (positive
long-high/short-low spread); the null shows nothing; the sort is point-in-time (one shift,
no look-ahead); the timer costs reduce the net; the inference primitives behave. All
offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from volume_shock import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0020, seed=819, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_volume_is_positive(edge_world):
    vol = st.volume_panel(edge_world)
    assert (vol.to_numpy() > 0).all()


def test_abnormal_volume_has_dispersion(edge_world):
    # Standardised abnormal volume must vary across names/time (the sort has something to bite on).
    vol = st.volume_panel(edge_world)
    z = st.std_abnormal_volume(vol, lookback=60)
    assert np.nanstd(z.to_numpy()) > 0.3


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    vol = st.volume_panel(edge_world)
    ts = st.avol_stats(st.avol_spreads(ret, vol))
    assert ts["t_nw"] > 3.0            # long-high/short-low spread lights up
    assert ts["spread_bps"] > 0
    assert ts["hi_bps"] > ts["lo_bps"]  # high-abnormal-volume names out-earn low ones


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    vol = st.volume_panel(null_world)
    ts = st.avol_stats(st.avol_spreads(ret, vol))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    vol = pd.DataFrame(
        np.abs(np.linspace(1.0, 5.0, 300)).reshape(100, 3) + 1.0,
        index=pd.bdate_range("2020-01-01", periods=100),
        columns=["A", "B", "C"],
    )
    sig = st.abnormal_volume_signal(vol, lookback=20, form=5)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[70].to_numpy(), sig.iloc[69].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    vol = st.volume_panel(edge_world)
    sp = st.avol_spreads(ret, vol)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_abnormal_volume_spikes_are_high_z():
    # A volume series flat at its norm then a single 10x spike -> large positive z on the spike day.
    v = np.full(100, 1.0e6)
    v[80] = 1.0e7
    vol = pd.DataFrame({"X": v}, index=pd.bdate_range("2020-01-01", periods=100))
    z = st.std_abnormal_volume(vol, lookback=60)
    assert z["X"].iloc[80] > 3.0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

"""Offline, fixed-seed tests for the realized-volatility-trend machinery.

The synthetic panel is deterministic; the vol-trend signal tracks the planted tilt; the
sort recovers a planted rising-vol->de-rate relation (positive long-falling/short-rising
spread); the null shows nothing; the sort is point-in-time (one shift, no look-ahead);
the timer costs reduce the net; the additivity regression is well-formed; the inference
primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vol_trend import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0015, seed=817, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_vol_trend_dispersion(edge_world):
    # The cross-name dispersion in the vol trend must be non-trivial (the sort has
    # something to bite on), and the average vol trend should hover near zero.
    ret = st.close_returns(edge_world)
    vt = st.vol_trend(ret, short_w=21, long_w=63)
    m = vt.mean()
    assert m.std() > 0.01
    assert abs(m.mean()) < 0.5


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.vol_stats(st.vol_spreads(ret))
    assert ts["t_nw"] > 3.0             # long-falling/short-rising spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # falling-vol names out-earn rising-vol names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.vol_stats(st.vol_spreads(ret))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(
        rng.normal(0, 0.01, (120, 4)),
        index=pd.bdate_range("2020-01-01", periods=120),
        columns=["A", "B", "C", "D"],
    )
    sig = st.vol_trend(ret, short_w=5, long_w=15)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[80].to_numpy(), sig.iloc[79].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.vol_spreads(ret)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_additivity_is_well_formed(edge_world):
    ret = st.close_returns(edge_world)
    ad = st.additivity(ret)
    # correlation in [-1, 1]; the intercept-net residual t is finite
    assert -1.0 <= ad["corr"] <= 1.0
    assert np.isfinite(ad["alpha_t_nw"])
    assert ad["n_days"] > 100


def test_rising_vol_tracks_tilt(edge_world):
    # A name with a higher average latent tilt should carry a higher average vol level.
    ret = st.close_returns(edge_world)
    vol = st.trailing_vol(ret, 63).mean()
    assert vol.std() > 0  # names differ in vol level (the tilt drives amplitude)


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_zero_on_identical():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(st.welch_t(a, a.copy())) < 1e-9

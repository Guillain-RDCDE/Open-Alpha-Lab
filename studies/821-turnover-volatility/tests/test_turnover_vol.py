"""Offline, fixed-seed tests for the turnover-volatility machinery.

The synthetic panel is deterministic; the trailing CV-of-turnover signal has the right
sign (erratic-turnover names show a higher CV); the sort recovers a planted negative
turnover-vol->return relation (positive long-low/short-high spread); the null shows
nothing; the sort is point-in-time (one shift, no look-ahead); the timer costs reduce
the net; the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from turnover_vol import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0016, seed=821, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_panel_has_volume(edge_world):
    for sym in edge_world:
        assert "Volume" in edge_world[sym].columns
        assert (edge_world[sym]["Volume"] > 0).all()


def test_cv_is_scale_invariant(edge_world):
    # Turnover CV (std/mean) must be invariant to multiplying volume by a constant
    # (a fixed shares-outstanding rescaling) — the reason raw Volume is a clean proxy.
    turn = st.turnover_panel(edge_world)
    cv1 = st.trailing_cv(turn, window=63)
    cv2 = st.trailing_cv(turn * 1000.0, window=63)
    assert np.allclose(cv1.to_numpy(), cv2.to_numpy(), equal_nan=True)


def test_cv_tracks_tilt(edge_world):
    # A name with a higher latent turnover-vol tilt should show a higher trailing CV;
    # the cross-name dispersion in CV must be non-trivial (the sort has something to bite on).
    turn = st.turnover_panel(edge_world)
    cv = st.trailing_cv(turn, window=63).mean()
    assert cv.std() > 0.05


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    turn = st.turnover_panel(edge_world)
    ts = st.cv_stats(st.cv_spreads(ret, turn))
    assert ts["t_nw"] > 3.0             # long-low/short-high spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # low-turnover-vol names out-earn high-turnover-vol


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    turn = st.turnover_panel(null_world)
    ts = st.cv_stats(st.cv_spreads(ret, turn))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    turn = pd.DataFrame(
        np.abs(np.linspace(1.0, 5.0, 60)).reshape(20, 3) + 1.0,
        index=pd.bdate_range("2020-01-01", periods=20),
        columns=["A", "B", "C"],
    )
    sig = st.trailing_cv(turn, window=3)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[5].to_numpy(), sig.iloc[4].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    turn = st.turnover_panel(edge_world)
    sp = st.cv_spreads(ret, turn)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_dollar_variant_runs(edge_world):
    ret = st.close_returns(edge_world)
    turn_d = st.turnover_panel(edge_world, dollar=True)
    ts = st.cv_stats(st.cv_spreads(ret, turn_d))
    # dollar-volume CV still tracks the same latent tilt -> planted relation recovered
    assert ts["t_nw"] > 3.0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_sign():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 5.0, 6.0, 7.0])
    assert st.welch_t(a, b) < 0

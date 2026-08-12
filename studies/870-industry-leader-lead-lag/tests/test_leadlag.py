"""Offline, fixed-seed tests for the industry-leader lead-lag machinery.

The synthetic panel is deterministic; the leader->follower diffusion is recovered by
the sort (positive spread, follower-after-up > follower-after-down); the null shows
nothing; the sort is point-in-time (leader week w -> follower week w+1, one shift);
costs reduce the net; the dollar-volume proxy re-designates the planted leaders; the
inference primitives behave. All offline — no real cache required.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from leader_lag import data, strategy as st  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_cache")
REAL_CACHE = os.path.join(CACHE, "panel_50_a2964d3d2ba7_2010-01-01.parquet")


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.6, seed=870, n_weeks=320)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_planted_relation_recovered(edge_world, sectors, leaders):
    wret = st.weekly_returns(edge_world)
    sp = st.leadlag_spreads(wret, sectors, leaders)
    ts = st.leadlag_stats(sp, wret, sectors, leaders)
    assert ts["t_nw"] > 3.0            # long-up / short-down spread lights up
    assert ts["spread_bps"] > 0
    assert ts["up_bps"] > ts["dn_bps"]  # followers earn more after an up-leader week


def test_null_world_no_signal(null_world, sectors, leaders):
    wret = st.weekly_returns(null_world)
    ts = st.leadlag_stats(st.leadlag_spreads(wret, sectors, leaders))
    assert abs(ts["t_nw"]) < 2.5


def test_placebo_p_high_under_null(null_world, sectors, leaders):
    wret = st.weekly_returns(null_world)
    pl = st.placebo_pvalue(wret, sectors, leaders, n_seeds=4, n_draws_per_seed=25)
    assert pl["n_draws"] > 0
    assert pl["p_value"] > 0.05        # observed not in the right tail under the null


def test_placebo_p_low_when_planted(edge_world, sectors, leaders):
    wret = st.weekly_returns(edge_world)
    pl = st.placebo_pvalue(wret, sectors, leaders, n_seeds=4, n_draws_per_seed=25)
    assert pl["obs_bps"] > pl["placebo_mean_bps"]
    assert pl["p_value"] < 0.05        # planted spread sits in the right tail


def test_sort_is_point_in_time(sectors, leaders):
    # A leader spike in exactly one week must move the spread only in the NEXT week.
    wret = pd.DataFrame(
        0.0,
        index=pd.date_range("2020-01-03", periods=8, freq="W-FRI"),
        columns=["S0L", "S0F0", "S0F1"],
    )
    secs = {"SEC0": ["S0L", "S0F0", "S0F1"]}
    lds = {"SEC0": "S0L"}
    wret.loc[wret.index[3], "S0L"] = 0.05          # leader up in week 3
    wret.loc[wret.index[4], ["S0F0", "S0F1"]] = 0.02  # followers up in week 4
    sp = st.leadlag_spreads(wret, secs, lds, min_sectors=1)
    # the only non-zero contribution is dated to week 4 (the followers' week)
    nz = sp[sp["spread"] != 0]
    assert len(nz) == 1
    assert nz.index[0] == wret.index[4]
    assert nz["spread"].iloc[0] > 0


def test_dollar_volume_recovers_leaders(edge_world, sectors, leaders):
    dv = st.dollar_volume(edge_world)
    dyn = st.designate_leaders(sectors, dv)
    assert dyn == leaders              # leaders carry ~10x follower volume by construction


def test_costs_reduce_net(edge_world, sectors, leaders):
    wret = st.weekly_returns(edge_world)
    sp = st.leadlag_spreads(wret, sectors, leaders)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_weekly_returns_shape(edge_world):
    wret = st.weekly_returns(edge_world)
    assert wret.shape[1] == len(edge_world)
    assert wret.index.is_monotonic_increasing


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_sign():
    a = np.array([0.02, 0.03, 0.02, 0.04])
    b = np.array([-0.01, 0.0, -0.02, 0.01])
    assert st.welch_t(a, b) > 0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


# ---- the ONE test that touches the real cache is gated so CI (no cache) stays green ---
@pytest.mark.skipif(not os.path.exists(REAL_CACHE), reason="real cache absent offline CI")
def test_real_panel_loads_and_runs():
    panel = data.load_panel()
    assert len(panel) >= 40
    wret = st.weekly_returns(panel)
    sp = st.leadlag_spreads(wret, data.SECTORS, data.LEADERS)
    ts = st.leadlag_stats(sp, wret, data.SECTORS, data.LEADERS)
    assert ts["n_weeks"] > 400
    assert np.isfinite(ts["t_nw"])

"""Offline, fixed-seed tests for the 52-week-high breakout machinery.

The synthetic panel is deterministic; the breakout flag is point-in-time (today excluded,
one shift, no look-ahead); the event sort recovers a planted breakout->forward-drift
relation (positive long-breakout / short-rest spread); the null shows nothing; the timer
costs reduce the net; the inference primitives behave. All offline, synthetic-only — the
suite passes with NO real cache present.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from breakout_high import data, strategy as st  # noqa: E402

REAL_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache", "panel_50_a2964d3d2ba7_2010-01-01.parquet",
)


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0015, seed=869, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_breakout_flag_is_fresh_high():
    # A strictly increasing series: every day after the first full window is a new high.
    closes = pd.DataFrame(
        {"A": np.arange(1.0, 401.0)},
        index=pd.bdate_range("2015-01-01", periods=400),
    )
    flags = st.breakout_flags(closes, lookback=252)
    # first 252 rows: no full prior window -> no flag; from row 252 on: a fresh high daily
    assert not flags["A"].iloc[:252].any()
    assert flags["A"].iloc[252:].all()


def test_flag_excludes_today_no_lookahead():
    # A single spike then flat: only the spike day is a fresh high, never a later day.
    x = np.full(300, 100.0)
    x[280] = 200.0        # one spike above the trailing max
    closes = pd.DataFrame({"A": x}, index=pd.bdate_range("2015-01-01", periods=300))
    flags = st.breakout_flags(closes, lookback=252)
    assert flags["A"].iloc[280]            # the spike is a fresh high
    assert not flags["A"].iloc[281]        # the day after (equal-or-below) is not
    # the flag on row t uses only prior closes: shifting the prior-max is the guard
    prior_max = closes.rolling(252, min_periods=252).max().shift(1)
    assert np.isnan(prior_max["A"].iloc[251])   # row 251 has no full prior window


def test_forward_return_has_one_lag():
    closes = pd.DataFrame(
        {"A": np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])},
        index=pd.bdate_range("2020-01-01", periods=6),
    )
    fwd = st.forward_returns(closes, horizon=2, lag=1)
    # fwd[0] enters at close[1]=11, exits at close[1+2]=close[3]=13 -> 13/11-1
    assert np.isclose(fwd["A"].iloc[0], 13.0 / 11.0 - 1.0)


def test_planted_relation_recovered(edge_world):
    closes = st.closes_frame(edge_world)
    ts = st.breakout_stats(st.breakout_spreads(closes, horizon=5), nw_lags=10)
    assert ts["t_nw"] > 3.0             # long-breakout / short-rest spread lights up
    assert ts["spread_bps"] > 0
    assert ts["brk_bps"] > ts["rest_bps"]   # breakout names out-earn the rest


def test_null_world_no_signal(null_world):
    closes = st.closes_frame(null_world)
    ts = st.breakout_stats(st.breakout_spreads(closes, horizon=5), nw_lags=20)
    assert abs(ts["t_nw"]) < 2.5


def test_costs_reduce_net(edge_world):
    closes = st.closes_frame(edge_world)
    sp = st.breakout_spreads(closes, horizon=5)
    gross = st.timer_stats(sp, horizon=5, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, horizon=5, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_spread_is_brk_minus_rest(edge_world):
    closes = st.closes_frame(edge_world)
    sp = st.breakout_spreads(closes, horizon=5)
    assert np.allclose((sp["brk"] - sp["rest"]).to_numpy(), sp["spread"].to_numpy())
    assert (sp["n_brk"] >= 1).all()


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_sign():
    a = np.array([2.0, 3.0, 2.5, 3.5, 3.0])
    b = np.array([0.0, 1.0, 0.5, 1.5, 1.0])
    assert st.welch_t(a, b) > 0


import pytest  # noqa: E402


@pytest.mark.skipif(not os.path.exists(REAL_CACHE), reason="real cache absent offline CI")
def test_real_cache_loads_if_present():
    panel = data.load_panel()
    assert len(panel) > 0
    closes = st.closes_frame(panel)
    assert "Close" in panel[next(iter(panel))].columns
    assert len(closes) > 1000

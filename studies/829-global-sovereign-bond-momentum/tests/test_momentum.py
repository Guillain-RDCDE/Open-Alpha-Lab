"""Offline, fixed-seed tests for the global-sovereign-bond-momentum machinery.

The synthetic panel is deterministic; the 12-1 momentum has the right definition and no
look-ahead; the time-series-momentum backtest recovers a planted trend (positive mean,
Newey-West t lights up) and stays silent on the null; costs reduce the net; the inference
primitives behave. Every test is offline synthetic-only — the one test that touches the
real cache is skipped when the cache is absent.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sovereign_momentum import data, strategy as st  # noqa: E402

CACHE = data.cache_path()


# --------------------------------------------------------------------------- #
# Synthetic determinism + signal definition
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.010, seed=829)
    assert np.allclose(edge_world.to_numpy(), p2.to_numpy())


def test_momentum_definition():
    # A clean geometric series: level_t = 1.1**t. mom_t = level_{t-1}/level_{t-12}-1 = 1.1**11-1.
    n = 40
    lvl = pd.DataFrame({"A": 1.1 ** np.arange(n)},
                       index=pd.date_range("2000-01-31", periods=n, freq="ME"))
    mom = st.momentum_12_1(lvl, lookback=12, gap=1)
    assert np.isnan(mom["A"].iloc[11])          # not enough history at row 11
    assert abs(mom["A"].iloc[20] - (1.1 ** 11 - 1)) < 1e-9


def test_momentum_skips_most_recent_month():
    # gap=1 => the value on row t uses prices only through t-1 (the most recent month is skipped).
    lvl = pd.DataFrame(
        np.cumprod(1 + np.linspace(-0.01, 0.02, 30 * 2).reshape(30, 2), axis=0),
        index=pd.date_range("2005-01-31", periods=30, freq="ME"), columns=["A", "B"],
    )
    mom = st.momentum_12_1(lvl, lookback=12, gap=1)
    # equals the 12->1 shifted ratio, which never references the same-row price
    expect = lvl.shift(1) / lvl.shift(12) - 1.0
    assert np.allclose(mom.to_numpy(), expect.to_numpy(), equal_nan=True)


def test_positions_are_point_in_time():
    # positions.iloc[t] = sign(mom).shift(1) => uses momentum known at t-1 only.
    lvl = data.synthetic_panel(edge=0.010, seed=829)
    mom = st.momentum_12_1(lvl)
    pos = st.positions(lvl)
    assert np.allclose(pos.iloc[20].to_numpy(),
                       np.sign(mom).iloc[19].to_numpy(), equal_nan=True)


def test_no_lookahead_future_shock_only_moves_last_month(edge_world):
    bt1 = st.tsmom_returns(edge_world)
    shocked = edge_world.copy()
    shocked.iloc[-1] = shocked.iloc[-1] * 1.5   # a shock to the final (future) month only
    bt2 = st.tsmom_returns(shocked)
    common = bt1.index.intersection(bt2.index)[:-1]  # every month before the shocked one
    assert np.allclose(bt1.loc[common, "ret"].to_numpy(),
                       bt2.loc[common, "ret"].to_numpy())


# --------------------------------------------------------------------------- #
# Planted effect recovered / null silent
# --------------------------------------------------------------------------- #
def test_planted_trend_recovered(edge_world):
    s = st.tsmom_stats(st.tsmom_returns(edge_world))
    assert s["mean_bps"] > 0
    assert s["t_nw"] > 3.0          # the trend detector lights up on a planted effect
    assert s["sharpe"] > 0.5


def test_null_world_no_signal(null_world):
    s = st.tsmom_stats(st.tsmom_returns(null_world))
    assert abs(s["t_nw"]) < 2.5     # nothing to find in a random walk


def test_null_silent_over_many_seeds():
    res = st.synthetic_mean_t(data, edge=0.0, n_seeds=20, base_seed=829)
    assert abs(res["mean_t"]) < 1.0
    assert res["fire_frac"] <= 0.10   # essentially never fires on the null


def test_placebo_separates_planted_from_null(edge_world, null_world):
    pe = st.placebo_pvalue(edge_world, n_perm=400)
    pn = st.placebo_pvalue(null_world, n_perm=400)
    assert pe["p_value"] < 0.05        # planted world sits in the right tail
    assert pn["p_value"] > 0.10        # null world does not


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(edge_world):
    gross = st.timer_stats(edge_world, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(edge_world, cost_bps=10.0, borrow_bps_yr=75.0)["net_bps"]
    assert net < gross


def test_long_only_has_no_short_borrow(edge_world):
    tm = st.timer_stats(edge_world, long_only=True, cost_bps=0.0, borrow_bps_yr=500.0)
    tm0 = st.timer_stats(edge_world, long_only=True, cost_bps=0.0, borrow_bps_yr=0.0)
    assert abs(tm["net_bps"] - tm0["net_bps"]) < 1e-9   # no shorts => borrow irrelevant


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_t_detects_mean_gap():
    rng = np.random.default_rng(1)
    a = rng.normal(0.02, 0.05, 500)
    b = rng.normal(0.00, 0.05, 500)
    assert st.welch_t(a, b) > 3.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


# --------------------------------------------------------------------------- #
# Real cache (skipped offline)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(CACHE), reason="real cache absent offline CI")
def test_real_panel_loads_and_is_monthly():
    panel = data.load_panel()
    assert panel.index.max() <= pd.Timestamp(data.AS_OF)
    assert set(["BWX", "IGOV", "BNDX", "EMB", "IEF"]).issuperset(set(panel.columns))
    bt = st.tsmom_returns(panel)
    assert len(bt) > 100

"""Offline, fixed-seed tests for the corporate-bond-ladder machinery.

The synthetic world is deterministic; a planted ladder premium is recovered with the right
sign and significance; the null (matched-duration ladder == fund) shows nothing; the fixed-
weight basket is point-in-time (no look-ahead); costs reduce the net diff; duration-matching
hits the target; and the inference primitives behave. All offline, no real cache required.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bond_ladder import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    w2 = data.synthetic_world(n_months=228, edge_annual=0.015, seed=892)
    assert np.allclose(edge_world["ladder"].to_numpy(), w2["ladder"].to_numpy())
    assert np.allclose(edge_world["fund"].to_numpy(), w2["fund"].to_numpy())


def test_synthetic_index_no_overflow():
    # A long horizon must not overflow the pandas ns-Timestamp cap: we use a RangeIndex.
    w = data.synthetic_world(n_months=6000, edge_annual=0.0, seed=1)
    assert isinstance(w.index, pd.RangeIndex)
    assert len(w) == 6000


def test_planted_premium_recovered(edge_world):
    d = st.synthetic_detect(edge_world)
    assert d["t_hac"] > 3.0                 # the planted ladder premium lights up
    assert d["diff_ann_pct"] > 0.5          # and is positive, ~ the +1.5%/yr knob
    assert 0.8 < d["diff_ann_pct"] < 2.5


def test_null_world_no_signal(null_world):
    d = st.synthetic_detect(null_world)
    assert abs(d["t_hac"]) < 2.5            # matched-duration ladder == fund: nothing to find


def test_edge_scales_with_knob():
    lo = st.synthetic_detect(data.synthetic_world(edge_annual=0.005, seed=7))["diff_ann_pct"]
    hi = st.synthetic_detect(data.synthetic_world(edge_annual=0.030, seed=7))["diff_ann_pct"]
    assert hi > lo + 1.0                    # a bigger planted knob => a bigger recovered premium


# --------------------------------------------------------------------------- #
# Portfolio construction
# --------------------------------------------------------------------------- #
def test_fixed_weight_basket_is_point_in_time():
    # A static buy-and-hold basket multiplies the SAME month's return — no shift, no look-ahead.
    ret = pd.DataFrame(
        {"SHY": np.linspace(0.001, 0.02, 10), "IEF": np.linspace(-0.01, 0.01, 10)},
        index=pd.period_range("2020-01", periods=10, freq="M").to_timestamp(how="end"),
    )
    w = {"SHY": 0.6, "IEF": 0.4}
    port = st.portfolio_returns(ret, w)
    expected = 0.6 * ret["SHY"] + 0.4 * ret["IEF"]
    assert np.allclose(port.to_numpy(), expected.to_numpy())


def test_duration_matching_hits_target():
    # The duration-matched ladder is tuned to ~6.0y (the fund's duration).
    dur = data.ladder_duration(data.DUR_LADDER)
    assert abs(dur - 6.0) < 0.15
    # The naive equal-weight ladder is materially LONGER (its apparent edge is a duration bet).
    assert data.ladder_duration(data.EW_LADDER) > dur + 1.0


def test_costs_reduce_net(edge_world):
    # Costs can only lower the ladder-minus-fund gap (the fund is buy-and-hold, free).
    ret = pd.DataFrame({"SHY": edge_world["ladder"] * 0.4, "IEF": edge_world["ladder"] * 0.6,
                        "AGG": edge_world["fund"], "BIL": edge_world["cash"]})
    ret.index = pd.period_range("2007-06", periods=len(ret), freq="M").to_timestamp(how="end")
    w = {"SHY": 0.4, "IEF": 0.6}
    c = st.costed_race(ret, w, fund="AGG", cash="BIL", spread_bps_oneway=5.0, annual_turnover=0.3)
    assert c["net_diff_ann_pct"] < c["gross_diff_ann_pct"]
    assert c["ladder_cost_bps_yr"] > 0


def test_max_drawdown_sign():
    port = pd.Series([0.05, -0.20, 0.03, -0.10, 0.08])
    assert st.max_drawdown(port) < 0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_detects_mean_gap():
    rng = np.random.default_rng(1)
    a = rng.normal(0.02, 0.01, 500)
    b = rng.normal(0.0, 0.01, 500)
    assert st.welch_t(a, b) > 5.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_bootstrap_sharpe_ci_brackets_point():
    rng = np.random.default_rng(2)
    x = rng.normal(0.004, 0.015, 300)
    b = st.block_bootstrap_sharpe_ci(x, n_boot=500, seed=2)
    assert b["ci_low"] < b["sharpe"] < b["ci_high"]


# --------------------------------------------------------------------------- #
# Real-cache test — skipped when the git-ignored cache is absent (e.g. on CI)
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


@pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE),
                    reason="real price cache absent (offline / CI)")
def test_real_cache_race_sane():
    prices = data.load_prices()
    ret = data.monthly_returns(prices)
    assert len(ret) > 150
    r = st.race(ret, data.DUR_LADDER, fund="AGG", cash="BIL")
    # Duration-matched, the ladder-minus-fund premium is economically ~0 (HTM is an accounting
    # illusion for default-free bonds) — |t| should be small, not a clean rejection.
    assert abs(r["t_hac"]) < 2.0
    assert r["diff_sharpe_ci"][0] < 0 < r["diff_sharpe_ci"][1]

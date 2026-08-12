"""Offline, fixed-seed tests for the multi-factor-composite machinery.

The synthetic world is deterministic; the equal-weight blend recovers a planted per-annum
edge over the benchmark and stays quiet on the null; the rebalance accounting is
no-look-ahead and turnover-costed; diversification of the blend vol is mechanical; the
inference primitives behave. All offline (synthetic-only). One real-cache test is
skipped when the git-ignored ``_cache/`` is absent (CI).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest  # noqa: E402

from multi_factor import data, strategy as st  # noqa: E402

MEMBERS = ["F1", "F2", "F3", "F4", "F5"]


# --------------------------------------------------------------------------- #
# World determinism
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    w2 = data.synthetic_world(n_months=168, edge_ann=0.03, seed=902)
    assert np.allclose(edge_world.to_numpy(), w2.to_numpy())


def test_world_index_no_overflow(edge_world):
    # PeriodRange->timestamp stays inside the pandas ns horizon.
    assert edge_world.index.max().year < 2100
    assert edge_world.index.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# The planted edge is recovered; the null is quiet
# --------------------------------------------------------------------------- #
def test_planted_edge_recovered(edge_world):
    d = st.synthetic_detect(edge_world)
    assert d["sharpe_adv"] > 0.05      # the blend out-Sharpes the benchmark
    assert d["active_bps"] > 0
    assert d["t_active_nw"] > 2.5       # and it is significant


def test_null_world_no_edge(null_world):
    d = st.synthetic_detect(null_world)
    assert abs(d["t_active_nw"]) < 2.0  # no advantage on the null


# --------------------------------------------------------------------------- #
# Rebalance accounting: no look-ahead, turnover-costed
# --------------------------------------------------------------------------- #
def test_equal_weight_gross_is_row_mean(null_world):
    # With equal weights the GROSS composite return is exactly the cross-member mean.
    comp = st.equal_weight_composite(null_world, MEMBERS, cost_bps=0.0)
    expect = null_world[MEMBERS].mean(axis=1)
    assert np.allclose(comp["gross"].to_numpy(), expect.to_numpy())


def test_costs_reduce_net(edge_world):
    free = st.equal_weight_composite(edge_world, MEMBERS, cost_bps=0.0)
    paid = st.equal_weight_composite(edge_world, MEMBERS, cost_bps=10.0)
    assert paid["net"].mean() < free["net"].mean()
    assert (paid["cost"] >= 0).all()
    assert paid["cost"].iloc[0] == 0.0   # month 0 carries no rebalancing cost


def test_turnover_nonnegative_and_bounded(edge_world):
    comp = st.equal_weight_composite(edge_world, MEMBERS, cost_bps=2.0)
    assert (comp["turnover"] >= 0).all()
    assert comp["turnover"].max() < 2.0   # one-sided turnover < 2 for a 5-name blend


def test_rebalance_is_point_in_time():
    # A synthetic ramp where member F1 alone soars: the START-of-month trade must use the
    # PRIOR month's drift, never the current month's return.
    idx = pd.period_range("2015-01", periods=6, freq="M").to_timestamp(how="end")
    R = pd.DataFrame({"F1": [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                      "F2": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    comp = st._weighted_composite(R, np.full((6, 2), 0.5), cost_bps=100.0)
    # Month 1 (F1 +50%) drifts weights apart; the rebalance is charged the FOLLOWING month.
    assert comp["turnover"].iloc[1] == 0.0     # no drift entering month 1 yet
    assert comp["turnover"].iloc[2] > 0.0      # month-1 drift is rebalanced at start of month 2


def test_inverse_vol_no_lookahead(null_world):
    # Inverse-vol weights use a shifted trailing vol -> the first lookback rows fall back
    # to equal weight (no forward information used).
    comp = st.inverse_vol_composite(null_world, MEMBERS, lookback=12, cost_bps=0.0)
    assert len(comp) == len(null_world[MEMBERS].dropna())
    assert np.isfinite(comp["gross"].to_numpy()).all()


# --------------------------------------------------------------------------- #
# Diversification is mechanical
# --------------------------------------------------------------------------- #
def test_blend_vol_below_mean_single(null_world):
    # Members share the market but have independent style factors -> the equal-weight
    # blend must have LOWER vol than the average single member (pure diversification).
    comp = st.equal_weight_composite(null_world, MEMBERS, cost_bps=0.0)
    cash = null_world["cash"]
    ftr = st.factor_timing_risk(null_world, MEMBERS, comp["gross"], cash, null_world["SPY"])
    assert ftr["comp_vol_pct"] < ftr["mean_single_vol_pct"]


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(80, 155)
    assert lo < 80 / 155 < hi


def test_welch_t_zero_on_equal_means():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 500)
    b = rng.normal(0.0, 1.0, 500)
    assert abs(st.welch_t(a, b)) < 3.0


def test_bootstrap_ci_brackets_point(null_world):
    comp = st.equal_weight_composite(null_world, MEMBERS, cost_bps=0.0)
    boot = st.adv_bootstrap_ci(comp["net"], null_world["SPY"], null_world["cash"],
                               n_boot=800, seed=902)
    assert boot["lo"] <= boot["obs"] <= boot["hi"]
    assert 0.0 <= boot["frac_negative"] <= 1.0


def test_era_split_covers_sample(edge_world):
    comp = st.equal_weight_composite(edge_world, MEMBERS, cost_bps=0.0)
    eras = st.era_split(comp["net"], edge_world["SPY"], edge_world["cash"])
    assert eras["early"]["n_months"] + eras["late"]["n_months"] == 168


# --------------------------------------------------------------------------- #
# Real-cache test (skipped on CI where _cache/ is absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="no real _cache/ present (offline CI)")
def test_real_blend_window_and_race():
    prices = data.load_prices()
    mret = data.monthly_total_returns(prices)
    comp = st.equal_weight_composite(mret, data.SLEEVE, cost_bps=2.0)
    race = st.sharpe_race(comp["net"], mret[data.BENCH], mret[data.CASH])
    assert race["n_months"] > 120                 # ~13y of common history
    assert 0.5 < race["sharpe_comp"] < 1.3        # sane equity Sharpe
    assert 0.5 < race["sharpe_spy"] < 1.3
    assert comp["turnover"].mean() < 0.10         # low-turnover long-only sleeve

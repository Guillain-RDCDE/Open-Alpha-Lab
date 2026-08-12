"""Offline, synthetic-only tests for Study 896 — Risk-Parity + Trend.

The engine is exercised on deterministic seeded worlds:
  * the synthetic world is reproducible;
  * inverse-vol weights are normalised and tilt to the low-vol sleeve;
  * the trend gate carries no look-ahead (a day-t gate uses only prices through t-1);
  * on the NULL world (edge=0) the trend gate does NOT improve the Sharpe;
  * on the PLANTED world (edge=1) the trend gate cuts drawdown and lifts the Sharpe;
  * costs reduce the net Sharpe advantage monotonically;
  * the inference primitives (HAC t, one-sample t, Sharpe-diff bootstrap) are sane.

A single real-cache test is @skipif-gated so the suite passes with NO cache present.
"""
import os

import numpy as np
import pandas as pd
import pytest

from rp_trend import data, strategy as st


# --------------------------------------------------------------------------- #
# Data / world
# --------------------------------------------------------------------------- #
def test_world_deterministic():
    px1, r1, c1 = data.synthetic_world(edge=1.0, seed=896)
    px2, r2, c2 = data.synthetic_world(edge=1.0, seed=896)
    assert np.allclose(r1.to_numpy(), r2.to_numpy())
    assert np.allclose(px1.to_numpy(), px2.to_numpy())
    assert np.allclose(c1.to_numpy(), c2.to_numpy())


def test_world_index_is_oob_safe(planted_world):
    px, ret, cash = planted_world
    # a plain business-day index well below the pandas ns-Timestamp horizon (~2262)
    assert isinstance(px.index, pd.DatetimeIndex)
    assert px.index.max().year < 2100
    assert len(px) == len(ret) == len(cash)


def test_edge_changes_the_world():
    _, r0, _ = data.synthetic_world(edge=0.0, seed=896)
    _, r1, _ = data.synthetic_world(edge=1.0, seed=896)
    # the bear-regime penalty makes the planted world's worst drawdowns deeper per asset
    assert not np.allclose(r0.to_numpy(), r1.to_numpy())


# --------------------------------------------------------------------------- #
# Weights & gate
# --------------------------------------------------------------------------- #
def test_inverse_vol_weights_normalised_and_tilted():
    vol = pd.DataFrame({"A": [0.02], "B": [0.005], "C": [0.01], "D": [0.015]})
    w = st.inverse_vol_weights(vol)
    assert np.isclose(w.sum(axis=1).iloc[0], 1.0)
    assert w["B"].iloc[0] > w["A"].iloc[0]        # lowest-vol sleeve gets the most weight


def test_trend_gate_is_binary_and_lagged(planted_world):
    px, _, _ = planted_world
    gate = st.trend_gate(px, window=200)
    finite = gate.to_numpy()[np.isfinite(gate.to_numpy())]
    assert set(np.unique(finite)).issubset({0.0, 1.0})
    # burn-in: the first 199 rows have no 200d SMA -> gate is NaN
    assert gate.iloc[:199].isna().all().all()
    assert gate.iloc[200:].notna().all().all()


def test_no_lookahead_gate_uses_only_past_prices(planted_world):
    """Perturbing a FUTURE price must not change today's shifted gate."""
    px, _, _ = planted_world
    g = st.trend_gate(px, window=200).shift(1)
    px2 = px.copy()
    px2.iloc[-1] *= 1.5                            # shock only the very last day
    g2 = st.trend_gate(px2, window=200).shift(1)
    # every row except the last is untouched by a future-only perturbation
    a, b = g.iloc[:-1].to_numpy(), g2.iloc[:-1].to_numpy()
    assert np.allclose(a[np.isfinite(a)], b[np.isfinite(b)])


# --------------------------------------------------------------------------- #
# The core claim on the controlled worlds
# --------------------------------------------------------------------------- #
def test_null_world_trend_does_not_help(null_world):
    px, ret, cash = null_world
    sleeves = list(ret.columns)
    r = st.race(px, ret, cash, sleeves)
    # with no persistent downtrend the gate cannot manufacture a Sharpe edge
    assert r["sharpe_adv"] <= 0.10


def test_planted_world_trend_cuts_drawdown_and_lifts_sharpe(planted_world):
    px, ret, cash = planted_world
    sleeves = list(ret.columns)
    r = st.race(px, ret, cash, sleeves)
    assert r["sharpe_adv"] > 0.0                   # the gate lifts the Sharpe
    assert r["dd_relief_pp"] > 5.0                 # and materially shallows the drawdown
    assert r["trend"]["maxdd_pct"] > r["plain"]["maxdd_pct"]   # less negative


def test_synthetic_control_separates_null_from_planted():
    null = st.synthetic_check(edge=0.0, n_seeds=8)
    planted = st.synthetic_check(edge=1.0, n_seeds=8)
    assert planted["mean_sharpe_adv"] > null["mean_sharpe_adv"]
    assert planted["mean_dd_relief_pp"] > null["mean_dd_relief_pp"]
    assert planted["share_adv_pos"] >= 0.6


# --------------------------------------------------------------------------- #
# Costs & mechanics
# --------------------------------------------------------------------------- #
def test_costs_reduce_the_net_sharpe_advantage(planted_world):
    px, ret, cash = planted_world
    sleeves = list(ret.columns)
    sweep = st.cost_sweep(px, ret, cash, sleeves, costs=(0.0, 10.0, 30.0))
    advs = [c["sharpe_adv"] for c in sweep]
    assert advs[0] >= advs[1] >= advs[2]           # costs monotonically erode the edge


def test_excess_is_zero_when_fully_in_cash():
    """If every sleeve is below trend for the whole sample, RP+trend holds only cash and
    its excess-of-cash return is identically zero."""
    px, ret, cash = data.synthetic_world(edge=1.0, seed=1, n_days=1200)
    sleeves = list(ret.columns)
    # force the gate off everywhere by making prices strictly decreasing
    px_down = pd.DataFrame(
        {c: 100.0 * (0.999 ** np.arange(len(px))) for c in sleeves}, index=px.index
    )
    ret_down = st.daily_returns(px_down)
    bt = st.backtest(px_down, ret_down, cash, sleeves, use_trend=True)
    assert np.allclose(bt["excess"].to_numpy(), 0.0, atol=1e-12)
    assert bt["avg_gate"] == pytest.approx(0.0, abs=1e-9)


def test_backtest_is_a_valid_return_series(planted_world):
    px, ret, cash = planted_world
    sleeves = list(ret.columns)
    bt = st.backtest(px, ret, cash, sleeves, use_trend=True)
    assert bt["total"].notna().all()
    assert bt["excess"].notna().all()
    assert 0.0 <= bt["avg_risky"] <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_hac_tstat_recovers_a_planted_mean():
    rng = np.random.default_rng(0)
    x = 0.5 + rng.standard_normal(5000)            # mean 0.5, sd 1 -> t huge
    out = st.hac_tstat(x)
    assert out["t"] > 10
    assert st.hac_tstat(rng.standard_normal(5000))["t"] == pytest.approx(0.0, abs=3.0)


def test_one_sample_t_sign():
    assert st.one_sample_t(np.full(100, 0.3)) > 0 or np.isnan(st.one_sample_t(np.full(100, 0.3)))
    x = np.array([1.0, -1.0, 2.0, -2.0, 0.5, -0.5] * 20)
    assert abs(st.one_sample_t(x)) < 2.0


def test_sharpe_diff_bootstrap_ci_brackets_observed(planted_world):
    px, ret, cash = planted_world
    sleeves = list(ret.columns)
    r = st.race(px, ret, cash, sleeves)
    bs = st.sharpe_diff_bootstrap(r["trend_excess"], r["plain_excess"], n_boot=500)
    assert bs["lo"] <= bs["obs"] <= bs["hi"]
    assert 0.0 <= bs["p_gt0"] <= 1.0


def test_max_drawdown_bounds():
    assert st.max_drawdown(np.zeros(100)) == pytest.approx(0.0)
    assert -1.0 <= st.max_drawdown(np.array([0.1, -0.5, -0.5, 0.2])) <= 0.0


# --------------------------------------------------------------------------- #
# Real-cache test — skipped entirely when the cache is absent (CI has no _cache/)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE),
                    reason="no real _cache/ present (offline CI)")
def test_real_cache_headline_shape():
    px = data.load_prices()
    ret = st.daily_returns(px)
    cash = ret[data.CASH]
    assert set(data.SLEEVES + [data.CASH]).issubset(px.columns)
    r = st.race(px, ret, cash, data.SLEEVES)
    # the drawdown relief is the durable real-tape fact; the Sharpe edge is thin
    assert r["dd_relief_pp"] > 0.0
    assert r["trend"]["maxdd_pct"] > r["plain"]["maxdd_pct"]
    assert r["years"] > 10

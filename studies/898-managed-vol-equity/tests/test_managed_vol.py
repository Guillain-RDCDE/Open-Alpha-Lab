"""Offline, fixed-seed tests for the managed-vol machinery.

The synthetic world is deterministic; the vol thermostat holds vol roughly constant; a
planted leverage-effect world is recovered (timing alpha lights up) while the risk-priced
null earns nothing; the rebalance is point-in-time (one shift, no look-ahead); a *constant*
scale leaves the excess-of-cash Sharpe unchanged (so any Sharpe gap is genuine timing);
costs reduce the net; the inference primitives behave. All offline, no real cache.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from managed_vol import data, strategy as st  # noqa: E402

CACHE = data.CACHE


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_world_deterministic():
    r1, s1 = data.synthetic_world(n_days=1500, disconnect=2.0, seed=898)
    r2, s2 = data.synthetic_world(n_days=1500, disconnect=2.0, seed=898)
    assert np.allclose(r1, r2) and np.allclose(s1, s2)


def test_worlds_share_vol_path_but_differ_in_mean():
    # same seed => identical vol path across disconnect settings (design invariant)
    r0, s0 = data.synthetic_world(n_days=2000, disconnect=0.0, seed=898)
    r2, s2 = data.synthetic_world(n_days=2000, disconnect=2.0, seed=898)
    assert np.allclose(s0, s2)
    assert not np.allclose(r0, r2)


# --------------------------------------------------------------------------- #
# The rule is point-in-time (one clean lag, no look-ahead)
# --------------------------------------------------------------------------- #
def test_weights_are_point_in_time():
    spy = pd.Series(np.linspace(-0.02, 0.02, 60),
                    index=pd.bdate_range("2020-01-01", periods=60))
    rv = st.realized_vol(spy, window=5)
    w = st.vol_target_weights(spy, target=0.12, window=5)
    # w on row t must equal target/RV known at t-1 (a pure shift of the raw signal)
    raw = (0.12 / rv).clip(upper=st.CAP)
    assert np.allclose(w.iloc[10], raw.iloc[9], equal_nan=True)
    # the first `window` rows are burn-in NaN
    assert w.iloc[:5].isna().all()


def test_no_lookahead_future_shock_does_not_move_todays_weight():
    rng = np.random.default_rng(0)
    spy = pd.Series(rng.normal(0, 0.01, 200),
                    index=pd.bdate_range("2020-01-01", periods=200))
    w0 = st.vol_target_weights(spy, window=21)
    spy2 = spy.copy()
    spy2.iloc[150] = 0.5                       # a huge future shock
    w1 = st.vol_target_weights(spy2, window=21)
    assert np.allclose(w0.iloc[:150].to_numpy(), w1.iloc[:150].to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------- #
# The thermostat actually targets constant vol
# --------------------------------------------------------------------------- #
def test_thermostat_holds_vol_near_target(null_world):
    ex, _ = null_world
    vt = st.vol_tracking(ex, target=0.12, window=21)
    # median realised vol of the managed book lands within a couple of vol-points of target
    assert abs(vt["median_roll_vol_pct"] - 12.0) < 3.5
    # and its vol is tamer than the raw asset's upper tail
    assert vt["p90_pct"] < vt["bh_p90_pct"] + 1e-9


# --------------------------------------------------------------------------- #
# Positive control: planted leverage effect recovered; risk-priced null flat
# --------------------------------------------------------------------------- #
def test_planted_world_lights_up(planted_world):
    ex, _ = planted_world
    r = st.race(ex)
    assert r["t_alpha"] > 3.0
    assert r["alpha_ann_pct"] > 0
    assert r["sharpe_gap"] > 0


def test_null_world_no_timing_alpha():
    # averaged over seeds (desk rule: no single-seed baselines)
    res = st.synthetic_check(0.0, n_seeds=15, n_days=3000)
    assert abs(res["mean_t"]) < 1.5           # centred on zero, unbiased
    assert res["mean_alpha_ann_pct"] < 2.0    # no material planted edge


def test_planted_beats_null_on_average():
    null = st.synthetic_check(0.0, n_seeds=12, n_days=3000)
    planted = st.synthetic_check(2.0, n_seeds=12, n_days=3000)
    assert planted["mean_t"] > null["mean_t"] + 2.0
    assert planted["share_t_ge_2"] > 0.5


# --------------------------------------------------------------------------- #
# Excess-of-cash & the leverage-timing decomposition
# --------------------------------------------------------------------------- #
def test_constant_scale_preserves_sharpe(null_world):
    # A *constant* weight is pure leverage, no timing: excess-of-cash Sharpe must equal
    # buy-and-hold's, so any Sharpe gap of the real overlay is genuine timing.
    ex, _ = null_world
    const_w = pd.Series(0.7, index=ex.index)
    ov = st.run_overlay(ex, weights=const_w)
    sh_const = st.perf(ov["strat"])["sharpe"]
    sh_bh = st.perf(ov["bh"])["sharpe"]
    assert abs(sh_const - sh_bh) < 1e-9


def test_decomposition_adds_up(planted_world):
    ex, _ = planted_world
    ov = st.run_overlay(ex)
    r = st.race(ex)
    mean_managed_bps = float(ov["strat"].mean()) * 1e4
    assert abs((r["exposure_bps"] + r["timing_bps"]) - mean_managed_bps) < 1e-6


# --------------------------------------------------------------------------- #
# Costs reduce the net
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(planted_world):
    ex, _ = planted_world
    gross = st.race(ex, cost_bps=0.0, borrow_spread_ann=0.0)
    net = st.race(ex, cost_bps=5.0, borrow_spread_ann=0.02)
    assert net["strat"]["cagr_pct"] < gross["strat"]["cagr_pct"]
    assert net["alpha_ann_pct"] < gross["alpha_ann_pct"]


def test_entry_trade_excluded():
    # first Δw is zeroed (both legs pay the entry) — turnover finite, no NaN blow-up
    spy = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300),
                    index=pd.bdate_range("2020-01-01", periods=300))
    ex = pd.DataFrame({"spy": spy, "cash": 0.0, "spy_excess": spy})
    ov = st.run_overlay(ex, cost_bps=10.0)
    assert np.isfinite(ov["turnover_ann"]) and ov["turnover_ann"] >= 0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_hac_alpha_recovers_known_beta():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, 8000)
    y = 0.0001 + 0.5 * x + rng.normal(0, 0.004, 8000)   # alpha 1bp/day, beta 0.5
    reg = st.hac_alpha(y, x)
    assert abs(reg["beta"] - 0.5) < 0.05
    assert abs(reg["alpha_bps"] - 1.0) < 0.6


def test_max_drawdown_sign_and_bounds():
    r = pd.Series([0.1, -0.5, 0.2, -0.3])
    dd = st.max_drawdown(r)
    assert -1.0 <= dd <= 0.0


def test_bootstrap_ci_brackets_point(planted_world):
    ex, _ = planted_world
    ov = st.run_overlay(ex)
    bs = st.sharpe_gap_bootstrap(ov["strat"].values, ov["bh"].values, n_boot=400)
    assert bs["ci_low"] <= bs["gap"] <= bs["ci_high"]


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped when the git-ignored cache is absent (CI)
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


@pytest.mark.skipif(not os.path.exists(CACHE), reason="real SPY/BIL cache absent (offline CI)")
def test_real_cache_headline_sane():
    ex = data.excess_returns()
    assert len(ex) > 3000
    r = st.race(ex)
    # managed book cuts drawdown and holds lower vol than buy-and-hold
    assert r["strat"]["maxdd_pct"] > r["bh"]["maxdd_pct"]      # less negative
    assert r["strat"]["vol_ann_pct"] < r["bh"]["vol_ann_pct"]
    # thermostat lands near the 12% target
    assert 9.0 < r["strat"]["vol_ann_pct"] < 16.0

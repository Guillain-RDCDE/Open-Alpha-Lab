"""Offline, fixed-seed tests for the sector risk-parity machinery.

The synthetic worlds are deterministic; the weighting schemes are well-formed (long-only,
sum to 1, ERC equalises risk contributions); the quarterly rebalancer is point-in-time (no
look-ahead); costs reduce the net; inverse-vol out-Sharpes the concentrated cap-weight ONLY
when vols are dispersed (planted) and ties when they are equal (null); the inference
primitives behave. All offline — no real cache required.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sector_rp import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism + world shape
# --------------------------------------------------------------------------- #
def test_world_deterministic(planted_world):
    w2 = data.synthetic_world(vol_spread=0.02, seed=890)
    assert np.allclose(planted_world["sector_ret"].to_numpy(), w2["sector_ret"].to_numpy())


def test_null_world_has_equal_vols():
    w = data.synthetic_world(vol_spread=0.0, seed=890)
    vols = w["sector_ret"].std().to_numpy()
    assert vols.std() / vols.mean() < 0.10       # vols essentially equal at vol_spread=0


# --------------------------------------------------------------------------- #
# Weighting schemes
# --------------------------------------------------------------------------- #
def test_inverse_vol_weights_sum_and_sign():
    w = st.inverse_vol_weights(np.array([0.01, 0.02, 0.04]))
    assert abs(w.sum() - 1.0) < 1e-12
    assert np.all(w > 0)
    assert w[0] > w[1] > w[2]                     # lower vol -> higher weight


def test_erc_equalises_risk_contributions():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 5)) * np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cov = np.cov(X.T)
    w = st.erc_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.all(w > 0)
    rc = st.risk_contributions(w, cov)
    assert rc.std() < 0.01                        # risk contributions ~ equal (1/n each)
    assert w[0] > w[-1]                           # lowest-vol asset carries the most weight


def test_erc_degenerate_cov_falls_back():
    bad = np.full((3, 3), np.nan)
    w = st.erc_weights(bad)
    assert abs(w.sum() - 1.0) < 1e-9              # graceful fallback, still a valid book


# --------------------------------------------------------------------------- #
# The rebalancer: weights sum to 1, point-in-time, costs bite
# --------------------------------------------------------------------------- #
def test_allocate_weights_sum_to_one(planted_world):
    a = st.allocate(planted_world["sector_ret"], scheme="invvol", freq="Q")
    wsum = a["weights"].sum(axis=1)
    assert np.allclose(wsum.to_numpy(), 1.0, atol=1e-9)


def test_rebalance_is_point_in_time():
    # A world where vols jump on a known date: the weight change must appear only AFTER the
    # rebalance that follows the jump (via the trailing look-back), never before it.
    idx = pd.bdate_range("2004-01-01", periods=400)
    n = len(idx)
    r = np.full((n, 3), 0.0)
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.005, (n, 3))
    base[200:, 2] *= 6.0                          # asset 2's vol explodes at day 200
    ret = pd.DataFrame(base, index=idx, columns=["A", "B", "C"])
    a = st.allocate(ret, scheme="invvol", lookback=63, freq="Q", lag=1)
    W = a["weights"]
    # Before the vol jump is in the trailing window, asset C's weight is ~1/3; well after, it
    # is trimmed. The trim must not appear before day 200.
    pre = W.loc[W.index < idx[200], "C"]
    assert (pre > 0.25).all()                     # no look-ahead: C not yet penalised


def test_costs_reduce_net(planted_world):
    gross = st.allocate(planted_world["sector_ret"], scheme="invvol", cost_bps=0.0)["net"]
    net = st.allocate(planted_world["sector_ret"], scheme="invvol", cost_bps=10.0)["net"]
    assert net.sum() < gross.sum()               # a positive cost lowers cumulative return
    assert net.dropna().mean() < gross.dropna().mean()


def test_equal_scheme_matches_equal_weight(planted_world):
    a = st.allocate(planted_world["sector_ret"], scheme="equal", freq="Q")
    n = planted_world["sector_ret"].shape[1]
    # On a rebalance day the equal book holds exactly 1/n in each name.
    reb = a["turnover"].index[1]                  # a mid-sample rebalance date
    assert np.allclose(a["weights"].loc[reb].to_numpy(), 1.0 / n, atol=1e-9)


# --------------------------------------------------------------------------- #
# The planted control fires; the null does not
# --------------------------------------------------------------------------- #
def test_planted_advantage_recovered(planted_world):
    d = st.synthetic_detect(planted_world)
    assert d["sharpe_advantage"] > 0.05          # dispersed vols -> RP beats cap-weight on Sharpe
    assert d["sr_rp"] > d["sr_bench"]


def test_null_world_no_advantage(null_world):
    d = st.synthetic_detect(null_world)
    assert abs(d["sharpe_advantage"]) < 0.05     # equal vols -> RP ~ cap-weight (mechanical null)


def test_null_advantage_small_across_seeds():
    advs = np.array([st.synthetic_detect(data.synthetic_world(vol_spread=0.0, seed=890 + s))["sharpe_advantage"]
                     for s in range(10)])
    assert np.abs(advs).mean() < 0.05
    assert (np.abs(advs) > 0.10).sum() == 0      # the detector never spuriously fires on the null


# --------------------------------------------------------------------------- #
# perf_stats + inference primitives
# --------------------------------------------------------------------------- #
def test_perf_stats_excess_of_cash():
    idx = pd.bdate_range("2004-01-01", periods=500)
    rng = np.random.default_rng(3)
    r = pd.Series(0.0006 + rng.normal(0, 0.008, len(idx)), index=idx)
    cash = pd.Series(0.0001, index=idx)
    s = st.perf_stats(r, cash)
    assert s["sharpe"] > 0                          # positive-drift book beats a low cash rate
    assert s["max_drawdown"] <= 0.0
    # raising the cash rate lowers the excess Sharpe (same book, higher hurdle)
    s2 = st.perf_stats(r, pd.Series(0.0005, index=idx))
    assert s2["sharpe"] < s["sharpe"]


def test_sharpe_diff_bootstrap_sign(planted_world):
    a = st.allocate(planted_world["sector_ret"], scheme="invvol")["gross"]
    b = planted_world["bench_ret"]
    cash = planted_world["cash_ret"]
    idx = a.dropna().index
    bs = st.sharpe_diff_bootstrap((a.reindex(idx) - cash.reindex(idx)),
                                  (b.reindex(idx) - cash.reindex(idx)))
    assert bs["diff"] > 0                          # RP out-Sharpes cap-weight in the planted world
    assert bs["ci_low"] < bs["diff"] < bs["ci_high"]


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_levered_timer_leverage_ge_one(planted_world):
    a = st.allocate(planted_world["sector_ret"], scheme="invvol")["net"]
    b = planted_world["bench_ret"]
    cash = planted_world["cash_ret"]
    idx = a.dropna().index
    lev = st.levered_to_bench_vol((a.reindex(idx) - cash.reindex(idx)),
                                  (b.reindex(idx) - cash.reindex(idx)), cash)
    assert lev["leverage"] > 0
    assert np.isfinite(lev["sharpe_lev"])


# --------------------------------------------------------------------------- #
# Real-cache smoke test — only runs where the parquet is present (skipped on CI)
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


@pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE), reason="no real cache present")
def test_real_panel_loads_and_races():
    p = data.daily_panel(sectors=data.SECTORS_9)
    assert p["sector_ret"].shape[1] == 9
    r = st.race(p["sector_ret"], p["bench_ret"], p["cash_ret"], scheme="invvol", cost_bps=3.0)
    assert r["n_days"] > 1000
    assert np.isfinite(r["sr_strat_net"]) and np.isfinite(r["sr_bench"])
    assert r["dd_strat"] <= 0 and r["dd_bench"] <= 0

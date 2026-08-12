"""Offline, fixed-seed tests for the EM-local FX-hedge machinery.

The synthetic world is deterministic; the UUP-overlay hedge recovers a PLANTED local-rate
carry (positive hedged excess with a strong HAC t) and stays silent on the null; the hedge
strips FX (the hedge ratio is negative = a long-USD overlay, and hedging cuts variance);
the walk-forward hedge has no look-ahead; costs reduce the net; the inference primitives
behave. All offline, synthetic-only — no real cache required.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from em_hedged import data, strategy as st  # noqa: E402


def test_world_deterministic(planted_world):
    p2 = data.synthetic_world(carry_annual=0.06, seed=906, n_months=220)
    assert np.allclose(planted_world.to_numpy(), p2.to_numpy())


def test_index_is_safe_periodindex_range(planted_world):
    # Monthly index must stay far inside the pandas ns-Timestamp horizon (~year 2262).
    assert planted_world.index.max().year < 2100
    assert planted_world.index.is_monotonic_increasing


def test_hedge_ratio_is_negative_long_dollar(planted_world):
    m = planted_world  # already a monthly-returns frame
    b = st.hedge_ratio(st.excess(m, "EMLC"), st.excess(m, "UUP"))
    # dollar up => EM-local down, so the variance-min hedge is a LONG-UUP overlay (b < 0).
    assert b < -0.5


def test_hedging_strips_fx_variance(planted_world):
    m = planted_world  # already a monthly-returns frame
    le = st.excess(m, "EMLC")
    he = st.hedged_series(m, "EMLC")
    # stripping the dollar-basket FX must lower the return variance.
    assert he.var() < le.var()


def test_planted_carry_recovered(planted_world):
    m = planted_world  # already a monthly-returns frame
    r = st.race(m)
    assert r["hedge_b"] < 0
    assert r["hedged_exc_ann_pct"] > 0          # the planted local carry surfaces
    assert r["t_hedged"] > 2.5                  # and it is significant on the planted world
    assert r["hedged_sharpe"] > r["local_sharpe"]  # hedging beats leaving the FX in


def test_null_world_no_carry(null_world):
    m = null_world  # already a monthly-returns frame
    r = st.race(m)
    assert abs(r["t_hedged"]) < 2.5             # nothing to harvest when carry == 0


def test_null_does_not_fire_across_seeds():
    fires = 0
    for s in range(12):
        w = data.synthetic_world(carry_annual=0.0, seed=906 + s, n_months=200)
        r = st.race(w)
        fires += int(abs(r["t_hedged"]) >= 2.0)
    assert fires <= 2                           # ~false-positive rate, not a biased detector


def test_walk_forward_hedge_no_lookahead(planted_world):
    m = planted_world  # already a monthly-returns frame
    wf = st.rolling_hedge_series(m, "EMLC", window=36, min_periods=24)
    # the walk-forward series starts only after the warm-up window (no early leakage).
    le = st.excess(m, "EMLC")
    assert wf.index.min() > le.index.min()
    assert len(wf) > 0
    # and it still recovers a positive planted carry (implementable, no in-sample b).
    assert wf.mean() > 0


def test_costs_reduce_net(planted_world):
    m = planted_world  # already a monthly-returns frame
    c = st.costed(m, local="EMLC")
    assert c["charge_ann_pct"] > 0
    assert c["net_hedged_ann_pct"] < c["gross_hedged_ann_pct"]


def test_era_split_runs(planted_world):
    m = planted_world  # already a monthly-returns frame
    es = st.era_split(m, split="2018-01-01")
    assert set(es) == {"early", "late"}
    assert es["early"]["n"] > 0 and es["late"]["n"] > 0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 3000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_t_sign():
    a = np.full(200, 0.01) + np.random.default_rng(1).normal(0, 1e-4, 200)
    b = np.full(200, 0.00) + np.random.default_rng(2).normal(0, 1e-4, 200)
    assert st.welch_t(a, b) > 5


def test_hac_ols_recovers_slope():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 500)
    y = 2.0 - 1.3 * x + rng.normal(0, 0.1, 500)
    reg = st.hac_ols(y, x)
    assert abs(reg["beta"] + 1.3) < 0.05
    assert abs(reg["alpha"] - 2.0) < 0.05


def test_max_drawdown_on_monotone():
    px = pd.Series(np.arange(1, 51, dtype=float),
                   index=pd.period_range("2015-01", periods=50, freq="M").to_timestamp())
    assert st.max_drawdown(px)["depth_pct"] == 0.0


def test_real_cache_race_if_present():
    """@skipif-style guard: only runs when the real parquet cache exists (absent on CI)."""
    if not data.have_real():
        import pytest
        pytest.skip("no real cache present (offline CI) — synthetic tests cover the machinery")
    m = st.monthly_returns(data.load_prices())
    r = st.race(m, local="EMLC")
    assert r["n"] > 120
    assert r["hedge_b"] < 0                       # long-dollar overlay on the real tape too
    assert r["hedged_sharpe"] > r["local_sharpe"]  # stripping FX helps on the real tape

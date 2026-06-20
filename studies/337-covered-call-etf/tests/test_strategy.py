"""The engine's invariants and the study's two spines, all on the deterministic synthetic
tape (never the network): (1) the income illusion — a fat distribution sitting on an eroding
NAV reads as ~100% return of capital, and the buy-write loses to the index with HAC t below
the bar; (2) the null world is a dead heat (no spread, no illusion, full capture)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covered_call_etf import strategy as st  # noqa: E402


# ---- performance primitives ------------------------------------------------
def test_cagr_of_constant_return():
    r = pd.Series([0.01] * 24)
    assert abs(st.cagr(r) - ((1.01 ** 12) - 1.0)) < 1e-9


def test_max_drawdown_is_nonpositive():
    r = pd.Series([0.05, -0.10, 0.02, -0.20, 0.03])
    assert st.max_drawdown(r) <= 0.0


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(st.sharpe(pd.Series([0.01] * 10)))


# ---- HAC / bootstrap inference ---------------------------------------------
def test_newey_west_recovers_mean():
    rng = np.random.default_rng(0)
    x = 0.002 + 0.01 * rng.standard_normal(400)
    mu, t = st.newey_west_t(x)
    assert abs(mu - x.mean()) < 1e-12
    assert t > 2.0  # a real positive mean clears the bar


def test_newey_west_zero_mean_is_insignificant():
    rng = np.random.default_rng(0)
    x = 0.01 * rng.standard_normal(400)  # mean ~0 (seed pinned so the null stays a null)
    _, t = st.newey_west_t(x)
    assert abs(t) < 2.0


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(2)
    x = 0.003 + 0.01 * rng.standard_normal(300)
    lo, hi = st.block_bootstrap_ci(x, block=6, seed=42)
    assert lo < x.mean() < hi


# ---- the income illusion (spine #1) ----------------------------------------
def test_income_illusion_is_all_return_of_capital(capped):
    """A fat distribution on an eroding NAV → return-of-capital share = 1 (the illusion)."""
    panel, _ = capped
    ill = st.income_illusion(panel["bw_price"], panel["bw_dist"])
    assert ill["dist_yield"] > 0.05          # marketed as high "income"
    assert ill["price_cagr"] < 0.0           # but the NAV is eroding
    assert ill["return_of_capital_share"] > 0.95   # so the income is your own capital


def test_buywrite_loses_to_index_with_robust_t(capped):
    """Spine #1 inference: the buy-write trails the index, HAC t clears the bar (negative)."""
    panel, _ = capped
    r = st.race(panel["bw_tr"], panel["index_tr"])
    assert r["spread_ann_pct"] < 0.0
    assert r["spread_t"] < -2.0
    assert r["spread_ci_hi_bps"] < 0.0       # the whole CI is below zero


def test_capped_upside_signature(capped):
    """The covered-call tell: keeps less of the up months than the down months (on the price leg)."""
    panel, _ = capped
    cap = st.capture(panel["bw_price"], panel["index_tr"])
    assert cap["up_capture"] < 0.75          # upside is shaved
    assert cap["down_capture"] > 0.95        # downside passes through
    assert cap["asymmetry"] < 0.0


# ---- the null world (spine #2) ---------------------------------------------
def test_null_world_is_a_dead_heat(null):
    """With no cap and no premium there is no spread, no illusion, full capture."""
    panel, _ = null
    r = st.race(panel["bw_tr"], panel["index_tr"])
    # identical series → spread is exactly zero (t is nan, CI degenerate)
    assert abs(r["spread_ann_pct"]) < 1e-9
    cap = st.capture(panel["bw_price"], panel["index_tr"])
    assert abs(cap["up_capture"] - 1.0) < 1e-9
    assert abs(cap["down_capture"] - 1.0) < 1e-9


# ---- replicator invariants & costs -----------------------------------------
def test_replicate_identity_and_costs(capped):
    panel, _ = capped
    rep = st.replicate_buywrite(panel["index_tr"], cap=0.018, premium=0.0065, cost_bps=0.0)
    assert np.allclose(rep["total_gross"], rep["price"] + rep["dist"])
    rep2 = st.replicate_buywrite(panel["index_tr"], cap=0.018, premium=0.0065, cost_bps=5.0)
    assert np.allclose(rep2["total_net"], rep2["total_gross"] - 5.0e-4)


def test_cost_monotonically_lowers_net(capped):
    panel, _ = capped
    means = [
        st.replicate_buywrite(panel["index_tr"], cap=0.018, premium=0.0065,
                              cost_bps=c)["total_net"].mean()
        for c in (0.0, 2.0, 5.0)
    ]
    assert means[0] > means[1] > means[2]


def test_execution_lag_changes_the_path_but_keeps_identity(capped):
    """The documented one-month execution-lag knob shifts when the cap binds, identity holds."""
    panel, _ = capped
    rep0 = st.replicate_buywrite(panel["index_tr"], cap=0.018, premium=0.0065, lag=0)
    rep1 = st.replicate_buywrite(panel["index_tr"], cap=0.018, premium=0.0065, lag=1)
    assert not np.allclose(rep0["price"].to_numpy(), rep1["price"].to_numpy())
    assert np.allclose(rep1["total_gross"], rep1["price"] + rep1["dist"])

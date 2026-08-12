"""Offline, fixed-seed tests for the IPO-anchoring machinery.

The synthetic panel is deterministic; the Fama-MacBeth anchoring-slope detector recovers a
planted pull (negative slope) and stays silent on the null; the below-offer basket split is a
genuine contrast; the placebo centres on zero; the inference primitives behave; the timer costs
bite; the curated table is well-formed. Everything here is synthetic-only and network-free —
the one real-tape check is skipped when the git-ignored cache is absent (i.e. on CI).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest  # noqa: E402

from ipo_anchor import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world — deterministic & unbiased machinery
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.3, seed=874)
    assert np.allclose(edge_world["gap"], p2["gap"], equal_nan=True)
    assert np.allclose(edge_world["fwd_abn"], p2["fwd_abn"], equal_nan=True)


def test_panel_shape(edge_world):
    M = len(edge_world["months"])
    N = len(edge_world["names"])
    assert edge_world["gap"].shape == (M, N)
    assert edge_world["fwd_abn"].shape == (M, N)
    assert edge_world["below"].shape == (M, N)
    # forward return's last month is undefined (no next month)
    assert np.all(np.isnan(edge_world["fwd_abn"][-1]))
    # a PeriodIndex kept as periods (timestamp-overflow trap avoided)
    assert isinstance(edge_world["months"], pd.PeriodIndex)


def test_planted_pull_recovered(edge_world):
    a = st.anchoring_stats(edge_world)
    assert a["mean_slope"] < 0            # anchoring pull => negative slope
    assert a["t_nw"] < -3.0               # and it lights up


def test_null_world_no_anchor(null_world):
    a = st.anchoring_stats(null_world)
    assert abs(a["t_nw"]) < 2.5           # gap predicts nothing in the null


def test_synthetic_control_separates():
    null = st.synthetic_control(0.0, n_seeds=20)
    planted = st.synthetic_control(0.3, n_seeds=20)
    assert null["reject_rate"] <= 0.10    # near-nominal false-positive rate
    assert planted["reject_rate"] >= 0.90  # planted pull almost always detected
    assert planted["mean_slope"] < null["mean_slope"]


def test_below_offer_split_is_contrast(edge_world):
    sp = st.below_offer_spreads(edge_world, min_side=3)
    assert len(sp) > 20
    # both baskets are populated every retained month (n >= 2*min_side)
    assert (sp["n"] >= 6).all()


def test_placebo_centres_on_zero(edge_world):
    pl = st.placebo_pvalue(edge_world, n_seeds=6, n_draws_per_seed=20)
    assert abs(pl["placebo_mean"]) < 5e-3         # permutation null ~ 0
    assert pl["p_left"] < 0.05                    # planted negative slope in the left tail


# --------------------------------------------------------------------------- #
# Point-in-time construction
# --------------------------------------------------------------------------- #
def test_forward_return_is_next_month(edge_world):
    # fwd_abn[t] must equal the realised next-month gap change in the synthetic world:
    # gap[t+1] - gap[t] == fwd_abn[t] wherever both are finite (pull-plus-noise identity).
    g = edge_world["gap"]; f = edge_world["fwd_abn"]
    dg = g[1:] - g[:-1]
    m = np.isfinite(dg) & np.isfinite(f[:-1])
    assert np.allclose(dg[m], f[:-1][m], atol=1e-9)


# --------------------------------------------------------------------------- #
# Costs & inference primitives
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(edge_world):
    sp = st.below_offer_spreads(edge_world)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_ann_pct=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=20.0, borrow_ann_pct=5.0)["net_bps"]
    assert net < gross


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_sign():
    a = np.full(50, 0.02); b = np.full(50, 0.05)
    a = a + np.random.default_rng(1).normal(0, 1e-4, 50)
    b = b + np.random.default_rng(2).normal(0, 1e-4, 50)
    assert st.welch_t(a, b) < 0          # a's mean below b's


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(30, 76)
    assert lo < 30 / 76 < hi


# --------------------------------------------------------------------------- #
# The curated anchor table is well-formed
# --------------------------------------------------------------------------- #
def test_curated_table_wellformed():
    tbl = data.ipo_table(include_direct=True)
    assert len(tbl) >= 40                            # honest-N floor
    assert (tbl["offer"] > 0).all()
    assert tbl.index.is_unique
    assert set(tbl["kind"].unique()) <= {"ipo", "direct"}
    ipo_only = data.ipo_table(include_direct=False)
    assert (ipo_only["kind"] == "ipo").all()
    assert len(ipo_only) < len(tbl)                  # direct listings actually dropped
    assert data.BENCH in data.tickers()              # benchmark always included


# --------------------------------------------------------------------------- #
# Real tape — only when the git-ignored cache is present (skipped on CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_panel_builds():
    px = data.load_prices()
    tbl = data.ipo_table(include_direct=True)
    P = st.build_panel(px, tbl, bench=data.BENCH, asof=data.AS_OF)
    cov = st.panel_coverage(P)
    assert cov["n_active_months"] > 30
    assert cov["n_obs"] > 500
    # as-of pinned: no month beyond the stamp
    assert P["months"].max() <= pd.Timestamp(data.AS_OF)

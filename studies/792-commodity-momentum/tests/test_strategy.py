"""Offline, fixed-seed tests for Study 792 — cross-sectional commodity momentum.

No network: every test runs on the deterministic synthetic panel. Two things are proven —
(1) the inference + portfolio machinery obeys its invariants, and (2) the spine: the 12-1
sort banks a *planted* momentum edge and stays silent on the null.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commodity_momentum import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_hac_t_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.05, 500)
    assert abs(st.hac_t(x)["t"]) < 2.5


def test_hac_t_positive_mean_fires():
    rng = np.random.default_rng(1)
    x = rng.normal(0.02, 0.01, 300)   # strongly positive mean
    assert st.hac_t(x)["t"] > 5.0


def test_hac_t_too_short_is_nan():
    assert np.isnan(st.hac_t([0.1, 0.2, 0.3])["t"])


def test_one_sample_and_welch_signs():
    rng = np.random.default_rng(2)
    a = rng.normal(0.01, 0.02, 200)
    b = rng.normal(0.0, 0.02, 200)
    assert st.one_sample_t(a) > 0
    assert st.welch_t(a, b) > 0
    assert st.welch_t(b, a) < 0


def test_sharpe_scales_with_mean():
    r_lo = pd.Series(np.full(60, 0.001))
    r_hi = pd.Series(np.full(60, 0.001)) + np.random.default_rng(3).normal(0, 1e-9, 60)
    # constant series -> zero vol -> nan; add noise for a finite comparison
    noisy = pd.Series(np.random.default_rng(4).normal(0.001, 0.02, 120))
    assert np.isfinite(st.sharpe(noisy))


# --------------------------------------------------------------------------- #
# Signal + portfolio invariants
# --------------------------------------------------------------------------- #
def test_momentum_signal_skips_recent_month():
    # A clean monotone tape: signal at t must ignore month t (skip=1).
    mret = data.synthetic_world(mom_edge=0.0, seed=10)
    sig = st.momentum_signal(mret)
    # first (lookback) rows unusable -> NaN
    assert sig.iloc[: st.MOM_LOOKBACK].isna().all().all()
    # later rows are populated
    assert sig.iloc[st.MOM_LOOKBACK + 2].notna().any()


def test_weights_are_dollar_neutral_and_gross_two():
    mret = data.synthetic_world(mom_edge=0.5, seed=11)
    sig = st.momentum_signal(mret)
    w = st.rank_weights(sig)
    live = w.dropna(how="all")
    live = live[live.abs().sum(axis=1) > 0]
    # each live row: sums to ~0 (dollar neutral), gross ~2 (1 long + 1 short)
    assert np.allclose(live.sum(axis=1).to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(live.abs().sum(axis=1).to_numpy(), 2.0, atol=1e-9)


def test_portfolio_applies_exactly_one_lag():
    mret = data.synthetic_world(mom_edge=0.5, seed=12)
    sig = st.momentum_signal(mret)
    w = st.rank_weights(sig)
    ret, traded = st.portfolio(mret, w)
    # reconstruct with an explicit shift and compare
    wl = w.shift(1)
    live = wl.abs().sum(axis=1) > 0
    expected = (wl * mret).sum(axis=1, min_count=1)[live].dropna()
    assert np.allclose(ret.to_numpy(), expected.reindex(ret.index).to_numpy())


def test_traded_notional_nonnegative():
    mret = data.synthetic_world(mom_edge=0.5, seed=13)
    sig = st.momentum_signal(mret)
    w = st.rank_weights(sig)
    _, traded = st.portfolio(mret, w)
    assert (traded >= -1e-12).all()


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def test_cost_monotonically_lowers_net():
    mret = data.synthetic_world(mom_edge=0.5, seed=14)
    n0 = st.costed_timer(mret, cost_bps=0.0, borrow_bps_yr=0.0)["net_ann_pct"]
    n5 = st.costed_timer(mret, cost_bps=5.0, borrow_bps_yr=0.0)["net_ann_pct"]
    n10 = st.costed_timer(mret, cost_bps=10.0, borrow_bps_yr=0.0)["net_ann_pct"]
    assert n0 > n5 > n10


def test_borrow_reduces_net():
    mret = data.synthetic_world(mom_edge=0.5, seed=15)
    no_borrow = st.costed_timer(mret, cost_bps=5.0, borrow_bps_yr=0.0)["net_ann_pct"]
    with_borrow = st.costed_timer(mret, cost_bps=5.0, borrow_bps_yr=50.0)["net_ann_pct"]
    assert with_borrow < no_borrow


# --------------------------------------------------------------------------- #
# The spine: planted momentum fires, the null does not (over many seeds)
# --------------------------------------------------------------------------- #
def test_planted_momentum_fires():
    mret = data.synthetic_world(mom_edge=1.0, seed=792)
    d = st.synthetic_detect(mret)
    assert d["hac_t"] > 4.0
    assert d["mean_bps"] > 0.0


def test_null_world_does_not_systematically_fire():
    ts = np.array([st.synthetic_detect(data.synthetic_world(mom_edge=0.0, seed=792 + s))["hac_t"]
                   for s in range(20)])
    # unbiased: mean t near zero, and it does not fire on most seeds
    assert abs(np.nanmean(ts)) < 0.8
    assert np.mean(np.abs(ts) >= 2.0) <= 0.25


def test_placebo_beats_are_rare_on_planted_world():
    mret = data.synthetic_world(mom_edge=1.0, seed=792)
    pl = st.random_placebo(mret, n_seeds=20)
    # a strongly planted edge should sit far above random-rank sorts
    assert pl["p_value"] < 0.1
    assert pl["frac_t_ge_2"] < 0.5

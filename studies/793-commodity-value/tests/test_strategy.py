"""Offline, fixed-seed tests for Study 793 — cross-sectional commodity value.

No network: every test runs on the deterministic synthetic price panel. Two things are
proven — (1) the inference + portfolio machinery obeys its invariants, and (2) the spine:
the 5-year value sort banks a *planted* long-horizon reversal edge and stays silent on the
null (random walk).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from commodity_value import data, strategy as st  # noqa: E402


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


def test_sharpe_is_finite_on_noisy_series():
    noisy = pd.Series(np.random.default_rng(4).normal(0.001, 0.02, 120))
    assert np.isfinite(st.sharpe(noisy))


# --------------------------------------------------------------------------- #
# Value signal + portfolio invariants
# --------------------------------------------------------------------------- #
def test_value_signal_sign_is_cheap_positive():
    # Deterministic construction: FALL is a monotone-declining price (cheap -> POSITIVE
    # value score); RISE is a monotone-rising price (expensive -> NEGATIVE score).
    n = 120
    idx = pd.period_range("2000-01", periods=n, freq="M").to_timestamp("M")
    prices = pd.DataFrame({
        "FALL": 100.0 * np.exp(-0.01 * np.arange(n)),   # falls ~1%/mo
        "RISE": 100.0 * np.exp(+0.01 * np.arange(n)),   # rises ~1%/mo
    }, index=idx)
    sig = st.value_signal(prices)
    last = sig.dropna(how="all").iloc[-1]
    assert last["FALL"] > 0      # fallen commodity reads cheap (positive value)
    assert last["RISE"] < 0      # risen commodity reads expensive (negative value)


def test_value_signal_requires_history():
    prices = data.synthetic_world(val_edge=0.0, seed=10)
    sig = st.value_signal(prices)
    # the first LOOKBACK months cannot carry a 5-year-ago reference -> NaN
    assert sig.iloc[: st.VAL_LOOKBACK - st.VAL_WINDOW].isna().all().all()
    # later rows are populated
    assert sig.iloc[st.VAL_LOOKBACK + 20].notna().any()


def test_weights_are_dollar_neutral_and_gross_two():
    prices = data.synthetic_world(val_edge=0.05, seed=11)
    sig = st.value_signal(prices)
    w = st.rank_weights(sig)
    live = w.dropna(how="all")
    live = live[live.abs().sum(axis=1) > 0]
    assert np.allclose(live.sum(axis=1).to_numpy(), 0.0, atol=1e-9)
    assert np.allclose(live.abs().sum(axis=1).to_numpy(), 2.0, atol=1e-9)


def test_portfolio_applies_exactly_one_lag():
    prices = data.synthetic_world(val_edge=0.05, seed=12)
    mret = prices.pct_change()
    sig = st.value_signal(prices)
    w = st.rank_weights(sig)
    ret, traded = st.portfolio(mret, w)
    wl = w.reindex(mret.index).shift(1)
    live = wl.abs().sum(axis=1) > 0
    expected = (wl * mret).sum(axis=1, min_count=1)[live].dropna()
    assert np.allclose(ret.to_numpy(), expected.reindex(ret.index).to_numpy())


def test_traded_notional_nonnegative():
    prices = data.synthetic_world(val_edge=0.05, seed=13)
    mret = prices.pct_change()
    sig = st.value_signal(prices)
    w = st.rank_weights(sig)
    _, traded = st.portfolio(mret, w)
    assert (traded >= -1e-12).all()


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def test_cost_monotonically_lowers_net():
    prices = data.synthetic_world(val_edge=0.05, seed=14)
    mret = prices.pct_change()
    n0 = st.costed_timer(prices, mret, cost_bps=0.0, borrow_bps_yr=0.0)["net_ann_pct"]
    n5 = st.costed_timer(prices, mret, cost_bps=5.0, borrow_bps_yr=0.0)["net_ann_pct"]
    n10 = st.costed_timer(prices, mret, cost_bps=10.0, borrow_bps_yr=0.0)["net_ann_pct"]
    assert n0 > n5 > n10


def test_borrow_reduces_net():
    prices = data.synthetic_world(val_edge=0.05, seed=15)
    mret = prices.pct_change()
    no_borrow = st.costed_timer(prices, mret, cost_bps=5.0, borrow_bps_yr=0.0)["net_ann_pct"]
    with_borrow = st.costed_timer(prices, mret, cost_bps=5.0, borrow_bps_yr=50.0)["net_ann_pct"]
    assert with_borrow < no_borrow


# --------------------------------------------------------------------------- #
# The spine: planted reversal fires, the null does not (over many seeds)
# --------------------------------------------------------------------------- #
def test_planted_reversal_fires():
    d = st.synthetic_detect(data.synthetic_world(val_edge=0.05, seed=793))
    assert d["hac_t"] > 4.0
    assert d["mean_bps"] > 0.0


def test_null_world_does_not_systematically_fire():
    ts = np.array([st.synthetic_detect(data.synthetic_world(val_edge=0.0, seed=793 + s))["hac_t"]
                   for s in range(20)])
    assert abs(np.nanmean(ts)) < 0.8
    assert np.mean(np.abs(ts) >= 2.0) <= 0.25


def test_placebo_beats_are_rare_on_planted_world():
    prices = data.synthetic_world(val_edge=0.05, seed=793)
    mret = prices.pct_change()
    pl = st.random_placebo(prices, mret, n_seeds=20)
    assert pl["p_value"] < 0.1
    assert pl["frac_t_ge_2"] < 0.5

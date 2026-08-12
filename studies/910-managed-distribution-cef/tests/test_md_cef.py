"""Offline, fixed-seed tests for the managed-distribution-CEF machinery.

The synthetic world is deterministic; the excess-vs-excess CAPM recovers a planted net carry as
alpha and the planted leverage as beta; the null (pure levered beta) does NOT manufacture an
alpha; a fat distribution that is all return-of-capital (carry == leak) produces NO alpha (the
mREIT trap); the basket builder is composition-stable and one-lag; costs reduce the net; the HAC
and Sharpe primitives behave. All offline — the real-cache test is skipped when _cache is absent.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from md_cef import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world determinism + planted-signal recovery
# --------------------------------------------------------------------------- #
def test_world_deterministic(planted_world):
    w2 = data.synthetic_world(carry_annual=0.05, roc_leak_annual=0.0, seed=910)
    assert np.allclose(planted_world["cef"].to_numpy(), w2["cef"].to_numpy())
    assert np.allclose(planted_world["mkt"].to_numpy(), w2["mkt"].to_numpy())


def test_planted_carry_recovered_as_alpha(planted_world, null_world):
    a_plant = st.synthetic_detect(planted_world)["alpha_ann_pct"]
    a_null = st.synthetic_detect(null_world)["alpha_ann_pct"]
    # the planted +5 %/yr net carry shows up as a +5 pp lift in the recovered alpha
    assert (a_plant - a_null) == pytest.approx(5.0, abs=0.5)


def test_planted_fires_null_silent(planted_world, null_world):
    assert abs(st.synthetic_detect(null_world)["t_alpha"]) < 2.0      # null: no alpha
    assert st.synthetic_detect(planted_world)["t_alpha"] > 2.0        # planted: lights up


def test_beta_recovers_planted_leverage(planted_world):
    d = st.synthetic_detect(planted_world)
    assert d["beta"] == pytest.approx(1.1, abs=0.15)                  # planted beta = 1.1
    assert d["r2"] > 0.7


def test_return_of_capital_trap_yields_no_alpha(roc_trap_world, null_world):
    # a fat distribution that is 100% return-of-capital (carry == leak) is indistinguishable
    # from the null: net carry is zero, so no alpha survives.
    assert abs(st.synthetic_detect(roc_trap_world)["t_alpha"]) < 2.0
    assert st.synthetic_detect(roc_trap_world)["alpha_ann_pct"] == pytest.approx(
        st.synthetic_detect(null_world)["alpha_ann_pct"], abs=1e-6
    )


def test_null_not_biased_over_seeds():
    # over 20 seeds the null alpha averages ~0 and rarely fires (a well-behaved t>=2 test)
    ts = np.array([
        st.synthetic_detect(data.synthetic_world(carry_annual=0.0, seed=910 + s))["t_alpha"]
        for s in range(20)
    ])
    assert (np.abs(ts) >= 2).sum() <= 3          # ~5% false-positive rate at t>=2


# --------------------------------------------------------------------------- #
# Basket construction: composition-stable, one-lag, vectorised
# --------------------------------------------------------------------------- #
def test_basket_is_equal_weight_mean():
    idx = pd.period_range("2015-01", periods=24, freq="M").to_timestamp(how="end").normalize()
    panel = pd.DataFrame({
        "PDI": np.linspace(0.01, 0.02, 24),
        "UTF": np.linspace(-0.01, 0.03, 24),
        "BST": np.full(24, 0.005),
        "RQI": np.linspace(0.02, -0.01, 24),
    }, index=idx)
    b = st.equal_weight_basket(panel, data.BASKET)
    assert np.allclose(b.to_numpy(), panel[data.BASKET].mean(axis=1).to_numpy())


def test_basket_drops_incomplete_composition():
    # while the youngest member is missing, the basket does not exist (composition-stable)
    idx = pd.period_range("2014-01", periods=12, freq="M").to_timestamp(how="end").normalize()
    panel = pd.DataFrame({
        "PDI": np.full(12, 0.01), "UTF": np.full(12, 0.01),
        "RQI": np.full(12, 0.01),
        "BST": [np.nan] * 6 + [0.02] * 6,   # BST only present in the back half
    }, index=idx)
    b = st.equal_weight_basket(panel, data.BASKET)
    assert len(b) == 6                       # only the fully-populated months survive
    assert np.allclose(b.to_numpy(), (0.01 * 3 + 0.02) / 4)


# --------------------------------------------------------------------------- #
# Excess-of-cash + costs
# --------------------------------------------------------------------------- #
def test_excess_of_cash_subtracts_cash():
    idx = pd.period_range("2015-01", periods=10, freq="M").to_timestamp(how="end").normalize()
    r = pd.Series(np.full(10, 0.01), index=idx, name="X")
    c = pd.Series(np.full(10, 0.002), index=idx)
    assert np.allclose(st.excess(r, c).to_numpy(), 0.008)


def test_costs_reduce_net(planted_world):
    # a synthetic 4-name panel so costed_net has a basket to charge
    idx = pd.period_range("2015-01", periods=140, freq="M").to_timestamp(how="end").normalize()
    rng = np.random.default_rng(1)
    panel = pd.DataFrame(
        {m: 0.008 + rng.normal(0, 0.03, 140) for m in data.BASKET}, index=idx)
    cash = pd.Series(np.full(140, 0.001), index=idx)
    spy = pd.Series(0.006 + rng.normal(0, 0.04, 140), index=idx)
    c = st.costed_net(panel, data.BASKET, cash, spy)
    assert c["net_exret_bps"] < c["gross_exret_bps"]
    assert c["charge_bps_per_mo"] > 0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_nw_matches_iid_on_white_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(0.005, 0.02, 3000)
    mean, t = st.nw_mean_t(x, lags=6)
    iid_t = mean / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(t - iid_t) < 0.6


def test_hac_ols_recovers_known_line():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.04, 2000)
    y = 0.001 + 1.3 * x + rng.normal(0, 0.005, 2000)
    reg = st.hac_ols(y, x, lags=6)
    assert reg["beta"] == pytest.approx(1.3, abs=0.05)
    assert reg["alpha"] == pytest.approx(0.001, abs=0.0006)


def test_ann_sharpe_sign_and_scale():
    idx = pd.period_range("2015-01", periods=120, freq="M").to_timestamp(how="end").normalize()
    x = pd.Series(np.full(120, 0.01), index=idx)  # constant positive -> +inf sd 0 guard
    # add tiny noise so sd>0
    x = x + np.random.default_rng(0).normal(0, 0.02, 120)
    s = st.ann_sharpe(x.to_numpy())
    assert s > 0


def test_max_drawdown_is_negative_on_decline():
    idx = pd.period_range("2015-01", periods=12, freq="M").to_timestamp(how="end").normalize()
    r = pd.Series([0.1, 0.1, -0.2, -0.2, -0.1, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
    assert st.max_drawdown(r) < 0


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(3)
    x = rng.normal(0.006, 0.03, 160)
    b = st.bootstrap_sharpe_ci(x, n_boot=1000)
    assert b["ci_low"] <= b["sharpe"] <= b["ci_high"]


# --------------------------------------------------------------------------- #
# Real-cache test — only runs when the yfinance cache is present (absent on CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE),
                    reason="real yfinance cache absent (offline / CI)")
def test_real_cache_shapes():
    prices = data.load_prices()
    panel = data.monthly_panel(prices, asof=data.AS_OF)
    assert panel.index.max() <= pd.Timestamp(data.AS_OF)
    for t in data.TICKERS:
        assert t in panel.columns
    basket = st.equal_weight_basket(panel, data.BASKET)
    s = st.fund_stats(basket, panel["SPY"], panel["BIL"])
    assert s["n"] > 100                 # ~11.5 years of monthly basket tape
    assert -1.0 < s["beta"] < 3.0

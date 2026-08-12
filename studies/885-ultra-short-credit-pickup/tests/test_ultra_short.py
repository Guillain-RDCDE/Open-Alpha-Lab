"""Offline, fixed-seed tests for the ultra-short credit-pickup machinery.

The synthetic world is deterministic; a planted pickup is recovered (positive excess
mean, HAC t lights up) and recovered by the *exact* planted amount; the null shows no
pickup; the excess is genuinely minus-cash; return construction is causal (no
look-ahead); costs reduce the net; the sub-era and drawdown helpers behave; the
inference primitives are sane. All offline — no real cache, no network.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ultra_short import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The synthetic world — deterministic, faithful machinery
# --------------------------------------------------------------------------- #
def test_world_deterministic(pickup_world):
    again = data.synthetic_world(pickup_bps_yr=120.0, seed=885, n_days=2000)
    assert np.allclose(pickup_world.to_numpy(), again.to_numpy())


def test_planted_pickup_recovered(pickup_world):
    ex = (pickup_world["CREDIT"] - pickup_world["CASH"]).dropna()
    h = st.hac_mean(ex)
    assert h["mean_bps_yr"] > 0            # a positive pickup over cash
    assert st.ann_sharpe(ex) > 0           # a positive excess Sharpe
    # the planted +120 bps/yr is in the right ballpark (noise-limited)
    assert 40.0 < h["mean_bps_yr"] < 200.0


def test_null_world_no_pickup(null_world):
    ex = (null_world["CREDIT"] - null_world["CASH"]).dropna()
    # a single seed can wander, but must not clear the significance bar from noise
    assert abs(st.hac_mean(ex)["t_nw"]) < 2.5


def test_null_fires_on_no_seed():
    """Across seeds the null must not manufacture a significant pickup."""
    fires = 0
    for s in range(12):
        w = data.synthetic_world(pickup_bps_yr=0.0, seed=885 + s, n_days=2000)
        ex = (w["CREDIT"] - w["CASH"]).dropna()
        if abs(st.hac_mean(ex)["t_nw"]) >= 2.0:
            fires += 1
    assert fires == 0


def test_recovery_is_exact():
    """mean(planted) - mean(null) must equal the planted carry (machinery faithful)."""
    null = data.synthetic_world(pickup_bps_yr=0.0, seed=885, n_days=2000)
    planted = data.synthetic_world(pickup_bps_yr=120.0, seed=885, n_days=2000)
    m0 = (null["CREDIT"] - null["CASH"]).mean() * st.TRADING_DAYS * 1e4
    m1 = (planted["CREDIT"] - planted["CASH"]).mean() * st.TRADING_DAYS * 1e4
    assert abs((m1 - m0) - 120.0) < 1e-6   # same seed -> only the drift term differs


# --------------------------------------------------------------------------- #
# Returns / excess / causality
# --------------------------------------------------------------------------- #
def test_excess_is_minus_cash():
    px = pd.DataFrame(
        {"A": np.linspace(100, 110, 30), "BIL": np.linspace(100, 101, 30)},
        index=pd.bdate_range("2020-01-01", periods=30),
    )
    r = st.daily_returns(px)
    ex = st.excess(r, "A", "BIL")
    manual = (r["A"] - r["BIL"]).dropna()
    assert np.allclose(ex.to_numpy(), manual.to_numpy())


def test_returns_are_causal_no_lookahead():
    """A day-t return must not depend on any price after day t."""
    rng = np.random.default_rng(0)
    px = pd.DataFrame(
        {"A": 100 * np.cumprod(1 + rng.normal(0, 0.002, 40))},
        index=pd.bdate_range("2020-01-01", periods=40),
    )
    full = st.daily_returns(px)
    truncated = st.daily_returns(px.iloc[:25])  # hide the future
    # the first 25 returns are identical whether or not the future exists
    assert np.allclose(full["A"].iloc[:25].to_numpy(),
                       truncated["A"].to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------- #
# Costs, eras, drawdown
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(pickup_world):
    ex = (pickup_world["CREDIT"] - pickup_world["CASH"]).dropna()
    gross = st.net_of_cost_excess(ex, cost_bps_oneway=0.0)["net_bps_yr"]
    net = st.net_of_cost_excess(ex, cost_bps_oneway=5.0, turnover_yr=1.0)["net_bps_yr"]
    assert net < gross
    assert abs((gross - net) - 10.0) < 1e-6   # 2 sides x 5 bp x 1 turnover = 10 bps/yr


def test_era_cut_splits(pickup_world):
    ex = (pickup_world["CREDIT"] - pickup_world["CASH"]).dropna()
    ec = st.era_cut(ex, split="2017-01-01")
    assert ec["early"]["n"] > 0 and ec["late"]["n"] > 0
    assert ec["early"]["n"] + ec["late"]["n"] == len(ex)


def test_max_drawdown_sign_and_bracket():
    px = pd.Series([100.0, 110.0, 99.0, 105.0],
                   index=pd.bdate_range("2020-01-01", periods=4))
    dd = st.max_drawdown(px)
    assert dd["depth_pct"] < 0
    assert abs(dd["depth_pct"] - (99.0 / 110.0 - 1.0) * 100.0) < 1e-9


def test_calendar_year_table_shape(pickup_world):
    r = st.daily_returns(
        pd.DataFrame({"CREDIT": 100 * (1 + pickup_world["CREDIT"]).cumprod()})
    )
    cyt = st.calendar_year_table(r, ["CREDIT"])
    assert "CREDIT" in cyt.columns and len(cyt) >= 1


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_welch_zero_on_equal_means():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 3000)
    b = rng.normal(0.0, 1.0, 3000)
    assert abs(st.welch_t(a, b)) < 3.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_sharpe_ci_brackets_point(pickup_world):
    ex = (pickup_world["CREDIT"] - pickup_world["CASH"]).dropna()
    ci = st.sharpe_ci(ex, n_boot=500, seed=885)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]


# --------------------------------------------------------------------------- #
# Real-cache test — only runs when the (git-ignored) cache is present
# --------------------------------------------------------------------------- #
CACHE = data.PRICES_CACHE


@pytest.mark.skipif(not os.path.exists(CACHE), reason="no real _cache present (offline CI)")
def test_real_cache_sane():
    px = data.load_prices()
    assert set(data.TICKERS).issubset(px.columns)
    assert px.index.max() <= pd.Timestamp(data.AS_OF)
    r = st.daily_returns(px)
    common = st.align_common(r, data.TICKERS)
    assert len(common) > 500
    # the sleeve pickup over BIL is positive in point estimate on the real tape
    sleeve = common[data.CREDIT].mean(axis=1)
    assert st.hac_mean((sleeve - common["BIL"]).dropna())["mean_bps_yr"] > 0

"""Offline, fixed-seed tests for the Defensive-Momentum machinery.

The synthetic world is deterministic; on a planted edge the blend beats MTUM's Sharpe and
posts a shallower drawdown; on the null (identical sleeves) any blend is exactly a sleeve
so the advantage is zero and drawdowns are equal; the inverse-vol weights are point-in-
time (one shift, no look-ahead); costs reduce the net; the inference primitives behave.
All offline — the whole suite must pass with NO real cache present.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from def_momentum import data, strategy as st  # noqa: E402

CACHE = data.TAPE_CACHE


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    w2 = data.synthetic_sleeves(edge=1.0, seed=895, n_months=160)
    assert np.allclose(edge_world.to_numpy(), w2.to_numpy())


def test_synthetic_index_is_period():
    w = data.synthetic_sleeves(edge=0.5, n_months=160)
    assert isinstance(w.index, pd.PeriodIndex)      # no ns-Timestamp overflow trap
    assert list(w.columns) == ["MTUM", "USMV", "SPY", "BIL"]


# --------------------------------------------------------------------------- #
# Null world — identical sleeves => no diversification benefit
# --------------------------------------------------------------------------- #
def test_null_sleeves_identical(null_world):
    assert np.allclose(null_world["MTUM"].to_numpy(), null_world["USMV"].to_numpy())


def test_null_blend_has_no_advantage(null_world):
    sleeves = null_world[["MTUM", "USMV"]]
    blend = st.fixed_blend(sleeves, 0.5)
    cash = null_world["BIL"]
    bl_ex = (blend["gross"] - cash)
    mt_ex = null_world["MTUM"] - cash
    race = st.sharpe_advantage(bl_ex, mt_ex)
    assert abs(race["sharpe_adv"]) < 1e-9        # blend == sleeve exactly
    assert abs(race["diff_bps"]) < 1e-6
    a = st.ann_stats(blend["gross"], cash)
    b = st.ann_stats(null_world["MTUM"], cash)
    assert abs(a["maxdd"] - b["maxdd"]) < 1e-9   # identical drawdown


# --------------------------------------------------------------------------- #
# Planted edge — the blend really is "momentum without the crashes"
# --------------------------------------------------------------------------- #
def test_planted_edge_shallower_drawdown(edge_world):
    sleeves = edge_world[["MTUM", "USMV"]]
    cash = edge_world["BIL"]
    blend = st.fixed_blend(sleeves, 0.5)
    bl_dd = st.ann_stats(blend["gross"], cash)["maxdd"]
    mt_dd = st.ann_stats(edge_world["MTUM"], cash)["maxdd"]
    assert bl_dd > mt_dd            # less negative => shallower than MTUM alone


def test_planted_edge_positive_sharpe_advantage(edge_world):
    sleeves = edge_world[["MTUM", "USMV"]]
    cash = edge_world["BIL"]
    blend = st.fixed_blend(sleeves, 0.5)
    bl_ex = blend["gross"] - cash
    mt_ex = edge_world["MTUM"] - cash
    race = st.sharpe_advantage(bl_ex, mt_ex)
    assert race["sharpe_adv"] > 0
    boot = st.bootstrap_sharpe_adv(bl_ex, mt_ex, n_draws=500, seed=895)
    assert boot["lo"] < boot["obs"] < boot["hi"]


def test_synthetic_detect_null_vs_edge(null_world, edge_world):
    assert abs(st.synthetic_detect(null_world)["sharpe_adv"]) < 1e-9
    assert st.synthetic_detect(edge_world)["sharpe_adv"] > 0


# --------------------------------------------------------------------------- #
# No look-ahead in the inverse-vol weights
# --------------------------------------------------------------------------- #
def test_inv_vol_weights_point_in_time():
    idx = pd.period_range("2013-05", periods=40, freq="M")
    rng = np.random.default_rng(0)
    sleeves = pd.DataFrame(rng.normal(0.01, 0.03, (40, 2)), index=idx,
                           columns=["MTUM", "USMV"])
    raw = (1.0 / sleeves.rolling(12, min_periods=12).std())
    raw = raw.div(raw.sum(axis=1), axis=0)
    w = st.inv_vol_weights(sleeves, lookback=12, lag=1)
    # month-t weight equals the unshifted weight from month t-1 (formed on known info)
    assert np.allclose(w.iloc[20].to_numpy(), raw.iloc[19].to_numpy(), equal_nan=True)


def test_inv_vol_weights_sum_to_one():
    w = data.synthetic_sleeves(edge=1.0, n_months=60)[["MTUM", "USMV"]]
    ww = st.inv_vol_weights(w, 12, 1).dropna()
    assert np.allclose(ww.sum(axis=1).to_numpy(), 1.0)


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(edge_world):
    sleeves = edge_world[["MTUM", "USMV"]]
    blend = st.vol_weighted_blend(sleeves, 12, 1)
    gross = blend["gross"].dropna().mean()
    net = st.apply_costs(blend, cost_bps_oneway=3.0).dropna().mean()
    assert net < gross


def test_fixed_blend_turnover_small_but_positive(edge_world):
    sleeves = edge_world[["MTUM", "USMV"]]
    blend = st.fixed_blend(sleeves, 0.5)
    tud = blend["turnover"].iloc[1:]     # after the establishing month
    assert (tud >= 0).all()
    assert tud.mean() < 0.2              # a 2-asset monthly rebalance barely trades


# --------------------------------------------------------------------------- #
# Crash-window drawdown helper
# --------------------------------------------------------------------------- #
def test_window_drawdown_matches_manual():
    idx = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    r = pd.Series([0.02, -0.10, -0.15, 0.08], index=idx)
    dd = st.window_drawdown(r, "2020-01-01", "2020-06-30")
    manual = (1 - 0.10) * (1 - 0.15) - 1.0        # peak after Jan, trough after Mar
    assert abs(dd - manual) < 1e-9


def test_calendar_year_returns():
    idx = pd.period_range("2020-01", periods=24, freq="M").to_timestamp(how="end")
    r = pd.Series(0.01, index=idx)
    cal = st.calendar_year_returns(r)
    assert abs(cal.loc[2020] - ((1.01 ** 12) - 1.0)) < 1e-9


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_sign():
    a = np.array([0.03, 0.04, 0.05, 0.02])
    b = np.array([-0.01, 0.0, -0.02, 0.01])
    assert st.welch_t(a, b) > 0


def test_wilson_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


# --------------------------------------------------------------------------- #
# Real-cache smoke test (skipped on CI where _cache/ is absent)
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


@pytest.mark.skipif(not os.path.exists(CACHE), reason="no real cache present (offline CI)")
def test_real_cache_shapes():
    prices = data.load_prices()
    for tk in data.TICKERS:
        assert tk in prices.columns
    mret = data.monthly_total_returns(prices)
    assert mret.index.max() <= pd.Timestamp(data.AS_OF)
    sleeves = mret[data.SLEEVES].dropna()
    blend = st.fixed_blend(sleeves, 0.5)
    assert blend["gross"].notna().sum() > 100

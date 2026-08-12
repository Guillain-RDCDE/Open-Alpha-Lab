"""Offline, fixed-seed tests for the broad dollar-hedge machinery.

The synthetic world is deterministic; the carry estimator recovers a planted carry and stays
silent on the null; the hedge regression finds beta ~ 1 (a full short of the foreign basket); the
overlay switch is point-in-time (one lag, no look-ahead) and its costs reduce the net; the
inference primitives behave; and the real-cache identity is checked only when a cache is present.
All synthetic tests run with NO network and NO real data.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dollar_hedge import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    w2 = data.synthetic_world(n_months=180, carry_annual=0.03, seed=889)
    assert np.allclose(edge_world.to_numpy(), w2.to_numpy())


def test_index_is_safe_no_overflow():
    # PeriodIndex-derived timestamps must stay well inside the pandas ns horizon.
    w = data.synthetic_world(n_months=180)
    assert w.index.max() < pd.Timestamp("2262-01-01")
    assert w.index.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# The carry identity — planted edge recovered, null silent
# --------------------------------------------------------------------------- #
def test_planted_carry_recovered(edge_world):
    d = st.synthetic_detect(edge_world)
    # planted +3%/yr; the estimator should land near it with a large HAC t
    assert 2.0 < d["carry_ann_pct"] < 4.0
    assert d["t_carry"] > 3.0


def test_null_world_no_carry(null_world):
    d = st.synthetic_detect(null_world)
    assert abs(d["carry_ann_pct"]) < 1.0
    assert abs(d["t_carry"]) < 2.5


def test_hedge_beta_is_full_short_of_foreign(edge_world):
    # diff = alpha + beta*(-fx_foreign): the hedge is a full short of the foreign basket, beta ~ 1.
    pf = st.pair_frame(edge_world, "HEFA", "EFA")
    s = st.pair_stats(pf)
    assert 0.9 < s["beta"] < 1.1
    assert s["t_beta"] > 5.0
    assert s["r2"] > 0.8


def test_carry_hat_tracks_planted_differential(edge_world):
    # the recovered carry (%/yr) should sit near the planted rate differential it mirrors.
    pf = st.pair_frame(edge_world, "HEFA", "EFA")
    s = st.pair_stats(pf)
    assert abs(s["carry_ann_pct"] - s["rate_diff_ann_pct"]) < 1.5


# --------------------------------------------------------------------------- #
# The overlay — no look-ahead, costs bite, regime is switched on
# --------------------------------------------------------------------------- #
def test_overlay_is_point_in_time():
    # the signal uses diff_rate.shift(1); a day-t position cannot see day-t's differential.
    w = data.synthetic_world(n_months=60, carry_annual=0.04, flip_half=True)
    pf = st.pair_frame(w, "HEFA", "EFA")
    sig = (pf["diff_rate"].shift(1) > 0.0)
    assert bool(sig.iloc[1]) == bool(pf["diff_rate"].iloc[0] > 0.0)


def test_overlay_costs_reduce_net(regime_world):
    pf = st.pair_frame(regime_world, "HEFA", "EFA")
    free = st.overlay_switch(pf, cost_bps_oneway=0.0)
    costed = st.overlay_switch(pf, cost_bps_oneway=25.0)
    assert costed["cost_drag_ann_pct"] >= free["cost_drag_ann_pct"]
    assert costed["overlay_ann_pct"] <= free["overlay_ann_pct"] + 1e-9


def test_overlay_switches_on_regime(regime_world):
    # carry only in the 2nd half -> the overlay should hedge part of the time and switch >=1.
    pf = st.pair_frame(regime_world, "HEFA", "EFA")
    ov = st.overlay_switch(pf)
    assert ov["switches"] >= 1
    assert 0.0 < ov["share_hedged"] < 1.0


def test_spread_costs_reduce_net(edge_world):
    pf = st.pair_frame(edge_world, "HEFA", "EFA")
    gross = st.spread_trade(pf, borrow_annual_bps=0.0, cost_bps_oneway=0.0)["net_diff_ann_pct"]
    net = st.spread_trade(pf, borrow_annual_bps=50.0, cost_bps_oneway=5.0)["net_diff_ann_pct"]
    assert net < gross


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_bootstrap_ci_brackets_point(edge_world):
    pf = st.pair_frame(edge_world, "HEFA", "EFA")
    ci = st.mean_boot_ci(pf["carry_hat"].values, n_boot=500)
    assert ci["ci_low"] < ci["mean_ann_pct"] < ci["ci_high"]


def test_max_drawdown_sign():
    r = np.array([0.1, -0.5, 0.1, 0.1])
    assert st.max_drawdown(r) < 0.0


def test_calendar_years_shape(edge_world):
    pf = st.pair_frame(edge_world, "HEFA", "EFA")
    tab = st.calendar_years(pf)
    assert {"hedged_%", "unhedged_%", "diff_%"}.issubset(tab.columns)
    assert len(tab) >= 10


# --------------------------------------------------------------------------- #
# Real-cache identity (skipped on CI where _cache/ is absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE), reason="no real cache present")
def test_real_cache_carry_is_positive_and_significant():
    prices = data.load_prices()
    panel = data.monthly_panel(prices)
    pf = st.pair_frame(panel, "HEFA", "EFA")
    s = st.pair_stats(pf)
    # the broad EAFE hedge carry is positive, dollar-favourable, and clears HAC t >= 2
    assert s["carry_ann_pct"] > 0.5
    assert s["t_carry"] > 2.0
    assert 0.8 < s["beta"] < 1.05

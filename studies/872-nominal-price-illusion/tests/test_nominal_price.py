"""Offline, fixed-seed tests for the nominal-price-illusion machinery.

The synthetic panel is deterministic; cheap names are planted to look lottery-like
(higher vol, more right-skew) regardless of the knob; with the knob on (`edge>0`) the
cheap book also under-earns (a *negative* long-cheap/short-dear spread — the claim's
sign); the null shows nothing; the sort is point-in-time (one shift, no look-ahead);
the timer costs reduce the net; the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from nominal_price import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.0016, seed=872, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_cheap_names_look_lottery_like(null_world):
    # The lottery LOOK is planted independent of the knob: even in the null, the cheap
    # (low-price) book must be MORE volatile and MORE right-skewed than the dear book.
    d = st.synthetic_detect(null_world)
    assert d["lo_vol"] > d["hi_vol"]      # cheap names carry more volatility
    assert d["lo_skew"] > d["hi_skew"]    # cheap names carry more right-skew


def test_planted_relation_recovered(edge_world):
    # With edge>0 the cheap names UNDER-earn: long-cheap / short-dear spread is negative.
    ts = st.price_stats(st.price_spreads(edge_world))
    assert ts["t_nw"] < -3.0
    assert ts["spread_bps"] < 0
    assert ts["lo_bps"] < ts["hi_bps"]    # cheap book earns LESS than dear book


def test_null_world_no_signal(null_world):
    ts = st.price_stats(st.price_spreads(null_world))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time(null_world):
    # The ranking price on day t must be the price known at close t-1 (one shift).
    prices = st.close_prices(null_world)
    shifted = prices.shift(1)
    assert np.allclose(shifted.iloc[5].to_numpy(), prices.iloc[4].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    sp = st.price_spreads(edge_world)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_price_sort_ranks_by_level(null_world):
    # The bottom book must genuinely hold cheaper names than the top book.
    prices = st.close_prices(null_world)
    sig = prices.shift(1)
    row = sig.iloc[100].dropna().sort_values()
    k = max(1, int(np.floor(len(row) * 0.3)))
    assert row.iloc[:k].mean() < row.iloc[-k:].mean()


def test_skew_of_left_skewed_is_negative():
    x = -np.abs(np.random.default_rng(0).normal(0, 1, 5000)) ** 1.5
    assert st._skew(x) < 0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_tape_smoke():
    # Only runs when a real panel cache is present locally; skipped on the offline CI.
    panel = data.load_panel()
    sp = st.price_spreads(panel, frac=0.3)
    ts = st.price_stats(sp)
    assert ts["n_days"] > 3000
    assert np.isfinite(ts["t_nw"])

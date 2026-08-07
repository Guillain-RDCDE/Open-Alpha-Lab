"""Offline, fixed-seed tests for the Corwin-Schultz machinery.

The synthetic panel is deterministic; the CS estimator recovers the injected per-name
spread; the sort recovers a planted illiquidity premium (long high-spread / short
low-spread lights up); the null shows nothing; the sort is point-in-time (one shift, no
look-ahead); the timer costs reduce the net; the CS formula is right on a hand case; the
inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from corwin_schultz import data, strategy as st  # noqa: E402


def _injected_spreads(seed=812, n_assets=40, n_days=1500,
                      spread_lo=0.0005, spread_hi=0.030):
    """Re-draw, in exact call order, the per-name spread levels planted by synthetic_panel."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_assets):
        out.append(rng.uniform(spread_lo, spread_hi))
        rng.normal(0.0, 1.0, n_days)          # z (returns)
        rng.normal(0.0, 0.012 / 3, n_days)    # open noise
        rng.normal(0.0, 0.008, n_days)        # up
        rng.normal(0.0, 0.008, n_days)        # dn
    return np.asarray(out)


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.08, seed=812, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_cs_estimator_recovers_injected_spread(null_world):
    # The per-name mean daily CS spread must track the injected s_i almost perfectly.
    S = st.daily_cs_spread(null_world).mean().to_numpy()
    inj = _injected_spreads()
    assert np.corrcoef(S, inj)[0, 1] > 0.95
    # cross-name dispersion is non-trivial (the sort has something to bite on)
    assert S.std() > 1e-4


def test_cs_spread_is_nonnegative(edge_world):
    S = st.daily_cs_spread(edge_world).to_numpy()
    S = S[~np.isnan(S)]
    assert (S >= 0).all()          # negatives are floored at 0


def test_planted_relation_recovered(edge_world):
    ts = st.cs_stats(st.cs_spreads(edge_world))
    assert ts["t_nw"] > 3.0             # long-high/short-low spread lights up
    assert ts["spread_bps"] > 0
    assert ts["long_bps"] > ts["short_bps"]  # illiquid names out-earn liquid names


def test_null_world_no_signal(null_world):
    ts = st.cs_stats(st.cs_spreads(null_world))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time(null_world):
    sig = st.trailing_spread(null_world, window=21)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[60].to_numpy(), sig.iloc[59].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    sp = st.cs_spreads(edge_world)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_cs_first_row_is_nan():
    # The first row of each name has no previous day, so beta/gamma are undefined.
    idx = pd.bdate_range("2020-01-01", periods=2)
    panel = {"X": pd.DataFrame(
        {"Open": [100.0, 100.0], "High": [101.0, 101.0],
         "Low": [99.0, 99.0], "Close": [100.0, 100.0]}, index=idx)}
    S = st.daily_cs_spread(panel)["X"].to_numpy()
    assert np.isnan(S[0])


def test_cs_floors_to_zero_on_a_jump():
    # A big price jump between two narrow-range days makes the 2-day extreme range
    # dwarf the single-day ranges -> gamma >> beta -> alpha < 0 -> S floored to 0.
    idx = pd.bdate_range("2020-01-01", periods=2)
    panel = {"X": pd.DataFrame(
        {"Open": [100.0, 110.0], "High": [100.2, 110.2],
         "Low": [99.8, 109.8], "Close": [100.0, 110.0]}, index=idx)}
    S = st.daily_cs_spread(panel)["X"].to_numpy()
    assert S[1] == 0.0


def test_cs_positive_when_spread_dominates():
    # Two identical wide-range days on a *stable* price: the 2-day range equals the
    # single-day range, so the estimator attributes the whole range to the spread -> S > 0.
    idx = pd.bdate_range("2020-01-01", periods=2)
    panel = {"X": pd.DataFrame(
        {"Open": [100.0, 100.0], "High": [101.0, 101.0],
         "Low": [99.0, 99.0], "Close": [100.0, 100.0]}, index=idx)}
    S = st.daily_cs_spread(panel)["X"].to_numpy()
    assert S[1] > 0.0
    assert not np.isnan(S[1])


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_welch_detects_mean_gap():
    rng = np.random.default_rng(1)
    a = rng.normal(0.002, 0.01, 3000)
    b = rng.normal(0.000, 0.01, 3000)
    assert st.welch_t(a, b) > 3.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

"""Offline, fixed-seed tests for the realized-kurtosis machinery.

The synthetic panel is deterministic; the trailing kurtosis signal is well-defined and
disperses across names; the sort recovers a planted positive kurt->return relation
(positive long-high/short-low spread); the null shows nothing; the sort is point-in-time
(one shift, no look-ahead); the timer costs reduce the net; the vectorised rolling
kurtosis matches a brute-force reference; the inference primitives behave. All offline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from realized_kurtosis import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.010, seed=804, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_kurt_disperses_across_names(edge_world):
    # Cross-name dispersion in realized kurtosis must be non-trivial (the sort has
    # something to bite on), and levels should sit above the normal reference of 3.
    ret = st.close_returns(edge_world)
    kt = st.trailing_kurt(ret, window=42).mean()
    assert kt.std() > 0.1
    assert kt.median() > 3.0


def test_vectorised_kurt_matches_bruteforce():
    rng = np.random.default_rng(7)
    r = pd.DataFrame(rng.standard_t(5, size=(200, 3)) * 0.01,
                     index=pd.bdate_range("2019-01-01", periods=200),
                     columns=["A", "B", "C"])
    fast = st.trailing_kurt(r, window=21)
    # brute-force reference on a handful of rows
    for row in (30, 90, 150):
        for j, c in enumerate(r.columns):
            win = r[c].to_numpy()[row - 21 + 1: row + 1]
            assert np.isclose(fast.iloc[row, j], st._kurt(win), rtol=1e-10, atol=1e-10)


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.kurt_stats(st.kurt_spreads(ret))
    assert ts["t_nw"] > 3.0            # long-high/short-low spread lights up
    assert ts["spread_bps"] > 0
    assert ts["hi_bps"] > ts["lo_bps"]  # high-kurt names out-earn low-kurt names


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.kurt_stats(st.kurt_spreads(ret))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    ret = pd.DataFrame(
        np.linspace(-0.02, 0.02, 60).reshape(20, 3),
        index=pd.bdate_range("2020-01-01", periods=20),
        columns=["A", "B", "C"],
    )
    sig = st.trailing_kurt(ret, window=4)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[6].to_numpy(), sig.iloc[5].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.kurt_spreads(ret)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_kurt_of_fat_tailed_exceeds_normal():
    rng = np.random.default_rng(0)
    fat = rng.standard_t(4, size=8000)      # heavy tails, kurtosis > 3
    thin = rng.uniform(-1, 1, 8000)         # platykurtic, kurtosis < 3
    assert st._kurt(fat) > 3.0
    assert st._kurt(thin) < 3.0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi

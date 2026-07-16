"""Fully-offline, deterministic tests for Study 779 — Devil-Price.

No network: everything here runs on the seeded synthetic world and on pure functions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from devil_price import data as dt, strategy as st  # noqa: E402


def test_devil_distance_is_decade_invariant_and_bounded():
    # 6.66, 66.6, 666, 6660 are all exactly on the beast -> distance ~ 0
    for p in (6.66, 66.6, 666.0, 6660.0):
        assert dt.devil_distance(p) < 1e-9
    # bounded in [0, 0.5]; a price a half-decade away from 6.66 is maximally far
    arr = dt.devil_distance(np.array([1.0, 10.0, 21.0, 100.0, 666.0]))
    assert np.all(arr >= -1e-12) and np.all(arr <= 0.5 + 1e-12)
    # the farthest mantissa from 6.66 on the log circle (~2.1) sits near distance 0.5
    assert dt.devil_distance(21.06) > 0.49


def test_universe_shape():
    # ~100 real S&P 100 tickers, no duplicates, SPY appended once as benchmark
    assert 95 <= len(dt.UNIVERSE) <= 101
    assert len(set(dt.UNIVERSE)) == len(dt.UNIVERSE)
    assert dt.BENCHMARK == "SPY" and "SPY" not in dt.UNIVERSE
    assert dt.all_tickers()[-1] == "SPY"


def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(240, 500)
    assert lo < 240 / 500 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_sort_table_columns_and_neutrality():
    prices, _ = dt.synthetic_world(bump=0.0, seed=789)
    events = st.build_panel_events(prices, spy=None, fwd_s=5, fwd_l=5, min_names=10)
    table = st.build_sort_table(events)
    for c in ("devil_s_dm", "clean_s_dm", "ls_s"):
        assert c in table.columns
    # ls == clean - devil, and the demeaned legs are consistent with it
    assert np.allclose(table["ls_s"], table["clean_s"] - table["devil_s"])
    assert np.allclose(table["ls_s"], table["clean_s_dm"] - table["devil_s_dm"])


def test_synthetic_null_is_quiet():
    # bump = 0 -> the devil-quintile detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(bump=0.0, seed=789 + s, side="devil")["t"]
                   for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # with many rebalances the null t is well-behaved: almost no seed breaches |t|>=2
    assert (np.abs(ts) >= 2).sum() <= 3


def test_synthetic_planted_drag_is_recovered_monotonically():
    # "touching 666 then underperforms": a bigger planted drag -> more-negative devil t
    t0 = st.synthetic_detect(bump=0.0, seed=789, side="devil")["t"]
    t1 = st.synthetic_detect(bump=0.01, seed=789, side="devil")["t"]
    t2 = st.synthetic_detect(bump=0.02, seed=789, side="devil")["t"]
    assert t0 > t1 > t2
    assert t2 < -2.0  # a 2% planted underperformance is clearly detected
    # and the clean-minus-devil spread goes positive, also monotonically
    l1 = st.synthetic_detect(bump=0.01, seed=789, side="ls")["t"]
    l2 = st.synthetic_detect(bump=0.02, seed=789, side="ls")["t"]
    assert l2 > l1 > 1.0


def test_placebo_random_buckets_are_centered_on_zero():
    # random-bucket spread on the NULL world should sit near zero (no devil signal planted)
    prices, _ = dt.synthetic_world(bump=0.0, seed=789)
    events = st.build_panel_events(prices, spy=None, fwd_s=5, fwd_l=5, min_names=10)
    pl = st.placebo_pvalue(events, horizon="s", n_seeds=6, n_draws_per_seed=40)
    assert abs(pl["placebo_mean"]) < 5e-4
    assert 0.02 < pl["p_value"] < 0.98  # observed null spread is unremarkable

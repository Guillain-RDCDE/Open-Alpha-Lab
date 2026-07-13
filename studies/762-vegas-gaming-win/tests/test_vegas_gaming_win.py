"""Offline, deterministic tests for Study 762 — Vegas-Gaming-Win.

No network: everything runs on the hardcoded GGR reconstruction and the fixed-seed synthetic
control. Asserts the pure-function machinery behaves (no look-ahead leakage, faithful
positive control) without touching yfinance or the price cache.

    pytest -q studies/762-vegas-gaming-win/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vegas_gaming_win import data, strategy as st


def test_ggr_series_is_clean_and_seasonal():
    g = data.ggr_series()
    # monthly, sorted, no NaN, no trailing unpublished (zero) bar at the very end
    assert g.index.is_monotonic_increasing
    assert not g.isna().any()
    assert g.iloc[-1] > 0
    # the COVID closure is preserved as a legitimate ~0 revenue print, not dropped
    apr2020 = g.loc["2020-04-30"]
    assert apr2020 == 0.0
    # aggregate calibration: a pre-COVID year sums near the published ~$6.6B Strip total
    y2019 = g.loc["2019"].sum()
    assert 6000 < y2019 < 7200


def test_ttm_momentum_needs_a_full_year():
    g = data.synthetic_ggr(n_months=60, edge=0.0)
    ttm = st.ggr_ttm(g)
    # first 11 months have no full trailing-12 window
    assert ttm.iloc[:11].isna().all()
    assert not np.isnan(ttm.iloc[11])


def test_no_lookahead_in_forward_returns():
    # forward return at the last row must be NaN (its future is unknown) — a shift guard
    g = data.synthetic_ggr(n_months=48, edge=0.0)
    fwd = st.forward_returns(g, months=3, lag=1)
    assert np.isnan(fwd.iloc[-1])
    assert np.isnan(fwd.iloc[-2])


def test_synthetic_control_no_false_positive():
    # with NO planted link, the Welch t must stay well under the |t|=2 bar
    syn = data.synthetic_ggr(n_months=360, edge=0.0, seed=762)
    s = st.summarize(syn, 1, k=3)
    assert abs(s["t"]) < 2.0


def test_synthetic_control_recovers_planted_edge():
    # a large planted bullish link must light the detector up (t well past +2)
    syn = data.synthetic_ggr(n_months=360, edge=0.05, seed=762)
    s = st.summarize(syn, 1, k=3)
    assert s["t"] > 2.0
    assert s["rising_mean"] > s["base_mean"]


def test_welch_t_matches_hand_formula():
    rng = np.random.default_rng(0)
    a = rng.normal(0.02, 0.1, 200)
    b = rng.normal(0.00, 0.1, 500)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    assert abs(st.welch_t(a, b) - (a.mean() - b.mean()) / se) < 1e-9

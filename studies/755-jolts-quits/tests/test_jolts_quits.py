"""Fully-offline, deterministic tests for Study 755 — JOLTS-Quits.

No network: everything runs on the synthetic control and pure functions. Asserts the
engine's positive control behaves (a planted quits->returns link is recovered; a no-link
series is not manufactured into bearish significance) and that the signal/momentum
plumbing is internally consistent.

    pytest -q studies/755-jolts-quits/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jolts_quits import data, strategy as st


def test_synthetic_null_no_bearish_false_positive():
    """edge=0: the FALLING-set forward mean must NOT be significantly *below* base."""
    syn = data.synthetic_quits(n_months=300, edge=0.0, seed=755)
    s = st.summarize(syn, 1, k=3, lag=1)
    # one-sided bearish placebo p should be far from significant (not <= 0.05)
    assert s["p_placebo"] > 0.05
    # and the Welch t is not a strong negative
    assert s["t"] > -2.0


def test_synthetic_planted_edge_lights_up():
    """A large planted falling-quits -> lower-return link must be detected (t well < -2)."""
    syn = data.synthetic_quits(n_months=300, edge=0.04, seed=755)
    s = st.summarize(syn, 1, k=3, lag=1)
    assert s["t"] < -2.0
    assert s["p_placebo"] < 0.05
    assert s["falling_mean"] < s["base_mean"]


def test_momentum_and_mask_consistent():
    """FALLING mask == (quits momentum < 0), and momentum is q_t - q_{t-k}."""
    syn = data.synthetic_quits(n_months=120, edge=0.0, seed=1)
    mom = st.quits_momentum(syn, k=3)
    mask = st.falling_mask(syn, k=3)
    both = mom.notna()
    assert (mask[both] == (mom[both] < 0)).all()
    # spot-check the identity at a concrete index
    q = syn["quits"]
    i = 50
    assert abs(mom.iloc[i] - (q.iloc[i] - q.iloc[i - 3])) < 1e-9


def test_quits_series_drops_partial_and_is_sorted():
    """The hardcoded snapshot loader drops not-yet-published (zero) months and sorts."""
    q = data.quits_series()
    assert q.is_monotonic_increasing is False or q.index.is_monotonic_increasing
    assert q.index.is_monotonic_increasing
    assert (q > 0).all()
    # JOLTS begins Dec 2000; last pinned print is May 2026 in this snapshot
    assert q.index.min().year == 2000 and q.index.min().month == 12
    assert q.max() <= 3.5 and q.min() >= 1.0


def test_forward_returns_execution_lag_no_lookahead():
    """forward_returns(lag=L, H) uses entry=t+L, exit=t+L+H (one shift each, no extra)."""
    syn = data.synthetic_quits(n_months=60, edge=0.0, seed=2)
    p = syn["spy"]
    fr = st.forward_returns(syn, months=1, lag=2, price="spy")
    # at index i, return should equal p[i+3]/p[i+2]-1
    i = 10
    expected = p.iloc[i + 3] / p.iloc[i + 2] - 1.0
    assert abs(fr.iloc[i] - expected) < 1e-12
    # the tail must be NaN (horizon overruns the tape) -> no look-ahead fabrication
    assert np.isnan(fr.iloc[-1])

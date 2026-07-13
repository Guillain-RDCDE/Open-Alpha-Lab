"""Fully offline, deterministic tests for Study 758 — TSA-Throughput.

No network: the TSA tape is hardcoded and always available, and the synthetic control is
seed-fixed. These assert pure-function properties of the engine, not live-data numbers.

    pytest -q studies/758-tsa-throughput/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tsa_throughput import data, strategy as st


def test_tsa_series_no_partial_bar_and_covid_trough():
    """The hardcoded TSA tape drops in-progress (zero) months and keeps the COVID trough."""
    s = data.tsa_series()
    assert (s > 0).all()                       # no zero/partial bars survive
    assert s.index.is_monotonic_increasing
    # famous April-2020 collapse present and is the minimum
    assert abs(s.min() - 0.11) < 1e-9
    assert s.idxmin().year == 2020 and s.idxmin().month == 4


def test_momentum_and_forward_return_lag_alignment():
    """One-month execution lag: signal at t is read against the return entered at t+1."""
    syn = data.synthetic_tsa(n_months=60, edge=0.0, seed=1)
    mom = st.tsa_momentum(syn, k=12)
    assert mom.iloc[:12].isna().all()          # YoY needs 12 months of lead-in
    fwd = st.forward_returns(syn, 1, col="basket", lag=1)
    # forward return at t equals basket[t+2]/basket[t+1]-1 (entry t+1, exit t+2)
    p = syn["basket"]
    expect = p.iloc[3] / p.iloc[2] - 1.0
    assert abs(fwd.iloc[1] - expect) < 1e-12


def test_synthetic_control_null_is_quiet_and_edge_lights_up():
    """edge=0 must not manufacture significance; a large planted edge must clear |t|>=2."""
    s0 = st.summarize(data.synthetic_tsa(n_months=240, edge=0.0, seed=758), 1, k=12)
    s1 = st.summarize(data.synthetic_tsa(n_months=240, edge=0.08, seed=758), 1, k=12)
    assert abs(s0["t"]) < 1.0                   # no false positive on the null
    assert s1["t"] > 2.0                        # planted edge recovered
    assert s1["accel_mean"] > s0["accel_mean"]


def test_welch_t_matches_manual_formula():
    """welch_t is the textbook unequal-variance statistic."""
    a = np.array([0.02, 0.03, -0.01, 0.05, 0.00])
    b = np.array([0.00, 0.01, 0.00, -0.02, 0.01, 0.02])
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    assert abs(st.welch_t(a, b) - (a.mean() - b.mean()) / se) < 1e-12


def test_overlay_costs_never_increase_return():
    """Net-of-cost overlay return is <= gross (costs and borrow only subtract)."""
    syn = data.synthetic_tsa(n_months=180, edge=0.0, seed=3)
    o = st.timing_overlay(syn, cost_bps=10.0)
    assert o["overlay_net_mean"] <= o["overlay_gross_mean"] + 1e-12
    assert 0.0 <= o["exposure"] <= 1.0

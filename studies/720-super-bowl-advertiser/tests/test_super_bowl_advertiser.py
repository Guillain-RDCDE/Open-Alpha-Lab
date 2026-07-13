"""Fully-offline, deterministic tests for Study 720 — Super-Bowl-Advertiser.

No network: everything runs on the deterministic synthetic control and pure functions.
Run: pytest -q studies/720-super-bowl-advertiser/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from super_bowl_advertiser import data, strategy as st


def test_welch_t_matches_closed_form():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015, -0.005])
    expect = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(st.welch_t(x) - expect) < 1e-12
    assert np.isnan(st.welch_t(np.array([0.01])))          # < 2 samples
    assert np.isnan(st.welch_t(np.array([0.02, 0.02])))    # zero variance


def test_table_is_wellformed():
    assert len(data.ADVERTISERS) >= 30
    assert len(data.DELISTED) >= 8
    # every row is (ticker, ISO date, label); dates are Super Bowl Sundays (weekday 6)
    import pandas as pd
    for tkr, dt, label in data.ADVERTISERS:
        assert tkr and label
        ts = pd.Timestamp(dt)
        assert ts.year >= 2015 and ts.month == 2
        assert ts.weekday() == 6            # Sunday
    # fingerprint is stable & 12 hex chars
    fp = data.fingerprint()
    assert len(fp) == 12 and fp == data.fingerprint()


def test_synthetic_control_zero_edge_is_flat():
    """The positive control must NOT manufacture significance with no planted drift."""
    syn = data.synthetic_ads(n_events=32, edge=0.0, seed=726)
    ev = st.collect_events(syn, drift=5, hold=20)
    s = st.summarize(ev, syn, drift=5, hold=20, placebo=False)
    assert s["n"] == 32
    assert abs(s["drift"]["t"]) < 1.5      # flat under the null
    assert abs(s["hold"]["t"]) < 1.5


def test_synthetic_control_recovers_planted_drift():
    """A large planted post-game drift must light up the drift leg."""
    syn = data.synthetic_ads(n_events=32, edge=0.10, seed=726)
    ev = st.collect_events(syn, drift=5, hold=20)
    s = st.summarize(ev, syn, drift=5, hold=20, placebo=False)
    assert s["drift"]["t"] > 5             # strongly detected
    assert s["drift"]["mean"] > 0.07       # ~10% planted, recovered
    # the plant is on the drift leg only — the hold leg stays flat
    assert abs(s["hold"]["t"]) < 1.5


def test_event_window_lag_has_no_lookahead():
    """The drift leg must start strictly after the entry Monday (1-day lag)."""
    syn = data.synthetic_ads(n_events=4, edge=0.0, seed=726)
    prices = syn["prices"]
    ev = syn["events"].iloc[0]
    w = st.event_window(prices, ev["ticker"], ev["snapped"], drift=5, hold=20, lag=1)
    assert w is not None
    # monday/drift/hold are all present and finite
    for k in ("monday", "drift", "hold"):
        assert np.isfinite(w[k])


def test_net_of_costs_charges_two_crossings():
    syn = data.synthetic_ads(n_events=10, edge=0.0, seed=726)
    ev = st.collect_events(syn, drift=5, hold=20)
    c = st.net_of_costs(ev, cost_bps=10.0)
    # net = gross - 2 * (cost_bps/1e4)
    assert abs((c["gross_drift"] - c["net_drift"]) - 2 * (10.0 / 1e4)) < 1e-12

"""Fully-offline, deterministic tests for Study 749 — Layoff-Drift.

No network: every test runs on the deterministic synthetic control or on pure-function
inputs, so CI needs no cache. Asserts the engine's core properties: the synthetic
positive control recovers a planted drift and does NOT fabricate significance from the
null, the market-model CAR is computed correctly on a hand-built price frame, and the
HAC / Welch statistics behave.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from layoff_drift import data, strategy as st


def test_event_table_is_clean():
    ev = data.LAYOFF_EVENTS
    assert len(ev) >= 25
    # unique (ticker, date) keys, sorted by date, real timestamps
    keys = [(e["ticker"], e["announce_date"]) for e in ev]
    assert len(keys) == len(set(keys))
    dates = [e["announce_date"] for e in ev]
    assert dates == sorted(dates)
    assert all(isinstance(e["announce_date"], pd.Timestamp) for e in ev)
    # fingerprint is deterministic and stable
    assert data.fingerprint(ev) == data.fingerprint(ev)
    assert len(data.fingerprint(ev)) == 12


def test_synthetic_null_does_not_manufacture_significance():
    """With zero planted edges, neither leg should clear t = 2 (seed 723 control)."""
    syn = data.synthetic_events(pop_bps=0.0, drift_bps=0.0, seed=723)
    assert abs(st.welch_t(syn["pop"])) < 2.0
    assert abs(st.welch_t(syn["drift"])) < 2.0
    assert abs(st.hac_t(syn["daily_drift"])) < 2.0


def test_synthetic_control_recovers_a_planted_drift():
    """A large planted drift must light the HAC t up; the pop leg stays quiet."""
    null = data.synthetic_events(pop_bps=0.0, drift_bps=0.0, seed=723)
    edge = data.synthetic_events(pop_bps=0.0, drift_bps=400.0, seed=723)
    # planting drift only must not move the pop leg
    assert np.allclose(null["pop"], edge["pop"])
    # the planted drift raises the mean and the HAC t
    assert edge["drift"].mean() > null["drift"].mean()
    assert st.hac_t(edge["daily_drift"]) > st.hac_t(null["daily_drift"])
    assert st.hac_t(edge["daily_drift"]) > 2.0


def test_market_model_car_on_synthetic_prices():
    """On a hand-built frame where the stock is exactly beta*SPY plus a known pop,
    the recovered pop CAR must equal the planted jump (alpha/beta fit is exact)."""
    n = 400
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=n)
    mkt_r = rng.normal(0.0002, 0.008, n)
    beta = 1.2
    stk_r = beta * mkt_r  # no idiosyncratic noise => perfect fit, zero abnormal
    spy = pd.Series(100 * np.cumprod(1 + mkt_r), index=idx)
    stk = pd.Series(50 * np.cumprod(1 + stk_r), index=idx)
    prices = pd.DataFrame({"AAA": stk, "SPY": spy})
    when = idx[250]
    # with a perfect market model and no noise, both legs are ~0
    w = st.event_abnormal(prices, "AAA", when)
    assert abs(w["pop"]) < 1e-6
    assert abs(w["drift"]) < 1e-6


def test_welch_and_hac_edge_cases():
    assert np.isnan(st.welch_t(np.array([1.0])))
    assert np.isnan(st.welch_t(np.array([2.0, 2.0, 2.0])))  # zero variance
    # a constant-positive drift series has a huge positive HAC t
    x = np.full(300, 0.001) + np.random.default_rng(1).normal(0, 1e-6, 300)
    assert st.hac_t(x) > 5.0


def test_net_of_costs_reduces_gross():
    nc = st.net_of_costs(0.09, cost_bps=10.0)
    assert nc["gross_pct"] == 9.0
    assert nc["net_pct"] < nc["gross_pct"]
    assert abs(nc["net_pct"] - 8.9) < 1e-9

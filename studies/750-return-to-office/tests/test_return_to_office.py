"""Offline, deterministic tests for Study 750 — Return-to-Office.

No network: everything runs on the deterministic synthetic control and on tiny synthetic
frames. Asserts the event-study machinery is faithful (recovers a planted edge, does NOT
manufacture one from noise) and that the pure helpers behave.

    pytest -q studies/750-return-to-office/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from return_to_office import data, strategy as st


def test_event_table_shape():
    ev = data.RTO_EVENTS
    assert len(ev) == 26
    n_strict = sum(e["strict"] for e in ev)
    assert n_strict == 18 and len(ev) - n_strict == 8
    # dates strictly sorted, all Timestamps
    dts = [e["date"] for e in ev]
    assert dts == sorted(dts)
    assert all(isinstance(e["date"], pd.Timestamp) for e in ev)


def test_fingerprint_is_stable():
    # deterministic content fingerprint of the (dated) event table
    assert data.fingerprint(data.RTO_EVENTS) == "69ca30fe1b8b"


def test_welch_t_matches_hand_value():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    # mean 2.5, sd(ddof=1) ~1.290994, se ~0.645497, t ~3.8730
    assert abs(st.welch_t(x) - 3.872983) < 1e-4
    assert np.isnan(st.welch_t(np.array([1.0])))  # too small


def test_basket_returns_equal_weight():
    idx = pd.date_range("2020-01-01", periods=4, freq="D")
    prices = pd.DataFrame({
        "SLG": [10, 11, 11, 12],     # +10%, 0, +9.0909%
        "BXP": [20, 20, 22, 22],     #  0, +10%, 0
        "SPY": [100, 101, 102, 103],
    }, index=idx)
    br = st.basket_returns(prices, ["SLG", "BXP"])
    # day 2: mean(+10%, 0) = 5%
    assert abs(br.iloc[0] - 0.05) < 1e-9
    # day 3: mean(0, +10%) = 5%
    assert abs(br.iloc[1] - 0.05) < 1e-9


def test_synthetic_zero_edge_no_false_strict_hybrid_gap():
    # With NO planted edge, the believers' object (strict - hybrid) must not clear t=2.
    syn = data.synthetic_events(car_bps=0.0, seed=750)
    diff_t = st.welch_t(syn["strict_car"], syn["hybrid_car"])
    assert abs(diff_t) < 2.0


def test_synthetic_large_edge_lights_up():
    # A large planted strict-bucket edge must be recovered with a big strict-hybrid t.
    syn = data.synthetic_events(car_bps=400.0, seed=750)
    diff_t = st.welch_t(syn["strict_car"], syn["hybrid_car"])
    assert diff_t > 4.0
    assert st.summarize_bucket(syn["strict_car"])["t"] > 4.0


def test_placebo_pvalue_bounds():
    null = np.linspace(-1, 1, 1001)
    # an observed mean far in the tail -> small p
    assert st.placebo_pvalue(5.0, null) < 0.05
    # an observed mean at the center -> large p
    assert st.placebo_pvalue(0.0, null) > 0.9


def test_net_of_costs_charges_once():
    r = st.net_of_costs(0.01, cost_bps=10.0)
    assert abs(r["gross_pct"] - 1.0) < 1e-9
    assert abs(r["net_pct"] - 0.9) < 1e-9  # 1.0% - 0.10%

"""Offline, deterministic tests for Study 746 — HQ-Relocation.

No network: every test runs on the deterministic synthetic control or on pure functions.
The two assertions that carry the study are the **positive control** (a large planted
tax-bucket edge must light up) and the **negative control** (a couple-dozen noisy events
must NOT manufacture significance).

    pytest -q studies/746-hq-relocation/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hq_relocation import data, strategy as st


def test_table_is_wellformed():
    assert len(data.HQ_MOVES) == 20
    assert all(k in data.HQ_MOVES[0] for k in ("ticker", "announce_date", "tax"))
    n_tax = sum(e["tax"] for e in data.HQ_MOVES)
    assert n_tax == 14 and len(data.HQ_MOVES) - n_tax == 6
    # deterministic content fingerprint of the frozen table
    assert data.fingerprint(data.HQ_MOVES) == "3fdda93c2d9a"


def test_welch_t_basics():
    # zero-mean-ish sample -> small t; a clearly shifted sample -> large t
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 500)
    assert abs(st.welch_t(x)) < 3
    assert st.welch_t(x + 5.0) > 10
    # two-sample: identical distributions -> small |t|
    y = rng.normal(0.0, 1.0, 500)
    assert abs(st.welch_t(x, y)) < 3


def test_synthetic_negative_control_is_flat():
    """No planted edge: the tax-minus-other test must stay well below t = 2."""
    syn = data.synthetic_events(car_bps=0.0, seed=746)
    diff_t = st.welch_t(syn["tax_car"], syn["other_car"])
    assert abs(diff_t) < 2.0
    assert abs(st.summarize_bucket(syn["tax_car"])["t"]) < 2.0


def test_synthetic_positive_control_lights_up():
    """A large planted tax-bucket edge must be recovered at t >> 2."""
    syn = data.synthetic_events(car_bps=500.0, seed=746)
    assert st.summarize_bucket(syn["tax_car"])["t"] > 3.0
    assert st.welch_t(syn["tax_car"], syn["other_car"]) > 2.5


def test_placebo_pvalue_is_a_probability():
    null = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    p = st.placebo_pvalue(0.0, null)
    assert 0.0 <= p <= 1.0
    # an extreme observation vs a tight null -> small p
    assert st.placebo_pvalue(100.0, null) <= 0.5


def test_net_of_costs_charges_once():
    d = st.net_of_costs(0.02, cost_bps=10.0)
    assert abs(d["gross_pct"] - 2.0) < 1e-9
    assert abs(d["net_pct"] - (2.0 - 0.10)) < 1e-9

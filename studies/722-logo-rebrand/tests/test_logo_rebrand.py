"""Fully offline, deterministic tests for Study 722 — Logo-Rebrand.

No network: everything runs on the deterministic synthetic control and on pure-function
inference. Confirms (a) the engine is faithful — it recovers a large *planted* renewal drift
and does NOT manufacture significance from noise — and (b) the inference primitives behave.

    pytest -q studies/722-logo-rebrand/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logo_rebrand import data, strategy as st


def test_welch_t_basic():
    # Too-small sample -> NaN by contract (len < 2).
    assert np.isnan(st.welch_t(np.array([0.05])))
    # Exactly-zero variance -> NaN by contract (se == 0).
    assert np.isnan(st.welch_t(np.zeros(3)))
    # A clearly-positive noisy sample clears t=2; a zero-mean one does not.
    rng = np.random.default_rng(0)
    pos = rng.normal(0.10, 0.02, 200)
    zero = rng.normal(0.0, 0.02, 200)
    assert st.welch_t(pos) > 2
    assert abs(st.welch_t(zero)) < 2


def test_synthetic_null_does_not_manufacture_significance():
    """edge=0: the drift leg must NOT reach |t| >= 2 (a couple-dozen noisy events)."""
    syn = data.synthetic_rebrands(n_events=26, edge=0.0, seed=722)
    ev = st.collect_events(syn, announce=5, drift=120)
    s = st.summarize(ev, syn, announce=5, drift=120, placebo=False)
    assert s["n"] == 26
    assert abs(s["drift"]["t"]) < 2.0
    assert abs(s["announce"]["t"]) < 2.0


def test_synthetic_planted_drift_lights_up():
    """A large planted renewal drift must be recovered with drift t well above 2."""
    syn = data.synthetic_rebrands(n_events=26, edge=0.30, seed=722)
    ev = st.collect_events(syn, announce=5, drift=120)
    s = st.summarize(ev, syn, announce=5, drift=120, placebo=False)
    assert s["drift"]["mean"] > 0.15          # ~+34% planted, recovered
    assert s["drift"]["t"] > 3.0


def test_synthetic_is_deterministic():
    a = st.collect_events(data.synthetic_rebrands(edge=0.30, seed=722))["drift"].values
    b = st.collect_events(data.synthetic_rebrands(edge=0.30, seed=722))["drift"].values
    assert np.allclose(a, b)


def test_net_of_costs_signs_and_penalty():
    """Costs must strictly reduce the hold P&L, and the hold = announce + drift identity holds."""
    syn = data.synthetic_rebrands(n_events=26, edge=0.30, seed=722)
    ev = st.collect_events(syn, announce=5, drift=120)
    c = st.net_of_costs(ev, cost_bps=10.0)
    assert c["net_hold"] < c["gross_hold"]
    assert np.isclose(c["gross_hold"], c["gross_announce"] + c["gross_drift"], atol=1e-9)


def test_table_shapes():
    """The hardcoded table is well-formed: 3 valid kinds, non-empty DELISTED caveat list."""
    kinds = {k for _, _, k, _ in data.REBRANDS}
    assert kinds <= {"name", "identity", "logo"}
    assert len(data.REBRANDS) >= 24
    assert len(data.DELISTED) >= 5

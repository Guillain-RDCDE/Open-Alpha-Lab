"""Offline, deterministic tests for Study 757 — Cass-Freight.

Pure-function checks on the synthetic control and the hardcoded proxy — NO network. The
synthetic positive control is the anchor: the engine must recover a PLANTED forward edge and
must NOT manufacture significance when the true edge is zero.

    pytest -q studies/757-cass-freight/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cass_freight import data, strategy as st  # noqa: E402


def test_cass_proxy_shape():
    """The hardcoded Cass proxy is a positive monthly level with a well-defined YoY."""
    c = data.cass_index()
    assert (c > 0).all()
    assert c.index.is_monotonic_increasing
    yoy = (c / c.shift(12) - 1.0).dropna()
    # a plausible freight cycle: sometimes expanding, sometimes contracting
    assert 0.2 < (yoy > 0).mean() < 0.8
    # the seasonal is multiplicative & mean~1, so YoY stays in a sane band
    assert yoy.abs().max() < 0.5


def test_synthetic_null_does_not_fire():
    """edge=0: the inference must NOT manufacture significance from a noise signal."""
    syn = data.synthetic(n_months=312, edge=0.0, seed=757)
    s = st.summarize(syn, "spy", 6)
    assert abs(s["t"]) < 2.0
    assert s["p_placebo"] > 0.05


def test_synthetic_planted_edge_is_recovered():
    """A large planted forward edge must light the test up (t >= 2, tiny placebo p)."""
    syn = data.synthetic(n_months=312, edge=0.05, seed=757)
    s = st.summarize(syn, "spy", 6)
    assert s["t"] >= 2.0
    assert s["p_placebo"] < 0.05
    # planted edge => conditional mean strictly above the base rate
    assert s["cond_mean"] > s["base_mean"]


def test_lead_lag_does_not_put_planted_freight_behind_stocks():
    """With a forward-planted edge, freight must NOT read as *lagging* stocks (peak k>=0).

    Contrast with the real tape, whose peak sits at k=-3 (stocks lead freight): a genuinely
    forward signal keeps the argmax at a non-negative lag.
    """
    syn = data.synthetic(n_months=312, edge=0.08, seed=757)
    ll = st.lead_lag_corr(syn, "spy", max_lag=6)
    assert ll["peak_lag"] >= 0  # freight-first / coincident, never stocks-first


def test_welch_t_matches_manual():
    """welch_t is the textbook unequal-variance statistic."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
    m1, m0 = a.mean(), b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    assert abs(st.welch_t(a, b) - (m1 - m0) / se) < 1e-12


def test_execution_lag_is_documented_two_months():
    """The documented publication+execution lag is applied, not a silent extra shift."""
    assert data.DEFAULT_LAG == 2
    syn = data.synthetic(n_months=120, edge=0.0, seed=1)
    # expanding_months at default lag == freight_yoy shifted by exactly DEFAULT_LAG > 0
    exp = st.expanding_months(syn)
    manual = (st.freight_yoy(syn).shift(data.DEFAULT_LAG) > 0)
    assert exp.equals(manual)

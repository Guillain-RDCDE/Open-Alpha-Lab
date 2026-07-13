"""Offline, deterministic tests for Study 756 — Challenger-Layoffs.

No network: everything runs on the hardcoded Challenger snapshot and the fixed-seed
synthetic control. Asserts pure-function behaviour and the positive-control contract
(edge=0 must not fake significance; a large planted edge must light the test up).

    pytest -q studies/756-challenger-layoffs/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from challenger_layoffs import data, strategy as st


def test_cuts_series_is_clean_and_drops_partial_months():
    s = data.cuts_series()
    assert s.index.is_monotonic_increasing
    assert (s > 0).all()                       # zero placeholders dropped
    # COVID record spike present and is the maximum
    assert int(s.max()) == 671
    assert s.idxmax().year == 2020 and s.idxmax().month == 4


def test_spike_signal_is_non_lookahead_and_reasonable_frequency():
    syn = data.synthetic_cuts(n_months=240, edge=0.0, seed=1)
    sig = st.cut_spike(syn)
    # trailing baseline needs history -> earliest months are NaN, never a look-ahead spike
    assert sig.iloc[0] != sig.iloc[0] or np.isnan(sig.iloc[0])  # NaN at t0
    sp = st.spike_mask(syn)
    freq = sp.mean()
    assert 0.2 < freq < 0.7                     # cuts spike a sensible fraction of months


def test_forward_returns_have_one_month_execution_lag():
    syn = data.synthetic_cuts(n_months=60, edge=0.0, seed=2)
    # 1-month forward return with lag=1: enter at t+1, exit at t+2
    fwd = st.forward_returns(syn, months=1, lag=1)
    spy = syn["spy"]
    expected = spy.shift(-2) / spy.shift(-1) - 1.0
    ok = fwd.notna() & expected.notna()
    assert np.allclose(fwd[ok].values, expected[ok].values)


def test_welch_and_hac_agree_in_sign_on_control():
    syn = data.synthetic_cuts(n_months=360, edge=0.04, seed=756)
    s = st.summarize(syn, 1)
    assert s["t"] < 0 and s["hac_t"] < 0        # planted bearish link -> negative both


def test_positive_control_contract():
    # edge = 0 -> no false positive (|t| well under 2)
    syn0 = data.synthetic_cuts(n_months=360, edge=0.0, seed=756)
    s0 = st.summarize(syn0, 1)
    assert abs(s0["t"]) < 2.0
    assert abs(s0["hac_t"]) < 2.0
    # large planted edge -> the test lights up (|t| >> 2)
    syn1 = data.synthetic_cuts(n_months=360, edge=0.04, seed=756)
    s1 = st.summarize(syn1, 1)
    assert s1["t"] < -3.0
    assert s1["hac_t"] < -3.0
    assert s1["p_placebo"] < 0.05


def test_placebo_pvalue_is_a_probability():
    syn = data.synthetic_cuts(n_months=240, edge=0.0, seed=3)
    p = st.placebo_pvalue(syn, 3, n_draws=2000)["p_value"]
    assert 0.0 <= p <= 1.0

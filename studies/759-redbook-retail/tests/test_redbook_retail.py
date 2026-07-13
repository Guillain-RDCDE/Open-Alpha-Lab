"""Offline, deterministic tests for Study 759 — Redbook-Retail.

No network: the Redbook proxy table is hardcoded and the synthetic control is fixed-seed.
These assert pure-function behaviour and the two positive-control poles (null must stay
flat, planted edge must light up).

    pytest -q studies/759-redbook-retail/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redbook_retail import data, strategy as st


def test_redbook_series_is_clean_and_dropped_partial():
    s = data.redbook_series()
    assert s.index.is_monotonic_increasing
    assert s.notna().all()
    # in-progress 2026 months flagged 0.0 are dropped -> last kept month is 2026-06
    assert str(s.index.max().date()) == "2026-06-30"
    # values are percent YoY in a plausible band, incl. the negative 2009/2020 troughs
    assert s.min() < 0.0 < s.max()
    assert s.min() > -20.0 and s.max() < 30.0


def test_momentum_and_masks_are_pure():
    syn = data.synthetic_redbook(n_months=120, edge=0.0, seed=1)
    m = st.redbook_momentum(syn, k=3)
    # first k entries are NaN (no 3-month history), rest finite
    assert m.iloc[:3].isna().all()
    assert np.isfinite(m.iloc[3:].values).all()
    accel = st.accel_mask(syn, k=3)
    assert accel.dropna().isin([True, False]).all()


def test_forward_returns_respect_lag_no_lookahead():
    syn = data.synthetic_redbook(n_months=60, edge=0.0, seed=2)
    fwd = st.forward_returns(syn, months=3, lag=1)
    # a 1-month lag + 3-month hold consumes the last 4 observations
    assert fwd.iloc[-4:].isna().all()
    assert np.isfinite(fwd.iloc[:-4].values).all()


def test_relative_return_is_xrt_minus_spy():
    syn = data.synthetic_redbook(n_months=80, edge=0.0, seed=3)
    abs_fwd = st.forward_returns(syn, months=6, lag=1, relative=False)
    rel_fwd = st.forward_returns(syn, months=6, lag=1, relative=True)
    # relative differs from absolute wherever both are defined (SPY leg is nonzero)
    both = abs_fwd.notna() & rel_fwd.notna()
    assert both.sum() > 0
    assert not np.allclose(abs_fwd[both].values, rel_fwd[both].values)


def test_synthetic_control_null_is_flat():
    syn = data.synthetic_redbook(n_months=360, edge=0.0, seed=759)
    s = st.summarize(syn, 1, k=3)
    # no planted link => no false positive
    assert abs(s["t"]) < 2.0
    assert s["p_placebo"] > 0.05


def test_synthetic_control_planted_edge_lights_up():
    syn = data.synthetic_redbook(n_months=360, edge=0.05, seed=759)
    s = st.summarize(syn, 1, k=3)
    # a large planted link must be detected, in the RIGHT (positive) direction
    assert s["t"] > 2.0
    assert s["accel_mean"] > s["base_mean"]
    assert s["p_placebo"] < 0.05


def test_welch_t_matches_hand_computation():
    a = np.array([0.02, 0.03, -0.01, 0.04, 0.00])
    b = np.array([0.01, 0.00, 0.02, -0.02, 0.03, 0.01])
    m1, m0 = a.mean(), b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    assert np.isclose(st.welch_t(a, b), (m1 - m0) / se)

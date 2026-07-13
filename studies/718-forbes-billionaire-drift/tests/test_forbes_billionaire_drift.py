"""Fully offline, deterministic tests for Study 718 — Forbes-Billionaire-Drift.

No network: everything runs on the fixed-seed synthetic control and pure functions.
Guards the two claims the study rests on:
  1. the inference engine does NOT manufacture significance from ~25 ultra-vol events
     when the planted post-list drift is zero (no false positive), but DOES light up on a
     huge planted drift (faithful engine);
  2. the market-model event-study primitives are internally consistent.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forbes_billionaire_drift import data, strategy as st


def test_event_table_wellformed():
    assert len(data.FORBES_EVENTS) >= 20
    for e in data.FORBES_EVENTS:
        assert e["ticker"] and isinstance(e["list_date"], pd.Timestamp)
        assert e["list_year"] in data.LIST_DATES
    # fingerprint is stable & deterministic
    assert data.fingerprint(data.FORBES_EVENTS) == data.fingerprint(data.FORBES_EVENTS)
    assert len(data.fingerprint(data.FORBES_EVENTS)) == 12


def test_synthetic_null_is_not_significant():
    """Zero planted drift -> ~25 high-vol events must NOT reach |t| >= 2 (no false positive)."""
    syn = data.synthetic_events(drift_bps=0.0, seed=718)
    t = st.welch_t(syn["post_car"])
    assert abs(t) < 2.0, f"null post-list t should be < 2, got {t:.2f}"


def test_synthetic_large_edge_lights_up():
    """A huge planted post-list drift must be detected (faithful engine / positive control)."""
    syn = data.synthetic_events(drift_bps=3000.0, seed=718)
    t = st.welch_t(syn["post_car"])
    assert t > 3.0, f"planted +3000bps should give t > 3, got {t:.2f}"
    # and the mean must move in the planted direction
    assert syn["post_car"].mean() > 0.15


def test_synthetic_is_deterministic():
    a = data.synthetic_events(drift_bps=1500.0, seed=718)["post_car"]
    b = data.synthetic_events(drift_bps=1500.0, seed=718)["post_car"]
    assert np.allclose(a, b)


def test_welch_and_placebo_pvalue_math():
    # constant-positive sample: t is +inf-ish (se=0 guarded -> nan), so use a spread sample
    x = np.array([0.01, 0.02, -0.005, 0.03, 0.0, 0.015])
    assert st.welch_t(x) > 0
    # two-sided placebo p of the null's own mean is ~1 (dead centre)
    null = np.random.default_rng(0).normal(0.0, 0.02, 5000)
    p = st.placebo_pvalue(float(null.mean()), null)
    assert 0.8 <= p <= 1.0


def test_raw_excess_matches_manual_on_toy_frame():
    """raw_excess_car sums (r_stock - r_bench) over the window; check on a tiny frame."""
    idx = pd.bdate_range("2020-01-01", periods=80)
    # SPY flat 0.1%/day; stock 0.3%/day -> excess 0.2%/day
    spy = pd.Series(100.0 * (1.001 ** np.arange(80)), index=idx)
    stk = pd.Series(50.0 * (1.003 ** np.arange(80)), index=idx)
    prices = pd.DataFrame({"XYZ": stk, "SPY": spy})
    ld = idx[60]
    car = st.raw_excess_car(prices, "XYZ", ld, window=(1, 5))
    # 5 days of ~0.2% excess -> ~1.0%
    assert 0.008 < car < 0.012


def test_windows_are_ordered():
    # PRE is entirely before the list; ANNOUNCE straddles the list days; POST is the
    # holdable window a trader entering *after* publication holds (starts at +1) and is
    # the default. ANNOUNCE [0,+2] and POST [+1,+63] overlap by design.
    assert st.PRE_WINDOW[1] < 0 <= st.ANNOUNCE_WINDOW[0]
    assert st.POST_WINDOW[0] == 1 and st.POST_WINDOW[1] > st.ANNOUNCE_WINDOW[1]
    assert st.POST_WINDOW == st.DEFAULT_WINDOW

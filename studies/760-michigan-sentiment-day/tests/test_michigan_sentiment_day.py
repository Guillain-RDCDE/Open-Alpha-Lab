"""Offline, deterministic tests for Study 760 — Michigan-Sentiment-Day.

No network: everything runs on the hardcoded sentiment snapshot and the deterministic
synthetic control. Asserts pure-function properties and the positive-control behaviour.

    pytest -q studies/760-michigan-sentiment-day/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from michigan_sentiment_day import data, strategy as st


def test_sentiment_snapshot_shape():
    s = data.sentiment_series()
    # monthly, sorted, no in-progress zeros, sane index of consumer sentiment values
    assert s.is_monotonic_increasing is False or s.index.is_monotonic_increasing
    assert (s > 20).all() and (s < 130).all()
    assert s.index.is_monotonic_increasing
    # the all-time-low print of June 2022 is present and is the minimum
    assert abs(s.loc["2022-06-30"] - 50.0) < 1e-9
    assert s.min() >= 49.0


def test_second_friday():
    # a known anchor: the second Friday of 2020-03 is 2020-03-13
    assert data.second_friday(2020, 3).strftime("%Y-%m-%d") == "2020-03-13"
    # always a Friday
    for y in (1999, 2010, 2026):
        for m in (1, 6, 12):
            assert data.second_friday(y, m).weekday() == 4


def test_welch_t_symmetry_and_sign():
    a = np.array([0.02, 0.03, 0.01, 0.04, 0.02])
    b = np.array([0.00, -0.01, 0.01, 0.00, -0.02])
    t = st.welch_t(a, b)
    assert t > 0                                   # a has the higher mean
    assert abs(st.welch_t(a, b) + st.welch_t(b, a)) < 1e-9   # antisymmetric
    assert np.isnan(st.welch_t(np.array([1.0]), b))          # too few points


def test_regime_masks_no_lookahead():
    syn = data.synthetic(n_months=200, edge=0.0, seed=1)
    m = st.regime_mask(syn["frame"])
    # low_rising is the intersection of low and rising
    assert (m["low_rising"] == (m["low"] & m["rising"])).all()
    # warmup: the first 35 months have no percentile => not "low"
    assert not m["low"].iloc[:35].any()


def test_forward_returns_align():
    syn = data.synthetic(n_months=120, edge=0.0, seed=2)
    f = syn["frame"]
    fwd = st.forward_returns(f, 1)
    manual = f["spy"].shift(-1) / f["spy"] - 1.0
    assert np.allclose(fwd.dropna().values, manual.dropna().values)


def test_synthetic_positive_control():
    # edge=0 must NOT manufacture significance; a large edge MUST light up the block boot.
    null = st.summarize_regime(data.synthetic(n_months=396, edge=0.0, seed=760)["frame"],
                               12, min_periods=12)
    live = st.summarize_regime(data.synthetic(n_months=396, edge=0.10, seed=760)["frame"],
                               12, min_periods=12)
    assert abs(null["t_low_rising"]) < 1.5          # no false positive on the null
    assert null["p_block"] > 0.15
    assert live["t_low_rising"] > 4.0               # planted edge recovered
    assert live["p_block"] < 0.05                   # bootstrap fires when the edge is real


def test_overlay_underperforms_on_null():
    # on a no-link synthetic, a rare long-only overlay cannot beat buy-and-hold
    o = st.timing_overlay(data.synthetic(n_months=396, edge=0.0, seed=3)["frame"])
    assert o["exposure"] < 0.6
    assert o["overlay_net_mean"] <= o["bh_mean"] + 1e-9

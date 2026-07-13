"""Fully offline, deterministic tests for Study 754 — Beige-Book-Tone.

No network: the synthetic control is generated in-process, and the pure-function event-study
math is checked on hand-built inputs. Run:  pytest -q studies/754-beige-book-tone/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from beige_book_tone import data, strategy as st


def _toy():
    """A tiny deterministic tape: 40 business days, prices +1% on odd days, 3 releases."""
    idx = pd.bdate_range("2020-01-01", periods=40)
    # geometric-ish path so forward returns are exact and easy to reason about
    price = pd.Series(100.0 * (1.005 ** np.arange(40)), index=idx, name="SPY")
    rel = pd.DataFrame({"tone": [0.5, -0.5, 0.5]},
                       index=[idx[5], idx[15], idx[25]])
    return rel, price


def test_release_calendar_shape():
    rel = data.releases()
    assert len(rel) == 14 * 8            # 2011..2024, eight books/year
    assert list(rel.columns) == ["tone"]
    assert rel.index.is_monotonic_increasing
    # every release date is snapped to a Wednesday
    assert set(rel.index.weekday) == {2}


def test_forward_return_is_close_to_close_from_release_day():
    rel, price = _toy()
    fwd = st.event_forward_returns(price, rel.index, h=3)
    # entry at the release-day close (index 5), exit 3 days later (index 8)
    expected = price.iloc[8] / price.iloc[5] - 1.0
    assert np.isclose(fwd.iloc[0], expected)
    assert not fwd.isna().any()          # all windows fit inside the 40-day tape


def test_holiday_resolves_forward_no_lookahead():
    rel, price = _toy()
    # a release on a weekend must map to the NEXT trading day, never an earlier one
    sat = pd.Timestamp("2020-01-11")     # a Saturday inside the span
    fwd = st.event_forward_returns(price, pd.DatetimeIndex([sat]), h=1)
    entry_pos = price.index.searchsorted(sat, side="left")
    assert price.index[entry_pos] >= sat
    assert not np.isnan(fwd.iloc[0])


def test_welch_t_zero_on_identical_samples():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    assert st.welch_t(x, x) == 0.0


def test_split_uses_strict_threshold():
    rel, price = _toy()
    pos, neg, allv = st.split_returns(rel, price, h=2, thresh=0.0)
    assert len(pos) == 2 and len(neg) == 1 and len(allv) == 3


def test_synthetic_null_is_insignificant_and_edge_lights_up():
    # edge=0 must NOT manufacture significance; a large planted edge MUST.
    rel0, spy0 = data.synthetic(n_years=14, edge=0.0, seed=754)
    s0 = st.summarize(rel0, spy0, 5)
    assert abs(s0["t"]) < 2.0

    rel1, spy1 = data.synthetic(n_years=14, edge=0.004, seed=754)
    s1 = st.summarize(rel1, spy1, 5)
    r1 = st.tone_drift_regression(rel1, spy1, 5)
    assert s1["t"] > 2.0
    assert r1["t_hac"] > 2.0
    assert r1["beta"] > 0.0


def test_regression_recovers_planted_positive_slope_sign():
    rel1, spy1 = data.synthetic(n_years=14, edge=0.004, seed=754)
    r = st.tone_drift_regression(rel1, spy1, 5)
    assert r["corr"] > 0.3               # a strong planted link shows a clear positive corr


def test_overlay_reports_gross_ge_net():
    rel1, spy1 = data.synthetic(n_years=14, edge=0.004, seed=754)
    o = st.event_overlay(rel1, spy1, h=5, cost_bps=1.0)
    assert o["per_event_gross"] >= o["per_event_net"]
    assert o["n_trades"] > 0

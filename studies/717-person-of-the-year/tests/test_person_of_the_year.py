"""Offline, deterministic tests for Study 717 — Person-of-the-Year.

No network: everything runs on the synthetic positive control and pure functions. These
guard the two things the study's honesty rests on — (1) the inference engine does NOT
manufacture a "curse" from four events when the true edge is zero, and it DOES recover a
large planted one; (2) the event table and pure helpers behave as documented.

    pytest -q studies/717-person-of-the-year/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from person_of_the_year import data, strategy as st


def test_event_table_is_the_documented_census():
    # Four tradable honorees, sorted by date, unique (ticker, date), fingerprint stable.
    assert [e["ticker"] for e in data.POY_EVENTS] == ["AMZN", "MSFT", "TSLA", "DJT"]
    assert all(e["announce_date"].month == 12 for e in data.POY_EVENTS)
    assert data.fingerprint(data.POY_EVENTS) == "ff8a712ac1d1"
    # the two named-but-untradable business picks are recorded (survivorship note)
    assert {d[0] for d in data._DROPPED} == {"META", "DJT"}


def test_synthetic_control_no_false_curse_at_zero_edge():
    # With NO planted drift, four heavy-tailed names may tilt negative by luck, but the
    # engine must not clear the |t| >= 2 bar — no manufactured curse.
    syn = data.synthetic_events(curse_bps=0.0, seed=717)
    b = st.summarize_bucket(syn["car"])
    assert b["n"] == 4
    assert abs(b["t"]) < 2.0


def test_synthetic_control_recovers_a_large_planted_curse():
    # A large planted decline must light up strongly (and be negative).
    syn = data.synthetic_events(curse_bps=-12000.0, seed=717)
    b = st.summarize_bucket(syn["car"])
    assert b["mean_pct"] < 0
    assert b["t"] < -3.0


def test_synthetic_is_deterministic():
    a = data.synthetic_events(curse_bps=-3000.0, seed=717)["car"]
    b = data.synthetic_events(curse_bps=-3000.0, seed=717)["car"]
    assert np.allclose(a, b)


def test_welch_t_and_placebo_pvalue_pure():
    # Welch t of a constant-shift sample vs 0.
    x = np.array([-0.5, -0.4, -0.6, -0.5])
    assert st.welch_t(x) < 0
    # one-sided-left placebo p: an obs at the far left tail is small.
    null = np.linspace(-1.0, 1.0, 1001)
    assert st.placebo_pvalue(-0.9, null, "left") < 0.10
    assert st.placebo_pvalue(0.0, null, "left") == 0.5 or abs(
        st.placebo_pvalue(0.0, null, "left") - 0.5) < 0.01


def test_net_of_costs_short_pays_borrow():
    # Shorting a stock that fell 50% earns +50% gross; borrow + trade make net a touch less.
    nc = st.net_of_costs(-0.50, horizon_days=252, borrow_ann=0.05, trade_bps=10.0)
    assert abs(nc["gross_pct"] - 50.0) < 1e-9
    assert nc["net_pct"] < nc["gross_pct"]
    assert abs(nc["net_pct"] - (50.0 - 5.0 - 0.1)) < 1e-6

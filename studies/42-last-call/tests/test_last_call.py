"""The synthetic world is deterministic; the turn-of-the-month bump is recovered only when present;
the [-1,+3] window is marked correctly; the tradable rule is in cash off-window and costs only hurt.
All offline on the seeded synthetic world."""

import numpy as np
import pandas as pd

from last_call import data, strategy as st


def test_world_deterministic(premium_world):
    s, truth = premium_world
    s2, _ = data.synthetic_daily(n_years=40, premium_bp=9.0, seed=42)
    assert np.allclose(s.to_numpy(), s2.to_numpy())
    assert truth.has_premium


def test_premium_recovered(premium_world):
    s, _ = premium_world
    d = st.tom_vs_rest(s)
    assert d["tom_bp"] > d["rest_bp"] + 3.0   # the bump shows up
    assert d["welch_t"] > 3.0                 # and is significant


def test_null_has_no_seasonal(null_world):
    s, _ = null_world
    d = st.tom_vs_rest(s)
    assert abs(d["tom_bp"] - d["rest_bp"]) < 3.0  # flat
    assert abs(d["welch_t"]) < 3.0


def test_tom_mask_marks_window():
    """On a clean business-day index, each month marks its last day + the first three of the next."""
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    m = st.tom_mask(idx, last=1, first=3)
    # February 2020: first three business days are 3,4,5 Feb; they must be flagged
    feb_first3 = pd.to_datetime(["2020-02-03", "2020-02-04", "2020-02-05"])
    assert m.loc[feb_first3].all()
    # last business day of January (31 Jan 2020) flagged
    assert m.loc[pd.Timestamp("2020-01-31")]
    # a mid-month day is not
    assert not m.loc[pd.Timestamp("2020-02-18")]


def test_tom_returns_in_cash_off_window_and_costs_hurt(premium_world):
    s, _ = premium_world
    pos = st.tom_mask(s.index).astype(float).shift(1).fillna(0.0)
    r = st.tom_returns(s, cost_bps=0.0)
    assert (r[pos == 0] == 0.0).all()                       # flat when out of the window
    gross = (1 + st.tom_returns(s, cost_bps=0.0)).prod()
    net = (1 + st.tom_returns(s, cost_bps=5.0)).prod()
    assert net < gross

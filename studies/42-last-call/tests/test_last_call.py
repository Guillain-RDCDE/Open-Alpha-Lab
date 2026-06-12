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
    m = st.tom_mask(s.index)
    r = st.tom_returns(s, cost_bps=0.0)
    assert (r[~m.values] == 0.0).all()                      # flat when out of the window
    gross = (1 + st.tom_returns(s, cost_bps=0.0)).prod()
    net = (1 + st.tom_returns(s, cost_bps=5.0)).prod()
    assert net < gross


def test_tom_returns_holds_exactly_the_window_including_day_minus_1():
    """The window is calendar-known ex ante, so the book must hold the masked days themselves —
    in particular day −1 (the last trading day of the month), and NOT day +4."""
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    r = pd.Series(0.01, index=idx)  # 1% every day, so held days are exactly the non-zero ones
    book = st.tom_returns(r, cost_bps=0.0)
    assert book.loc[pd.Timestamp("2020-01-31")] == 0.01     # day −1 is held
    assert book.loc[pd.Timestamp("2020-02-05")] == 0.01     # day +3 is held
    assert book.loc[pd.Timestamp("2020-02-06")] == 0.0      # day +4 is not
    assert book.loc[pd.Timestamp("2020-02-18")] == 0.0      # mid-month is not
    m = st.tom_mask(idx)
    assert (book.to_numpy() != 0).sum() == int(m.sum())     # held days == masked days, no lag


def test_tom_returns_credits_cash_on_off_days(premium_world):
    """With rf set, off-window days earn the cash return instead of zero; the Sharpe race can then
    be run excess-vs-excess without favouring either side."""
    s, _ = premium_world
    m = st.tom_mask(s.index)
    rf_ann = 0.03
    book = st.tom_returns(s, cost_bps=0.0, rf=rf_ann)
    assert np.allclose(book[~m.values], rf_ann / 252.0)     # cash leg credited
    assert np.allclose(book[m.values], s[m.values])         # window leg untouched
    # excess Sharpe strips the credit back out: identical to the zero-cash book's excess Sharpe
    ex_credit = st.summary(book, rf=rf_ann)["sharpe"]
    ex_zero = st.summary(st.tom_returns(s, cost_bps=0.0), rf=0.0)["sharpe"]
    assert abs(ex_credit - ex_zero) < 0.15                  # same book up to the constant rf shift


def test_premium_change_flat_when_seasonal_is_stable(premium_world):
    """A constant injected premium must NOT register as a significant change across the split."""
    s, _ = premium_world
    mid = s.index[len(s) // 2]
    pc = st.tom_premium_change(s, split=str(mid.date()))
    assert abs(pc["t_change"]) < 2.0
    assert pc["premium_pre_bp"] > 3.0 and pc["premium_post_bp"] > 3.0  # the bump shows in both halves

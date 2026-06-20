"""Signals, the backtest engine's invariants, the honest stats, and the study's two
spines: (1) a crack-timer beats buy-and-hold *only* when a genuine lead is planted, and
(2) the predictive regression certifies the lead only when it is real."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crack_spread import data, strategy as st  # noqa: E402


# ---- signals --------------------------------------------------------------
def test_signals_are_binary(coincident):
    frame, _ = coincident
    for sig in (st.level_signal(frame["crack"]), st.momentum_signal(frame["crack"])):
        assert set(np.unique(sig.to_numpy())) <= {0.0, 1.0}
        assert len(sig) == len(frame)


def test_momentum_signal_matches_definition(coincident):
    frame, _ = coincident
    lb = 10
    sig = st.momentum_signal(frame["crack"], lookback=lb)
    expected = (frame["crack"] > frame["crack"].shift(lb)).astype(float).fillna(0.0)
    assert np.array_equal(sig.to_numpy(), expected.to_numpy())


# ---- engine invariants ----------------------------------------------------
def test_one_execution_lag(coincident):
    """The position at the close of t earns the return of t+1 (exactly one shift)."""
    frame, _ = coincident
    pos = pd.Series(0.0, index=frame.index)
    pos.iloc[10] = 1.0                       # long only on bar 10
    book = st.book_returns(pos, frame["stock_ret"], cost_bps=0.0)
    # The only non-cost return earned is on bar 11 (t+1), gross of the switch cost at 0bps.
    earned = book.copy()
    assert abs(earned.iloc[11] - frame["stock_ret"].iloc[11]) < 1e-12
    assert abs(earned.iloc[12]) < 1e-12      # back to flat the next day


def test_cost_only_charged_on_position_change(coincident):
    frame, _ = coincident
    pos = pd.Series(0.0, index=frame.index)
    pos.iloc[5:15] = 1.0                     # one entry, one exit
    gross = st.book_returns(pos, frame["stock_ret"], cost_bps=0.0)
    net = st.book_returns(pos, frame["stock_ret"], cost_bps=10.0)
    diff = (gross - net)
    # Costs charged on exactly two bars (the lagged entry and exit); each = 10bps.
    charged = diff[diff.abs() > 1e-12]
    assert len(charged) == 2
    assert np.allclose(charged.to_numpy(), 10e-4)


def test_cost_monotonically_lowers_active_return(coincident):
    frame, _ = coincident
    pos = st.momentum_signal(frame["crack"], 20)
    actives = [st.summarize_timer(frame, pos, cost_bps=c)["active_ann"] for c in (0.0, 5.0, 10.0)]
    assert actives[0] > actives[1] > actives[2]


def test_buy_and_hold_is_always_long(coincident):
    frame, _ = coincident
    assert np.allclose(st.buy_and_hold(frame["stock_ret"]).to_numpy(),
                       frame["stock_ret"].to_numpy(), equal_nan=True)


# ---- honest stats ---------------------------------------------------------
def test_hac_tstat_on_pure_noise_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert abs(st.hac_tstat(x)) < 3.0        # mean-zero noise: |t| should be small


def test_block_bootstrap_ci_orders(coincident):
    frame, _ = coincident
    lo, hi = st.block_bootstrap_ci(frame["stock_ret"], n_boot=400)
    assert lo <= hi


# ---- the two spines -------------------------------------------------------
def test_predictive_regression_certifies_only_a_real_lead(coincident, leading):
    """Spine #2: lag-1 predictive HAC t clears the bar only with a planted lead."""
    null, _ = coincident
    lead, _ = leading
    t_null = st.predictive_regression(null)["tstat"]
    t_lead = st.predictive_regression(lead)["tstat"]
    assert abs(t_null) < 2.0                 # null: not distinguishable from zero
    assert t_lead > 2.0                      # planted lead: significant


def test_coincident_is_significant_even_in_the_null(coincident):
    """The foil: the same-day relation is strong even with NO predictability — exactly
    the trap the study exposes."""
    null, _ = coincident
    assert st.contemporaneous_regression(null)["tstat"] > 2.0


def test_timer_beats_bh_only_with_a_real_lead(coincident, leading):
    """Spine #1: a 1-day crack-momentum timer adds active return only when a lead exists."""
    null, _ = coincident
    lead, _ = leading
    s_null = st.summarize_timer(null, st.momentum_signal(null["crack"], 1))
    s_lead = st.summarize_timer(lead, st.momentum_signal(lead["crack"], 1))
    assert abs(s_null["active_t"]) < 2.0     # null: no value over buy-and-hold
    assert s_lead["active_t"] > 2.0          # real lead: the timer genuinely adds value


def test_pct_in_market_is_a_fraction(coincident):
    frame, _ = coincident
    s = st.summarize_timer(frame, st.level_signal(frame["crack"]))
    assert 0.0 <= s["pct_in_market"] <= 100.0
    assert s["n_switches"] >= 0

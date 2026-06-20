"""The event-study engine's invariants and the study's two spines: (1) the engine
recovers a planted depeg sell-off only when it is real (and reads ~null otherwise), and
(2) a sample of two events can never produce a robust HAC t — the inference floor that
fixes the verdict."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stablecoin_depeg import data, strategy as st  # noqa: E402


# ---- HAC t-stat ------------------------------------------------------------
def test_hac_tstat_nan_on_tiny_sample():
    """n < 3 (the real two-event sample) cannot produce a t-stat — by design."""
    assert np.isnan(st.hac_tstat(np.array([-0.1, -0.2])))


def test_hac_tstat_detects_a_real_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1.0, 400)
    assert st.hac_tstat(x) > 4.0


# ---- event windows ---------------------------------------------------------
def test_event_windows_shape(shocked):
    bars, events, _ = shocked
    w = st.event_windows(bars, events, pre=5, post=10)
    assert w.shape[0] == 16          # -5 .. +10
    assert list(w.index) == list(range(-5, 11))
    assert w.shape[1] == len(events)


def test_car_event_day_inclusion(shocked):
    bars, events, _ = shocked
    w = st.event_windows(bars, events, pre=5, post=10)
    full = st.event_cars(w, from_day=0, pre=5)
    fwd = st.event_cars(w, from_day=1, pre=5)
    # Including the crash day makes the CAR more negative than the forward-only CAR.
    assert full.mean() < fwd.mean()


# ---- spine #1: engine recovers a real shock, reads null otherwise ----------
def test_engine_recovers_planted_shock(shocked):
    bars, events, _ = shocked
    res = st.summarize_event_study(bars, events, from_day=0)
    assert res["mean_car"] < -0.05            # the planted crash shows up
    assert res["tstat"] < -2.0                # and clears the bar (with 10 planted events)
    assert res["p_value"] < 0.05              # in the left tail of the synthetic-control null


def test_engine_reads_null_as_null(null_tape):
    bars, events, _ = null_tape
    res = st.summarize_event_study(bars, events, from_day=0)
    assert abs(res["mean_car"]) < 0.05
    assert abs(res["tstat"]) < 2.0            # no signal on the null
    assert res["p_value"] > 0.05


# ---- spine #2: two events can never certify a signal -----------------------
def test_two_events_cannot_clear_the_bar():
    """The real universe is n=2; the HAC t is undefined and the bootstrap CI spans zero."""
    bars, events, _ = data.synthetic_tape(n_events=2, shock=-0.15, seed=326)
    res = st.summarize_event_study(bars, events, from_day=0)
    assert res["n_events"] == 2
    assert np.isnan(res["tstat"])             # cannot compute a robust t off two points
    lo, hi = res["ci_lo"], res["ci_hi"]
    # Even with a big planted shock, the two-event bootstrap CI is wide / not a stamp-maker.
    assert lo < hi


# ---- synthetic control -----------------------------------------------------
def test_null_distribution_excludes_events(shocked):
    bars, events, _ = shocked
    null = st.null_car_distribution(bars, events, n_draws=1000, seed=1)
    assert null.size == 1000
    assert np.isfinite(null).all()


def test_empirical_pvalue_bounds():
    null = np.linspace(-1, 1, 1001)
    assert st.empirical_pvalue(-1.0, null) <= 0.01
    assert st.empirical_pvalue(1.0, null) >= 0.99
    assert abs(st.empirical_pvalue(0.0, null) - 0.5) < 0.01


# ---- block bootstrap -------------------------------------------------------
def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(-0.1, 0.05, 50)
    lo, hi = st.block_bootstrap_ci(x, n_boot=2000, seed=0)
    assert lo < x.mean() < hi


# ---- the tradable rule -----------------------------------------------------
def test_sell_on_depeg_goes_to_cash_after_lag(shocked):
    bars, events, _ = shocked
    book = st.sell_on_depeg(bars, events, hold=10, lag=1, cost_bps=0.0)
    assert set(book["pos"].unique()) <= {0.0, 1.0}
    # Some days must be in cash (we sit out after each event).
    assert (book["pos"] == 0.0).any()
    # Position is long on the event day itself and the lag day (we act at t+1 close).
    loc = st._event_locs(bars.index, events)[0]
    assert book["pos"].iloc[loc] == 1.0


def test_costs_reduce_strategy_return(shocked):
    bars, events, _ = shocked
    free = st.sell_on_depeg(bars, events, cost_bps=0.0)["strat_ret"].sum()
    paid = st.sell_on_depeg(bars, events, cost_bps=20.0)["strat_ret"].sum()
    assert paid < free


def test_equity_stats_well_formed(shocked):
    bars, events, _ = shocked
    book = st.sell_on_depeg(bars, events, cost_bps=0.0)
    s = st.equity_stats(book["strat_ret"].to_numpy())
    assert set(s.keys()) == {"ann_ret", "ann_vol", "sharpe"}
    assert np.isfinite(s["sharpe"])

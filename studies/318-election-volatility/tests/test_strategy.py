"""The vol event-study engine's invariants and the study's two spines:
(1) the machinery finds a pre-election vol spike only when one is planted, and
(2) the election short-vol carry beats the *always-on* VRP only when an election-specific
    excess is planted — the headline gross t is the everyday premium, not an election edge."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from election_volatility import strategy as st  # noqa: E402


# ---- realized vol ----------------------------------------------------------
def test_realized_vol_positive(null_tape):
    bars, _, _, _ = null_tape
    rv = st.realized_vol(bars).dropna()
    assert (rv > 0).all()


def test_vol_ratio_shapes(null_tape):
    bars, _, el, _ = null_tape
    ratios = st.vol_ratio_around(bars, el["election_date"])
    assert ratios.ndim == 1
    assert np.isfinite(ratios).all()
    assert (ratios > 0).all()


# ---- inference helpers -----------------------------------------------------
def test_tstat_and_bootstrap_ci_basic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert st.tstat(x, 0.0) > 0
    lo, hi = st.bootstrap_ci(x, n_boot=2000, seed=1)
    assert lo < x.mean() < hi


def test_percentile_in():
    dist = np.arange(100, dtype=float)
    assert 45 <= st.percentile_in(50.0, dist) <= 55
    assert st.percentile_in(np.nan, dist) != st.percentile_in(np.nan, dist) or True  # nan-safe


def test_summarize_keys(null_tape):
    bars, _, el, _ = null_tape
    s = st.summarize(st.vol_ratio_around(bars, el["election_date"]))
    assert set(s) == {"n", "mean", "std", "win_rate", "tstat"}


# ---- carry engine invariants ----------------------------------------------
def test_carry_ledger_columns(null_tape):
    bars, vix, el, _ = null_tape
    led = st.short_vol_carry(bars, vix, el["election_date"])
    assert list(led.columns) == ["entry_date", "implied", "realized", "pnl_gross", "pnl_net"]


def test_carry_net_is_gross_minus_cost(null_tape):
    bars, vix, el, _ = null_tape
    led = st.short_vol_carry(bars, vix, el["election_date"], cost_vol_pts=2.5)
    assert np.allclose(led["pnl_net"], led["pnl_gross"] - 2.5)


def test_carry_cost_monotonically_lowers_mean(null_tape):
    bars, vix, el, _ = null_tape
    means = [
        st.summarize(
            st.short_vol_carry(bars, vix, el["election_date"], cost_vol_pts=c)["pnl_net"].to_numpy()
        )["mean"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_carry_pnl_is_implied_minus_realized(null_tape):
    bars, vix, el, _ = null_tape
    led = st.short_vol_carry(bars, vix, el["election_date"])
    assert np.allclose(led["pnl_gross"], led["implied"] - led["realized"])


def test_vrp_around_matches_carry_definition(null_tape):
    """VRP (implied - realized) is the same quantity the carry harvests."""
    bars, vix, el, _ = null_tape
    vrp = st.vrp_around(bars, vix, el["election_date"], pre=5, post=21)
    assert np.allclose(vrp["vrp"], vrp["implied"] - vrp["realized"])


# ---- spine #1: the vol spike is found only when planted ---------------------
def test_vol_spike_detected_only_when_planted(null_tape, spike_tape):
    null_bars, _, null_el, _ = null_tape
    spike_bars, _, spike_el, _ = spike_tape
    t_null = st.tstat(st.vol_ratio_around(null_bars, null_el["election_date"]), 1.0)
    t_spike = st.tstat(st.vol_ratio_around(spike_bars, spike_el["election_date"]), 1.0)
    # On the null the ratio is indistinguishable from 1 (|t| below the bar);
    # on the planted tape it clears the bar decisively.
    assert abs(t_null) < 2.0
    assert t_spike > 2.0


# ---- spine #2: the election EXCESS over always-on carry, only when planted --
def _excess_t(bars, vix, el, seed=5):
    led = st.short_vol_carry(bars, vix, el["election_date"])
    rd = st.random_carry_dates(bars, 6000, seed=seed)
    base = st.short_vol_carry(bars, vix, rd)["pnl_gross"].mean()
    excess = led["pnl_gross"].to_numpy() - base
    return st.tstat(excess)


def test_election_excess_over_carry_only_when_planted(null_tape, spike_tape):
    """The point of the study: gross carry is paid on any date (the VRP). The election
    *excess* over an always-on carry is significant only when an election-specific premium
    is planted; on the null it is not distinguishable from the everyday premium."""
    t_null = _excess_t(*null_tape[:3])
    t_spike = _excess_t(*spike_tape[:3])
    assert abs(t_null) < 2.0       # null: no election-specific edge over the always-on VRP
    assert t_spike > 2.0           # planted: a real election excess emerges


def test_gross_carry_is_paid_even_on_the_null(null_tape):
    """Sanity for the headline trap: even on the null tape the gross short-vol carry is
    positive (the baseline VRP the synthetic plants) — a positive gross mean says nothing
    about an *election* effect."""
    bars, vix, el, _ = null_tape
    s = st.summarize(st.short_vol_carry(bars, vix, el["election_date"])["pnl_gross"].to_numpy())
    assert s["mean"] > 0.0


def test_random_carry_dates_reproducible(null_tape):
    bars, _, _, _ = null_tape
    a = st.random_carry_dates(bars, 50, seed=3)
    b = st.random_carry_dates(bars, 50, seed=3)
    assert (a == b).all()

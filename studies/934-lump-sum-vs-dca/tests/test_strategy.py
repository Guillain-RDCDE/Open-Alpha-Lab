"""Experiment invariants, inference sanity, and the study's spine — all offline/synthetic.

The spine: on a tape with a planted risk premium the lump sum must finish richer on
average; on the null the gap must sit on zero and the win rate on one half; on a
falling tape DCA must win. Around that, the mechanical invariants: one execution lag
and no look-ahead, a one-tranche DCA that *is* the lump sum, a proportional cost that
cancels because both arms invest the same dollar, and a DCA arm whose terminal wealth
always lands in a tighter band.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lump_vs_dca import data, strategy as st  # noqa: E402


def _small(signal_strength, seed):
    """A shorter synthetic world, so the multi-seed controls stay quick."""
    return data.synthetic_daily(n_years=15, signal_strength=signal_strength, seed=seed)


# --------------------------------------------------------------------------- #
# The monthly grid & the execution lag
# --------------------------------------------------------------------------- #
def test_month_end_dates_are_month_ends():
    idx = pd.bdate_range("2015-01-01", periods=600)
    me = st.month_end_dates(idx)
    s = pd.Series(idx, index=idx)
    for d in me:
        same_month = s[(s.index.year == d.year) & (s.index.month == d.month)]
        assert d == same_month.index[-1]
    assert len(me) >= 27


def test_execution_dates_are_strictly_after_the_decision():
    idx = pd.bdate_range("2015-01-01", periods=400)
    me = st.month_end_dates(idx)
    ex = st.execution_dates(idx, me)
    for d, e in zip(me, ex):
        if e is None:
            continue
        assert e > d
        # exactly one trading day later — the single documented execution lag
        assert idx.get_loc(e) == idx.get_loc(d) + 1
    assert ex[-1] is None or ex[-1] > me[-1]


def test_last_decision_without_a_successor_is_dropped():
    idx = pd.bdate_range("2015-01-01", periods=400)
    me = st.month_end_dates(idx)
    ex = st.execution_dates(idx, me)
    # the final month-end is the last bar in its month; if it is also the last bar of
    # the sample there is no next day to trade on
    if me[-1] == idx[-1]:
        assert ex[-1] is None


# --------------------------------------------------------------------------- #
# Experiment invariants
# --------------------------------------------------------------------------- #
def test_one_tranche_dca_is_the_lump_sum(rising):
    prices, _ = rising
    exp = st.run_experiment(prices["asset"], prices["cash"], n_tranches=1, cost_bps=5.0)
    assert len(exp) > 50
    assert np.allclose(exp["w_lump"].to_numpy(), exp["w_dca"].to_numpy())
    assert np.allclose(exp["gap_cents"].to_numpy(), 0.0, atol=1e-9)


def test_proportional_cost_very_nearly_cancels(rising):
    """Both arms put the same $1 to work, so a bps-of-NAV cost hits both the same."""
    prices, _ = rising
    a, c = prices["asset"], prices["cash"]
    g0 = st.run_experiment(a, c, cost_bps=0.0)["gap_cents"]
    g25 = st.run_experiment(a, c, cost_bps=25.0)["gap_cents"]
    assert np.allclose(g0.to_numpy(), g25.to_numpy(), atol=0.15)


def test_both_arms_are_valued_on_the_same_date(rising):
    prices, _ = rising
    exp = st.run_experiment(prices["asset"], prices["cash"])
    assert (exp["end"] > exp.index).all()
    # the terminal date is the 12th month-end execution after the start
    assert exp["end"].is_monotonic_increasing


def test_no_lookahead_windows_ignore_the_far_future(rising):
    """Perturbing the tape after a window's end must not change that window's row."""
    prices, _ = rising
    a, c = prices["asset"].copy(), prices["cash"]
    exp_full = st.run_experiment(a, c)
    cut = exp_full["end"].iloc[60]
    a2 = a.copy()
    a2[a2.index > cut] *= 3.0
    exp_pert = st.run_experiment(a2, c)
    head_full = exp_full["gap_cents"].iloc[:60].to_numpy()
    head_pert = exp_pert["gap_cents"].iloc[:60].to_numpy()
    assert np.allclose(head_full, head_pert)


def test_dca_lands_in_a_tighter_band_than_the_lump_sum(rising):
    """The one genuine part of the folklore: DCA reduces dispersion of the outcome."""
    prices, _ = rising
    s = st.race(prices["asset"], prices["cash"], n_boot=200)
    assert 0.0 < s["dispersion_ratio"] < 1.0
    assert s["worst_w_dca"] > s["worst_w_lump"]


def test_cash_leg_is_credited_to_dca():
    """With a flat asset, DCA's edge is exactly the T-bill yield its cash earns."""
    idx = pd.bdate_range("2010-01-04", periods=1000)
    asset = pd.Series(100.0, index=idx)
    cash = pd.Series(100.0 * (1.0 + 0.05 / 252) ** np.arange(1000), index=idx)
    exp = st.run_experiment(asset, cash, cost_bps=0.0)
    # asset never moves, so the lump sum returns exactly 0 and DCA banks the bill yield
    assert np.allclose(exp["w_lump"].to_numpy(), 1.0)
    assert (exp["gap_cents"] < 0).all()


# --------------------------------------------------------------------------- #
# Exposure vs timing — the beta control on the gap itself
# --------------------------------------------------------------------------- #
def test_average_dca_exposure_is_analytic():
    """(n+1)/(2n): 13/24 for twelve tranches, 1.0 for a single tranche."""
    assert st.average_dca_exposure(12) == 13 / 24
    assert st.average_dca_exposure(1) == 1.0
    assert st.average_dca_exposure(24) < st.average_dca_exposure(12)


def test_exposure_matching_annihilates_the_gap_in_every_world(rising, flat, falling):
    """The gap is EXPOSURE, not timing: match the average weight and it collapses.

    Whatever drift is planted — fat premium, none, or negative — a static portfolio
    holding DCA's own average exposure lands within a cent of DCA, while the raw gap
    ranges over many cents. If this ever failed, the headline would be a timing claim;
    it is not.
    """
    for fixture in (rising, flat, falling):
        prices, _ = fixture
        exp = st.run_experiment(prices["asset"], prices["cash"])
        em = st.exposure_matched_race(exp, n_boot=300)
        assert abs(em["mean_gap_cents"]) < 1.0
        assert abs(em["mean_gap_cents"]) < abs(exp["gap_cents"].mean()) + 1.0
        # matching the weight also brings the dispersions together
        assert 0.75 < em["sd_matched"] / em["sd_dca"] < 1.25


def test_exposure_matched_race_is_excess_of_cash_on_both_legs(rising):
    """Both arms are read net of the SAME cash leg — never one raw against one excess."""
    prices, _ = rising
    exp = st.run_experiment(prices["asset"], prices["cash"])
    em = st.exposure_matched_race(exp, n_boot=200)
    cash = exp["cash_ret"].to_numpy()
    assert np.isclose(em["mean_excess_lump"], (exp["w_lump"] - 1 - cash).mean())
    assert np.isclose(em["mean_excess_dca"], (exp["w_dca"] - 1 - cash).mean())
    # on a planted-premium tape both arms are paid for their risk at a similar rate
    assert abs(em["reward_disp_lump"] - em["reward_disp_dca"]) < 0.25


def test_dispersion_matched_weight_equalises_dispersion(rising):
    prices, _ = rising
    exp = st.run_experiment(prices["asset"], prices["cash"])
    w = st.dispersion_matched_weight(exp)
    assert 0.3 < w < 0.9
    em = st.exposure_matched_race(exp, weight=w, n_boot=200)
    assert abs(em["sd_matched"] / em["sd_dca"] - 1.0) < 0.05


# --------------------------------------------------------------------------- #
# Inference primitives sanity
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.5 + rng.normal(0, 1.0, 4000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 1.0, 4000))) < 3


def test_newey_west_is_more_conservative_under_overlap():
    """On an overlapping (autocorrelated) series the HAC t must be below the naive t."""
    rng = np.random.default_rng(3)
    raw = 0.3 + rng.normal(0, 1.0, 3000)
    overlapping = pd.Series(raw).rolling(12).mean().dropna().to_numpy()
    assert st.newey_west_t(overlapping, lags=12) < st.one_sample_t(overlapping)


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(76, 100)
    assert lo < 0.76 < hi
    assert lo > 0.5


def test_block_bootstrap_brackets_the_mean():
    rng = np.random.default_rng(7)
    x = 5.0 + rng.normal(0, 10.0, 300)
    b = st.block_bootstrap_mean_ci(x, n_boot=500, seed=934)
    assert b["ci_low"] <= b["mean"] <= b["ci_high"]


def test_welch_and_one_sample_finite(rising):
    prices, _ = rising
    exp = st.run_experiment(prices["asset"], prices["cash"])
    assert np.isfinite(st.one_sample_t(exp["gap_cents"].to_numpy()))
    assert np.isfinite(st.welch_t(exp["w_lump"].to_numpy(), exp["w_dca"].to_numpy()))


# --------------------------------------------------------------------------- #
# Cuts & sweeps
# --------------------------------------------------------------------------- #
def test_conditional_cuts_cover_the_sample(rising):
    prices, _ = rising
    exp = st.run_experiment(prices["asset"], prices["cash"])
    cuts = st.conditional_cuts(exp)
    assert {"after_drawdown", "near_highs"}.issubset(cuts.keys())
    assert cuts["after_drawdown"]["n"] + cuts["near_highs"]["n"] == len(exp)


def test_era_cut_returns_both_halves(rising):
    prices, _ = rising
    exp = st.run_experiment(prices["asset"], prices["cash"])
    split = str(exp.index[len(exp) // 2].date())
    eras = st.era_cut(exp, split=split)
    assert eras["early"]["n"] > 5 and eras["late"]["n"] > 5


def test_ticket_charge_only_widens_the_gap(rising):
    prices, _ = rising
    rows = st.ticket_sweep(prices["asset"], prices["cash"], ticket_grid=(0.0, 5.0, 10.0))
    gaps = [r["mean_gap_cents"] for r in rows]
    assert gaps[0] < gaps[1] < gaps[2]


def test_more_tranches_means_more_cash_drag(rising):
    """In a rising world, the longer you take to average in, the more it costs."""
    prices, _ = rising
    rows = st.tranche_sweep(prices["asset"], prices["cash"], grid=(3, 12, 24))
    gaps = [r["mean_gap_cents"] for r in rows]
    ratios = [r["dispersion_ratio"] for r in rows]
    assert gaps[0] < gaps[1] < gaps[2]
    # Each variant also smooths the outcome, though the ratios are not comparable
    # across grid points: a 24-tranche race is a 24-month horizon, not a 12-month one.
    assert all(0.0 < r < 1.0 for r in ratios)


def test_zero_cash_variant_runs(rising):
    prices, _ = rising
    z = st.zero_cash_variant(prices["asset"], prices["cash"])
    assert np.isfinite(z["mean_gap_cents"])


# --------------------------------------------------------------------------- #
# The study's spine — the two-sided synthetic control
# --------------------------------------------------------------------------- #
def test_planted_premium_crowns_the_lump_sum():
    c = st.synthetic_control(1.0, seeds=range(934, 940), synth=_small)
    assert c["mean_gap_cents"] > 1.0
    assert c["mean_win_rate"] > 0.55
    assert c["frac_seeds_lump_wins"] >= 0.8


def test_falling_tape_crowns_dca():
    c = st.synthetic_control(-1.0, seeds=range(934, 940), synth=_small)
    assert c["mean_gap_cents"] < -1.0
    assert c["mean_win_rate"] < 0.45
    assert c["frac_seeds_lump_wins"] <= 0.2


def test_null_is_quiet():
    c = st.synthetic_control(0.0, seeds=range(934, 940), synth=_small)
    assert abs(c["mean_gap_cents"]) < 2.0
    assert abs(c["mean_win_rate"] - 0.5) < 0.15


def test_bond_leg_shows_a_thinner_gap_than_equity(panel):
    """A thinner planted premium means a thinner cash-drag penalty — same sign, less of it."""
    prices, _ = panel
    eq = st.race(prices["equity"], prices["cash"], n_boot=200)
    bd = st.race(prices["bond"], prices["cash"], n_boot=200)
    assert eq["mean_gap_cents"] > bd["mean_gap_cents"]
    assert bd["dispersion_ratio"] < 1.0

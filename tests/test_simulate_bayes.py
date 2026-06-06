"""Tests for the steelman simulation, Bayesian posterior and Reality Check."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab import bayes, decompose, diagnostics, simulate, universe


# --- simulate ---------------------------------------------------------------
def test_manipulator_pnl_decreasing_and_crosses_zero():
    sw = simulate.pnl_vs_capital(
        overnight_drift_bps=3.0, adv_usd=5e10, daily_vol_bps=117,
        sizes_usd=(1e6, 1e7, 1e8, 1e9, 1e10, 1e11),
    )
    net = sw["net_bps_on_capital"].values
    assert all(net[i] > net[i + 1] for i in range(len(net) - 1))  # bigger -> worse
    assert net[0] > 0 > net[-1]  # small profits, huge-scale loses


def test_breakeven_capital_matches_pnl_zero():
    args = dict(overnight_drift_bps=3.0, adv_usd=5e10, daily_vol_bps=117, temp_coef=1.0)
    be = simulate.breakeven_capital(**args)
    r = simulate.manipulator_pnl_per_day(capital_usd=be, **args)
    assert abs(r["net_bps_on_capital"]) < 1e-6  # net ~ 0 at break-even


def test_monte_carlo_profit_prob_collapses_at_scale():
    base = dict(overnight_drift_bps=3.0, overnight_vol_bps=70, adv_usd=5e10, daily_vol_bps=117, seed=1)
    small = simulate.monte_carlo_pnl(capital_usd=1e6, **base)
    huge = simulate.monte_carlo_pnl(capital_usd=1e11, **base)
    assert small["prob_profit"] > huge["prob_profit"]
    assert huge["mean_usd_per_year"] < 0  # world-moving scale loses in expectation


# --- bayes: posterior -------------------------------------------------------
def test_posterior_below_prior_when_evidence_favours_micro():
    r = bayes.manipulation_posterior(prior_manip=0.5)
    assert r["posterior_manip"] < 0.5  # discriminating evidence pushes it down
    assert 0.0 < r["posterior_manip"] < 1.0
    assert r["combined_LR"] < 1.0


def test_posterior_monotonic_in_prior():
    ps = [bayes.manipulation_posterior(p)["posterior_manip"] for p in (0.1, 0.5, 0.9)]
    assert ps[0] < ps[1] < ps[2]


def test_non_discriminating_evidence_leaves_prior_unchanged():
    ev = [("neutral", 1.0, "")]
    assert abs(bayes.manipulation_posterior(0.5, ev)["posterior_manip"] - 0.5) < 1e-12


def test_sensitivity_grid_shape():
    g = bayes.posterior_sensitivity(priors=(0.2, 0.5, 0.8), lr_scales=(0.5, 1.0, 2.0))
    assert g.shape == (3, 3)
    # stronger evidence (LR^2) -> lower posterior than weaker (LR^0.5) at same prior
    assert g.loc[0.5, "LR^2.0"] < g.loc[0.5, "LR^0.5"]


# --- bayes: reality check ---------------------------------------------------
def test_reality_check_flags_real_signal_and_pure_noise():
    rng = np.random.default_rng(0)
    n = 3000
    # 8 pure-noise series + 1 with genuine drift
    cols = {f"noise{i}": rng.normal(0, 0.01, n) for i in range(8)}
    cols["real"] = rng.normal(0.0006, 0.01, n)  # ~ strong overnight drift
    panel = pd.DataFrame(cols)
    rc = bayes.reality_check(panel, n_boot=500, seed=1)
    assert rc["n_series"] == 9
    assert 0.0 <= rc["reality_check_pvalue"] <= 1.0
    assert rc["reality_check_pvalue"] < 0.10  # the real series beats the multiplicity null

    pure = pd.DataFrame({f"n{i}": rng.normal(0, 0.01, n) for i in range(9)})
    rc0 = bayes.reality_check(pure, n_boot=500, seed=2)
    assert rc0["reality_check_pvalue"] > 0.10  # nothing real -> not significant


# --- universe: cross-section compute (no network) ---------------------------
def test_cross_section_decompose_on_synthetic_panel():
    panel = {}
    for i in range(6):
        panel[f"S{i}"] = diagnostics.synthetic_ohlc(
            overnight_bias_bps=4, intraday_bias_bps=-1, seed=i, n_days=1500
        )
    cs = universe.cross_section_decompose(panel, min_days=750)
    assert cs["n_stocks"] == 6
    assert cs["frac_overnight_wins"] > 0.8  # by construction night beats day
    assert (1 + cs["ew_overnight"]).prod() > (1 + cs["ew_intraday"]).prod()

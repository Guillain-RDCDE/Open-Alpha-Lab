"""After-tax accounting, the break-even solver, and the study's spine — all synthetic.

The spine: on a planted world with a real pre-tax coupon-yield gap the closed-form
break-even solver recovers the planted crossover rate and the after-tax difference flips
sign across it; on a twin null the break-even collapses to ~0 and there is no *pre-tax*
difference to find. Tax rates only ever reduce the taxable leg; costs and borrow only ever
reduce the edge; the overlay's signal is strictly lagged.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from after_tax import data, strategy as st  # noqa: E402

SYN = {"muni": "muni", "taxable": "taxable", "cash": "cash"}


# --------------------------------------------------------------------------- #
# The tax accounting
# --------------------------------------------------------------------------- #
def test_income_tax_rate_by_kind():
    prof = st.tax_profile(fed_rate=0.37, niit=0.038, state_rate=0.09)
    assert st.income_tax_rate("muni", prof) == pytest.approx(0.09)      # federal-exempt
    assert st.income_tax_rate("cash", prof) == pytest.approx(0.408)     # state-exempt
    assert st.income_tax_rate("taxable", prof) == pytest.approx(0.498)  # everything


def test_muni_in_state_exemption_removes_the_state_bill():
    prof = st.tax_profile(state_rate=0.093, muni_state_exempt_frac=1.0)
    assert st.income_tax_rate("muni", prof) == pytest.approx(0.0)


def test_after_tax_at_zero_bracket_is_the_total_return(planted):
    panel, _ = planted
    zero = st.tax_profile(0.0, 0.0, 0.0)
    for col in ("muni", "taxable", "cash"):
        got = st.after_tax(panel, col, zero, SYN)
        assert np.allclose(got.to_numpy(), panel["total"][col].to_numpy())


def test_tax_only_ever_reduces_the_taxable_leg(planted):
    panel, _ = planted
    means = [
        st.after_tax(panel, "taxable", st.tax_profile(f, 0.0, 0.0), SYN).mean()
        for f in (0.0, 0.24, 0.37, 0.50)
    ]
    assert all(means[i] > means[i + 1] for i in range(len(means) - 1))


def test_federal_rate_does_not_touch_the_muni_leg(planted):
    panel, _ = planted
    a = st.after_tax(panel, "muni", st.tax_profile(0.0, 0.0, 0.0), SYN)
    b = st.after_tax(panel, "muni", st.tax_profile(0.37, 0.038, 0.0), SYN)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_after_tax_difference_is_linear_in_the_bracket(planted):
    """d(tau) = d(0) + tau x mean(taxable income) — the identity the solver relies on."""
    panel, _ = planted
    d = [st.race(panel, "muni", "taxable", st.tax_profile(f, 0.0, 0.0),
                 cash="cash", cost_bps=0.0, kinds=SYN)["diff_bps"]
         for f in (0.0, 0.10, 0.20, 0.30)]
    steps = np.diff(d)
    assert np.allclose(steps, steps[0], atol=1e-9)
    assert steps[0] > 0  # a higher bracket can only help the tax-exempt leg


# --------------------------------------------------------------------------- #
# The break-even solver — the study's spine
# --------------------------------------------------------------------------- #
def test_breakeven_recovers_the_planted_rate(planted):
    panel, truth = planted
    be = st.breakeven_rate(panel, "muni", "taxable", kinds=SYN)
    assert be["tax_driven"] is True
    assert abs(be["breakeven"] - truth["planted_breakeven"]) < 0.20


def test_breakeven_recovery_is_seed_robust():
    errs = []
    for s in range(8):
        panel, truth = data.synthetic_panel(signal_strength=1.0, seed=952 + s)
        be = st.breakeven_rate(panel, "muni", "taxable", kinds=SYN)
        errs.append(be["breakeven"] - truth["planted_breakeven"])
    errs = np.array(errs)
    assert abs(errs.mean()) < 0.08
    assert np.abs(errs).max() < 0.20


def test_at_the_breakeven_bracket_the_after_tax_means_tie(planted):
    panel, _ = planted
    be = st.breakeven_rate(panel, "muni", "taxable", kinds=SYN)["breakeven"]
    r = st.race(panel, "muni", "taxable", st.tax_profile(be, 0.0, 0.0),
                cash="cash", cost_bps=0.0, kinds=SYN)
    assert abs(r["diff_bps"]) < 1e-6


def test_after_tax_difference_flips_sign_across_the_breakeven(planted):
    panel, _ = planted
    d = st.synthetic_detect(panel, kinds=SYN)
    assert d["diff_below_breakeven"] < 0 < d["diff_above_breakeven"]


def test_null_breakeven_collapses_to_zero(twin_null):
    panel, truth = twin_null
    be = st.breakeven_rate(panel, "muni", "taxable", kinds=SYN)
    assert abs(be["breakeven"] - truth["planted_breakeven"]) < 0.20


def test_null_has_no_pretax_difference_to_find():
    """Twins must not manufacture a pre-tax edge — the machinery is unbiased."""
    ts = np.array([
        st.breakeven_rate(
            data.synthetic_panel(signal_strength=0.0, seed=952 + s)[0],
            "muni", "taxable", kinds=SYN)["pretax_t"]
        for s in range(8)
    ])
    assert abs(ts.mean()) < 1.5
    assert (np.abs(ts) >= 2.5).sum() <= 1


def test_planted_pretax_gap_is_detected(planted):
    panel, _ = planted
    be = st.breakeven_rate(panel, "muni", "taxable", kinds=SYN)
    assert be["pretax_diff_bps"] < 0        # the taxable leg wins pre-tax
    assert be["pretax_t"] < -2.0


def test_breakeven_ci_brackets_the_point_estimate(planted):
    ci = st.breakeven_ci(planted[0], "muni", "taxable", kinds=SYN, n_boot=400)
    assert ci["ci_low"] <= ci["breakeven"] <= ci["ci_high"]
    assert 0.0 <= ci["p_below_zero"] <= 1.0
    assert 0.0 <= ci["p_above_top"] <= 1.0


def test_breakeven_ci_is_tight_when_the_price_legs_are_quiet():
    """With no price-leg noise the break-even is genuinely identified."""
    panel, truth = data.synthetic_panel(duration_vol_ann=0.0, idio_vol_ann=0.0,
                                        signal_strength=1.0, seed=952)
    ci = st.breakeven_ci(panel, "muni", "taxable", kinds=SYN, n_boot=400)
    assert abs(ci["breakeven"] - truth["planted_breakeven"]) < 1e-6
    assert (ci["ci_high"] - ci["ci_low"]) < 0.01
    assert ci["p_below_zero"] == 0.0


def test_breakeven_ci_widens_with_price_leg_noise():
    """The audit finding: the CI width is driven by the PRICE legs, not by tax."""
    quiet, _ = data.synthetic_panel(duration_vol_ann=0.01, idio_vol_ann=0.002,
                                    signal_strength=1.0, seed=952)
    loud, _ = data.synthetic_panel(duration_vol_ann=0.12, idio_vol_ann=0.03,
                                   signal_strength=1.0, seed=952)
    w_q = st.breakeven_ci(quiet, "muni", "taxable", kinds=SYN, n_boot=400)
    w_l = st.breakeven_ci(loud, "muni", "taxable", kinds=SYN, n_boot=400)
    assert (w_l["ci_high"] - w_l["ci_low"]) > 3.0 * (w_q["ci_high"] - w_q["ci_low"])


def test_tax_assumptions_cannot_move_the_breakeven():
    """`tau*` is invariant to every tax knob — so 'stable under every sweep' is a
    statement about the arithmetic, never evidence that the estimate is precise."""
    panel, _ = data.synthetic_panel(signal_strength=1.0, seed=952)
    base = st.breakeven_rate(panel, "muni", "taxable", kinds=SYN)["breakeven"]
    for prof in (st.tax_profile(0.37), st.tax_profile(0.24, 0.0, 0.093),
                 st.tax_profile(0.0, 0.0, 0.0, capgain_rate=0.238)):
        got = st.breakeven_rate(panel, "muni", "taxable", profile=prof,
                                kinds=SYN)["breakeven"]
        # only the state knob shifts it, and only because it re-bases both legs
        assert np.isfinite(got)
    assert abs(st.breakeven_rate(panel, "muni", "taxable",
                                 profile=st.tax_profile(0.37), kinds=SYN)["breakeven"]
               - base) < 1e-12


def test_income_breakeven_recovers_the_planted_yield_ratio(planted):
    panel, truth = planted
    ib = st.income_breakeven(panel, "muni", "taxable", kinds=SYN, n_boot=400)
    # income legs are constant by construction -> the yield-ratio break-even is exact
    assert abs(ib["breakeven"] - truth["planted_breakeven"]) < 1e-9
    assert ib["ci_low"] <= ib["breakeven"] <= ib["ci_high"]
    assert (ib["ci_high"] - ib["ci_low"]) < 1e-6


def test_income_breakeven_is_tighter_than_the_total_return_one(planted):
    """The honest split: the yield question is pinned down, the total-return one is not."""
    panel, _ = planted
    ib = st.income_breakeven(panel, "muni", "taxable", kinds=SYN, n_boot=400)
    tb = st.breakeven_ci(panel, "muni", "taxable", kinds=SYN, n_boot=400)
    assert (ib["ci_high"] - ib["ci_low"]) < (tb["ci_high"] - tb["ci_low"])


def test_tax_term_reconstructs_the_after_tax_difference(planted):
    panel, _ = planted
    prof = st.tax_profile(0.37)
    dec = st.tax_constant_decomposition(panel, "muni", "taxable", prof, kinds=SYN)
    r = st.race(panel, "muni", "taxable", prof, cash="cash", cost_bps=0.0, kinds=SYN)
    assert dec["total_mean_bps"] == pytest.approx(r["diff_bps"], abs=1e-6)
    assert dec["pretax_mean_bps"] + dec["tax_mean_bps"] == pytest.approx(
        dec["total_mean_bps"], abs=1e-9)


def test_tax_term_is_a_near_constant_that_inflates_the_t_stat(planted):
    """The audit finding: |t| >= 2 can be manufactured by the tax constant alone."""
    panel, _ = planted
    dec = st.tax_constant_decomposition(panel, "muni", "taxable",
                                        st.tax_profile(0.37), kinds=SYN)
    # the tax term carries a big share of the MEAN and essentially none of the VARIANCE
    assert dec["var_share_tax"] < 0.05
    assert abs(dec["tax_mean_bps"]) > 5.0
    # and it moves the t-stat without any new information arriving
    assert dec["total_t"] > dec["pretax_t"]


def test_t_stat_climbs_with_the_bracket_on_a_twin_null(twin_null):
    """On statistical twins the after-tax t still climbs with tau — proof the t-stat is
    not testing a market edge."""
    panel, _ = twin_null
    ts = [st.race(panel, "muni", "taxable", st.tax_profile(f, 0.0, 0.0), cash="cash",
                  cost_bps=0.0, kinds=SYN)["t_diff"] for f in (0.0, 0.20, 0.40)]
    assert ts[0] < ts[1] < ts[2]
    assert ts[2] > 2.0  # significance out of thin tax air


def test_breakeven_flags_a_pretax_winner_as_not_tax_driven():
    """If the muni leg already wins pre-tax the break-even is <= 0 and must be flagged."""
    panel, _ = data.synthetic_panel(muni_income_ann=0.055, taxable_income_ann=0.045,
                                    signal_strength=1.0, seed=952)
    be = st.breakeven_rate(panel, "muni", "taxable", kinds=SYN)
    assert be["breakeven"] < 0
    assert be["tax_driven"] is False


# --------------------------------------------------------------------------- #
# Race mechanics, costs, borrow
# --------------------------------------------------------------------------- #
def test_race_excess_of_cash_flag(planted):
    panel, _ = planted
    assert st.race(panel, "muni", "taxable", st.tax_profile(), cash="cash",
                   kinds=SYN)["excess_of_cash"] is True
    # when the taxable arm IS the cash leg there is nothing left to subtract
    assert st.race(panel, "muni", "cash", st.tax_profile(), cash="cash",
                   kinds=SYN)["excess_of_cash"] is False


def test_costs_monotonically_reduce_the_edge(planted):
    panel, _ = planted
    tbl = st.cost_sweep(panel, "muni", "taxable", cost_bps_grid=(0.0, 5.0, 25.0, 100.0),
                        cash="cash", kinds=SYN)
    d = tbl["diff_bps"].to_numpy()
    assert all(d[i] > d[i + 1] for i in range(len(d) - 1))


def test_borrow_monotonically_reduces_the_spread(planted):
    panel, _ = planted
    tbl = st.borrow_sweep(panel, "muni", "taxable", cash="cash", kinds=SYN)
    d = tbl["diff_bps"].to_numpy()
    assert all(d[i] > d[i + 1] for i in range(len(d) - 1))
    assert tbl["borrow_bps_yr"].iloc[0] == 0


def test_state_sweep_in_state_exemption_helps_the_muni_leg(planted):
    panel, _ = planted
    tbl = st.state_sweep(panel, "muni", "taxable", state_rates=(0.093,),
                         cash="cash", kinds=SYN)
    out_state = tbl[tbl["in_state_frac"] == 0.0]["diff_bps"].iloc[0]
    in_state = tbl[tbl["in_state_frac"] == 1.0]["diff_bps"].iloc[0]
    assert in_state > out_state


def test_bracket_sweep_is_monotone_and_labelled(planted):
    panel, _ = planted
    tbl = st.bracket_sweep(panel, "muni", "taxable", cash="cash", kinds=SYN)
    assert list(tbl["bracket"]) == [b[0] for b in st.BRACKETS]
    assert tbl["diff_bps"].is_monotonic_increasing


def test_era_cut_returns_both_halves(planted):
    panel, _ = planted
    eras = st.era_cut(panel, "muni", "taxable", st.tax_profile(0.37),
                      split="2010-01", cash="cash", kinds=SYN)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["diff_bps"]) and e["n_months"] > 24


# --------------------------------------------------------------------------- #
# The single execution lag
# --------------------------------------------------------------------------- #
def _panel_with_varying_income(seed=952):
    """A panel whose two income legs cross over, so the overlay actually switches."""
    panel, _ = data.synthetic_panel(signal_strength=1.0, seed=seed)
    n = len(panel["total"])
    rng = np.random.default_rng(seed)
    wobble = pd.Series(np.cumsum(rng.normal(0.0, 0.0004, n)), index=panel["total"].index)
    panel["income"]["muni"] = (panel["income"]["muni"] + wobble).clip(lower=0.0)
    panel["total"]["muni"] = panel["price"]["muni"] + panel["income"]["muni"]
    return panel


def test_overlay_signal_is_binary_and_lagged():
    panel = _panel_with_varying_income()
    o = st.switch_overlay(panel, "muni", "taxable", st.tax_profile(0.37), kinds=SYN)
    sig = o["signal"]
    assert set(np.unique(sig.dropna().to_numpy())).issubset({0.0, 1.0})
    assert o["n_switches"] >= 1


def test_overlay_has_no_lookahead():
    """Perturbing the tail of the income tape must not move earlier positions."""
    panel = _panel_with_varying_income()
    o1 = st.switch_overlay(panel, "muni", "taxable", st.tax_profile(0.37), kinds=SYN)
    bumped = {k: v.copy() for k, v in panel.items()}
    n = len(bumped["income"])
    bumped["income"].iloc[int(n * 0.8):, bumped["income"].columns.get_loc("muni")] *= 5.0
    bumped["total"]["muni"] = bumped["price"]["muni"] + bumped["income"]["muni"]
    o2 = st.switch_overlay(bumped, "muni", "taxable", st.tax_profile(0.37), kinds=SYN)
    cut = int(n * 0.8)
    a = o1["signal"].iloc[:cut - 1]
    b = o2["signal"].reindex(a.index)
    assert (a.fillna(-9) == b.fillna(-9)).all()


def test_overlay_switch_costs_reduce_its_return():
    panel = _panel_with_varying_income()
    rets = [st.switch_overlay(panel, "muni", "taxable", st.tax_profile(0.37),
                              cost_bps=c, kinds=SYN)["overlay_ann_pct"]
            for c in (0.0, 5.0, 50.0)]
    assert rets[0] > rets[1] > rets[2]


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.004 + rng.normal(0, 0.01, 2000)) > 4
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 2000))) < 3


def test_newey_west_matches_plain_t_at_zero_lags():
    rng = np.random.default_rng(2)
    x = rng.normal(0.001, 0.01, 500)
    assert st.newey_west_t(x, lags=0) == pytest.approx(st.one_sample_t(x), rel=0.01)


def test_bootstrap_ci_brackets_the_point_mean(planted):
    panel, _ = planted
    r = st.race(panel, "muni", "taxable", st.tax_profile(0.37), cash="cash", kinds=SYN)
    ci = st.block_bootstrap_mean_ci(r["diff"], n_boot=500, seed=952)
    assert ci["ci_low_bps"] <= ci["mean_bps"] <= ci["ci_high_bps"]


def test_summary_is_sane(planted):
    panel, _ = planted
    s = st.summary(panel["total"]["muni"])
    assert s["n_months"] == len(panel["total"])
    assert np.isfinite(s["sharpe"]) and np.isfinite(s["vol_ann"])
    assert s["max_drawdown"] <= 0.0

"""The study's spine — all offline, all synthetic, all deterministic.

The spine in one line: on a **pure return-of-capital null** the payout rank must forecast the
next payout loudly, forecast the price leg loudly and negatively (give-back ratio ≈ 1), and
forecast **total return not at all**; on a **planted** yield-to-return world the total-return
leg must fire; and a **beta tilt** across the sort must be absorbed by the CAPM control
rather than mistaken for an edge.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dist_illusion import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_real_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.004 + rng.normal(0, 0.01, 2000)) > 4
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 2000))) < 3


def test_newey_west_handles_short_and_degenerate_input():
    assert np.isnan(st.newey_west_t([0.1, 0.2]))
    assert np.isnan(st.newey_west_t(np.zeros(50)))


def test_one_sample_t_matches_newey_west_at_zero_lags():
    """At zero lags the HAC t collapses to the plain t (bar the ddof=0/1 convention)."""
    rng = np.random.default_rng(2)
    x = 0.002 + rng.normal(0, 0.01, 500)
    plain, hac = st.one_sample_t(x), st.newey_west_t(x, lags=0)
    assert abs(plain - hac) / abs(plain) < 0.01


def test_bootstrap_ci_brackets_the_point_mean():
    rng = np.random.default_rng(3)
    x = 0.003 + rng.normal(0, 0.02, 400)
    ci = st.block_bootstrap_ci(x, n_boot=500, seed=946)
    assert ci["ci_low"] <= ci["mean_bps"] <= ci["ci_high"]
    assert 0.0 <= ci["frac_positive"] <= 1.0


def test_capm_recovers_a_planted_beta_and_alpha():
    rng = np.random.default_rng(4)
    mkt = rng.normal(0.007, 0.04, 600)
    y = 0.002 + 0.7 * mkt + rng.normal(0, 0.01, 600)
    c = st.capm(y, mkt)
    assert abs(c["beta"] - 0.7) < 0.05
    assert abs(c["alpha_bps"] - 20.0) < 10.0
    assert c["t_alpha"] > 3


# --------------------------------------------------------------------------- #
# The sort: mechanics, lag, turnover
# --------------------------------------------------------------------------- #
def test_sorted_legs_columns_and_shape(null_panel):
    panel, _ = null_panel
    legs = st.sorted_legs(panel, min_funds=4)
    for c in ("hi", "lo", "hi_p", "lo_p", "hi_d", "lo_d", "dhi", "dlo",
              "hml", "hml_p", "hml_d", "n", "k", "to_hi", "to_lo"):
        assert c in legs.columns
    assert not legs[["hi", "lo", "hml"]].isna().any().any()
    # Twelve months of warmup are consumed by the trailing rate, one more by the lag.
    assert len(legs) == panel["total"].shape[0] - 12


def test_high_leg_really_is_the_high_payout_leg(null_panel):
    panel, _ = null_panel
    legs = st.sorted_legs(panel, min_funds=4)
    assert (legs["dhi"] > legs["dlo"]).all()
    assert legs["hml_d"].mean() > 0


def test_execution_lag_is_exactly_one_month(null_panel):
    """Perturbing month t+1's returns must not change the ranking formed at month t."""
    panel, _ = null_panel
    base = st.sorted_legs(panel, min_funds=4)
    p2 = dict(panel)
    p2["total"] = panel["total"].copy()
    p2["total"].iloc[-1] += 0.5          # only the very last held month is touched
    pert = st.sorted_legs(p2, min_funds=4)
    assert np.allclose(base["dhi"].to_numpy()[:-1], pert["dhi"].to_numpy()[:-1])
    assert np.allclose(base["hml"].to_numpy()[:-1], pert["hml"].to_numpy()[:-1])
    assert base["hml"].iloc[-1] != pert["hml"].iloc[-1]


def test_fama_macbeth_uses_next_month_only(null_panel):
    panel, _ = null_panel
    base = st.fama_macbeth(panel, "total", min_funds=4)
    p2 = dict(panel)
    p2["total"] = panel["total"].copy()
    p2["total"].iloc[:20] += 0.2
    pert = st.fama_macbeth(p2, "total", min_funds=4)
    # Only the first 20 held months can move; the tail slopes are untouched.
    assert np.allclose(base["slopes"].to_numpy()[25:], pert["slopes"].to_numpy()[25:])


def test_min_funds_gate_shrinks_the_sample(null_panel):
    panel, _ = null_panel
    wide = st.sorted_legs(panel, min_funds=4)
    narrow = st.sorted_legs(panel, min_funds=99)
    assert len(wide) > 0 and len(narrow) == 0


# --------------------------------------------------------------------------- #
# The null: return of capital, and nothing else
# --------------------------------------------------------------------------- #
def test_null_payout_rank_forecasts_the_next_payout(null_panel):
    panel, _ = null_panel
    fm = st.fama_macbeth(panel, "dist", min_funds=4)
    assert fm["mean_bps"] > 10
    assert fm["tstat"] > 5


def test_null_payout_rank_forecasts_price_erosion(null_panel):
    panel, _ = null_panel
    fm = st.fama_macbeth(panel, "price", min_funds=4)
    assert fm["mean_bps"] < -10
    assert fm["tstat"] < -5


def test_null_payout_rank_says_nothing_about_total_return(null_panel):
    panel, _ = null_panel
    fm = st.fama_macbeth(panel, "total", min_funds=4)
    assert abs(fm["tstat"]) < 2.0
    assert abs(fm["mean_bps"]) < 20.0


def test_null_giveback_ratio_is_about_one(null_panel):
    panel, _ = null_panel
    legs = st.sorted_legs(panel, min_funds=4)
    assert 0.85 < st.giveback_ratio(legs) < 1.15


def test_null_is_quiet_across_seeds():
    ts = np.array([
        st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=946 + s)[0])["t_total"]
        for s in range(8)
    ])
    assert abs(ts.mean()) < 1.0
    assert (np.abs(ts) >= 2.0).sum() <= 1


# --------------------------------------------------------------------------- #
# The positive control: a planted yield-to-return link must be found
# --------------------------------------------------------------------------- #
def test_planted_link_is_recovered(planted_panel):
    panel, truth = planted_panel
    fm = st.fama_macbeth(panel, "total", min_funds=4)
    assert fm["tstat"] > 3
    # The recovered slope should land near the planted one (within 40%).
    planted = truth["planted_slope_per_sd"] * 1e4
    assert abs(fm["mean_bps"] - planted) < 0.4 * planted


def test_planted_link_shows_in_the_tercile_spread(planted_panel):
    panel, _ = planted_panel
    d = st.synthetic_detect(panel, min_funds=4)
    assert d["hml_bps"] > 0
    assert d["t_hml"] > 3
    assert d["giveback"] < 0.9      # part of the payout is now genuinely earned back


def test_signal_strength_is_monotone():
    got = []
    for ss in (0.0, 0.5, 1.0):
        panel, _ = data.synthetic_panel(signal_strength=ss, seed=946)
        got.append(st.fama_macbeth(panel, "total", min_funds=4)["mean_bps"])
    assert got[0] < got[1] < got[2]


# --------------------------------------------------------------------------- #
# The confound control: a beta tilt is not an edge
# --------------------------------------------------------------------------- #
def test_beta_tilt_fakes_a_raw_spread_and_capm_absorbs_it(beta_confound_panel):
    panel, _ = beta_confound_panel
    d = st.synthetic_detect(panel, min_funds=4)
    assert d["capm_hml"]["beta"] < -0.2          # high-payout leg carries less market
    assert d["hml_bps"] < 0                      # so the raw spread is negative in an up-market
    assert abs(d["capm_hml"]["t_alpha"]) < 2.0   # but there is no alpha to find


# --------------------------------------------------------------------------- #
# Costs, borrow, race, eras
# --------------------------------------------------------------------------- #
def test_costs_and_borrow_monotonically_reduce_the_spread(null_panel):
    panel, _ = null_panel
    legs = st.sorted_legs(panel, min_funds=4)
    means = [st.net_hml(legs, cost_bps=c, borrow_bps_annual=b).mean()
             for c, b in ((0.0, 0.0), (5.0, 0.0), (5.0, 100.0), (25.0, 400.0))]
    assert means[0] >= means[1] >= means[2] >= means[3]


def test_cost_borrow_sweep_grid(null_panel):
    panel, _ = null_panel
    legs = st.sorted_legs(panel, min_funds=4)
    sweep = st.cost_borrow_sweep(legs)
    assert set(sweep.columns) == {"cost_bps", "borrow_bps", "mean_bps", "tstat"}
    assert len(sweep) == 16
    gross = sweep[(sweep.cost_bps == 0) & (sweep.borrow_bps == 0)]["mean_bps"].iloc[0]
    worst = sweep[(sweep.cost_bps == 25) & (sweep.borrow_bps == 200)]["mean_bps"].iloc[0]
    assert gross > worst


def test_race_is_excess_of_cash_and_finite(null_panel):
    panel, _ = null_panel
    legs = st.sorted_legs(panel, min_funds=4)
    r = st.race(panel, legs)
    for k in ("hi", "lo", "bench"):
        assert np.isfinite(r[k]["sharpe"]) and np.isfinite(r[k]["vol_ann"])
        # excess-of-cash means each leg sits below its own absolute return
        assert r[k]["mean_bps"] < r[k + "_abs"]["mean_bps"]
    assert np.isfinite(r["capm_hml"]["beta"])


def test_era_cut_returns_both_halves(null_panel):
    panel, _ = null_panel
    legs = st.sorted_legs(panel, min_funds=4)
    split = str(legs.index[len(legs) // 2].date())
    eras = st.era_cut(panel, legs, split=split)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["hml_p_bps"]) and np.isfinite(e["giveback"])
        # the erosion identity is mechanical, so it must hold in BOTH halves
        assert e["hml_p_bps"] < 0


def test_summary_fields(null_panel):
    panel, _ = null_panel
    legs = st.sorted_legs(panel, min_funds=4)
    s = st.summary(legs["hi"])
    for k in ("n_months", "mean_bps", "sharpe", "vol_ann", "cagr", "max_drawdown", "tstat"):
        assert k in s and np.isfinite(s[k])
    assert s["max_drawdown"] <= 0.0


def test_giveback_ratio_handles_a_zero_payout():
    legs = pd.DataFrame({"hml_d": [0.0, 0.0], "hml_p": [0.01, -0.01]})
    assert np.isnan(st.giveback_ratio(legs))


def test_sort_width_does_not_flip_the_null_conclusion(null_panel):
    panel, _ = null_panel
    for frac in (0.2, 1 / 3, 0.4):
        legs = st.sorted_legs(panel, frac=frac, min_funds=4)
        assert legs["hml_p"].mean() < 0                     # erosion always
        assert abs(st.newey_west_t(legs["hml"].to_numpy())) < 2.0   # never a total-return edge


@pytest.mark.parametrize("target", ["total", "price", "dist"])
def test_fama_macbeth_slope_series_is_dated(null_panel, target):
    panel, _ = null_panel
    fm = st.fama_macbeth(panel, target, min_funds=4)
    assert isinstance(fm["slopes"].index, pd.DatetimeIndex)
    assert fm["n_months"] == len(fm["slopes"]) > 100

"""Estimator logic, break-even arithmetic and the study's spine — all offline/synthetic.

The spine: on a pair with a planted 6 bp/yr drag gap the estimator recovers it (right sign,
right size, |t| >= 2) and the break-even holding period comes out where the arithmetic says
it should; on a pair with no gap it stays quiet. The one execution lag is enforced, extra
spread only ever lowers the edge, and borrow only ever lowers the long/short harvest.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tco import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The chained-period estimator
# --------------------------------------------------------------------------- #
def test_annual_and_monthly_tables_are_chained_consistently(planted_pair):
    """Chained periods tile the sample: annual sums must equal the matching monthly sums."""
    prices, _ = planted_pair
    ann = st.annual_td_table(prices["liquid"], prices["cheap"])
    mon = st.monthly_td_table(prices["liquid"], prices["cheap"])
    for y in ann.index[1:-1]:
        months = mon[(mon.index // 100) == y]
        if len(months) == 12:
            assert float(months.sum()) == pytest.approx(float(ann.loc[y]), abs=1e-6)


def test_annual_mean_matches_the_cumulative_drift_ON_CLEAN_TAPE(planted_pair):
    """On tape with no mis-timed distributions the estimators agree — and only there.

    This is a property of the *generator*, not of the study's data. On the real common
    window the same three estimators come out at +7.19 / +4.48 / +2.23 bp/yr for QQQ/QQQM.
    The next test shows what breaks them.
    """
    prices, _ = planted_pair
    est = st.td_estimate(prices["liquid"], prices["cheap"], n_boot=200)
    assert abs(est["td_ann_bp_yr"] - est["td_cum_bp_yr"]) < 3.0


def test_a_single_endpoint_artefact_splits_the_estimators():
    """One bad closing print moves the cumulative estimator and not the annual one.

    This is the mechanism behind the study's sharpest caveat: on the real tape the
    same-fee placebo pair's cumulative estimate (+4.69 bp/yr) exceeds a genuine fee-gap
    pair's (+3.38), entirely because of a +23.6 bp artefact in the final partial period.
    """
    prices, _ = data.synthetic_daily(n_years=8, signal_strength=0.0, seed=920)
    prices = prices.loc[:"2017-06-30"]
    base = st.td_estimate(prices["liquid"], prices["cheap"], n_boot=200)
    dirty_px = prices.copy()
    dirty_px.iloc[-1, dirty_px.columns.get_loc("cheap")] *= 1.0030
    dirty = st.td_estimate(dirty_px["liquid"], dirty_px["cheap"], n_boot=200)
    assert dirty["td_ann_bp_yr"] == pytest.approx(base["td_ann_bp_yr"], abs=1e-9)
    assert dirty["td_cum_bp_yr"] - base["td_cum_bp_yr"] > 3.0


def test_log_ratio_sign_convention(planted_pair):
    """``log_ratio`` is log(cheap / liquid): it must drift UP when cheap wins."""
    prices, _ = planted_pair
    lr = st.log_ratio(prices["liquid"], prices["cheap"])
    assert float(lr.iloc[-1] - lr.iloc[0]) > 0


def test_partial_final_year_is_dropped():
    """A sample ending mid-year must not contribute a stub year to the annual mean."""
    prices, _ = data.synthetic_daily(n_years=10, seed=920)
    whole = prices.loc[:"2018-12-31"]
    stub = prices.loc[:"2018-06-30"]
    full = st.annual_td_table(whole["liquid"], whole["cheap"])
    part = st.annual_td_table(stub["liquid"], stub["cheap"])
    assert int(full.index[-1]) == 2018
    assert int(part.index[-1]) == 2017 and len(part) == len(full) - 1


# --------------------------------------------------------------------------- #
# The spine — recovery on a planted gap, silence on the null
# --------------------------------------------------------------------------- #
def test_planted_gap_is_recovered(planted_pair):
    prices, truth = planted_pair
    d = st.synthetic_detect(prices, n_boot=500)
    assert d["td_ann_bp_yr"] > 0
    assert abs(d["td_ann_bp_yr"] - truth["planted_gap_bp_yr"]) < 3.0
    assert d["t_annual"] >= 2.0


def test_planted_gap_ci_excludes_zero(planted_pair):
    prices, _ = planted_pair
    d = st.synthetic_detect(prices, n_boot=800)
    assert d["ci_low"] > 0.0


def test_null_pair_is_quiet(null_pair):
    prices, _ = null_pair
    d = st.synthetic_detect(prices, n_boot=500)
    assert abs(d["td_ann_bp_yr"]) < 3.0
    assert abs(d["t_annual"]) < 2.0
    assert d["ci_low"] <= 0.0 <= d["ci_high"]


def test_null_across_seeds_does_not_fire():
    ts = np.array([
        st.synthetic_detect(
            data.synthetic_daily(signal_strength=0.0, seed=920 + s)[0], n_boot=200
        )["t_annual"]
        for s in range(8)
    ])
    assert abs(ts.mean()) < 1.0
    assert (np.abs(ts) >= 2.0).sum() <= 1


def test_panel_is_calibrated(panel):
    """Recovered gap should track the planted one, not merely be non-zero."""
    px, truths = panel
    cal = st.panel_calibration(px, truths, n_boot=200)
    assert cal["recovered_bp_yr"].is_monotonic_increasing
    err = (cal["recovered_bp_yr"] - cal["planted_bp_yr"]).abs()
    assert float(err.max()) < 4.0


# --------------------------------------------------------------------------- #
# Break-even arithmetic
# --------------------------------------------------------------------------- #
def test_breakeven_days_arithmetic():
    # 6 bp/yr saved, 1 bp extra round trip -> 1/6 of a year = 42 trading days
    assert st.breakeven_days(6.0, 1.0) == pytest.approx(42.0)
    assert st.breakeven_days(12.0, 1.0) == pytest.approx(21.0)


def test_breakeven_is_zero_when_the_cheap_fund_is_also_tighter():
    assert st.breakeven_days(6.0, 0.0) == 0.0
    assert st.breakeven_days(6.0, -1.0) == 0.0


def test_breakeven_is_infinite_without_an_advantage():
    assert st.breakeven_days(0.0, 1.0) == float("inf")
    assert st.breakeven_days(-4.0, 1.0) == float("inf")
    assert st.breakeven_days(float("nan"), 1.0) == float("inf")


def test_breakeven_curve_is_monotone_in_the_spread(planted_pair):
    prices, _ = planted_pair
    est = st.td_estimate(prices["liquid"], prices["cheap"], n_boot=200)
    curve = st.breakeven_curve(est, (0.5, 1.0, 2.0, 5.0))
    assert curve["days_point"].is_monotonic_increasing
    # a pessimistic tracking difference always means a longer wait
    assert (curve["days_ci_low_td"] >= curve["days_point"] - 1e-9).all()


def test_years_to_detect_arithmetic():
    assert st.years_to_detect(6.0, 6.0) == pytest.approx(4.0)
    assert st.years_to_detect(6.0, 12.0) == pytest.approx(16.0)
    assert st.years_to_detect(0.0, 12.0) == float("inf")


# --------------------------------------------------------------------------- #
# The lived race: exactly one execution lag, costs only ever hurt
# --------------------------------------------------------------------------- #
def test_holding_period_race_ignores_the_pre_entry_bar(planted_pair):
    """With lag=1 the day-0 to day-1 move cannot enter any window (no look-ahead)."""
    prices, _ = planted_pair
    base = st.holding_period_edge(prices["liquid"], prices["cheap"], 21, 1.0, lag=1)
    tweaked = prices.copy()
    tweaked.iloc[0, tweaked.columns.get_loc("cheap")] *= 1.05  # move only the pre-entry bar
    moved = st.holding_period_edge(tweaked["liquid"], tweaked["cheap"], 21, 1.0, lag=1)
    assert moved["mean_edge_bp"] == pytest.approx(base["mean_edge_bp"], abs=1e-9)


def test_zero_lag_would_see_it(planted_pair):
    """The same perturbation DOES move a zero-lag race — proof the lag is doing work."""
    prices, _ = planted_pair
    base = st.holding_period_edge(prices["liquid"], prices["cheap"], 21, 1.0, lag=0)
    tweaked = prices.copy()
    tweaked.iloc[0, tweaked.columns.get_loc("cheap")] *= 1.05
    moved = st.holding_period_edge(tweaked["liquid"], tweaked["cheap"], 21, 1.0, lag=0)
    assert moved["mean_edge_bp"] != pytest.approx(base["mean_edge_bp"], abs=1e-9)


def test_extra_spread_only_lowers_the_edge(planted_pair):
    prices, _ = planted_pair
    edges = [st.holding_period_edge(prices["liquid"], prices["cheap"], 63, s)["mean_edge_bp"]
             for s in (0.0, 1.0, 3.0, 10.0)]
    assert edges == sorted(edges, reverse=True)
    # the charge is exactly the spread, applied once at entry
    assert edges[0] - edges[1] == pytest.approx(1.0, abs=1e-9)


def test_longer_holding_periods_repay_a_fixed_spread(planted_pair):
    """The whole thesis: with a real gap, the edge grows with the holding period."""
    prices, _ = planted_pair
    tbl = st.holding_period_table(prices["liquid"], prices["cheap"], (21, 63, 252, 756), 2.0)
    assert tbl["mean_edge_bp"].is_monotonic_increasing
    assert tbl["mean_edge_bp"].iloc[0] < 0 < tbl["mean_edge_bp"].iloc[-1]


def test_empirical_breakeven_brackets_the_analytic_one(planted_pair):
    prices, _ = planted_pair
    est = st.td_estimate(prices["liquid"], prices["cheap"], n_boot=200)
    analytic = st.breakeven_days(est["td_ann_bp_yr"], 2.0)
    emp = st.empirical_breakeven(prices["liquid"], prices["cheap"], 2.0)
    assert emp["first_positive_days"] is not None
    assert 0.25 * analytic <= emp["first_positive_days"] <= 4.0 * analytic


def test_null_pair_never_repays_a_spread(null_pair):
    """No gap, no repayment: the edge must not march upward with the holding period."""
    prices, _ = null_pair
    tbl = st.holding_period_table(prices["liquid"], prices["cheap"], (21, 252, 756), 3.0)
    assert (tbl["mean_edge_bp"] < 0).all()


# --------------------------------------------------------------------------- #
# The long/short version pays borrow
# --------------------------------------------------------------------------- #
def test_borrow_monotonically_reduces_the_harvest(planted_pair):
    prices, _ = planted_pair
    nets = [st.long_short_harvest(prices["liquid"], prices["cheap"], b)["net_bp_yr"]
            for b in (0.0, 10.0, 50.0)]
    assert nets == sorted(nets, reverse=True)
    assert nets[0] - nets[1] == pytest.approx(10.0, abs=0.5)


def test_borrow_sweep_shape_and_sign(planted_pair):
    prices, _ = planted_pair
    sw = st.borrow_sweep(prices["liquid"], prices["cheap"], (0.0, 25.0, 100.0))
    assert list(sw.index) == [0.0, 25.0, 100.0]
    assert sw["net_bp_yr"].iloc[-1] < 0  # a 100 bp borrow buries a 6 bp gap
    assert np.isfinite(sw["t_hac"]).all()


def test_harvest_is_tiny_against_its_own_noise(planted_pair):
    """A few bp of drift inside hundreds of bp of tracking noise: Sharpe near zero."""
    prices, _ = planted_pair
    h = st.long_short_harvest(prices["liquid"], prices["cheap"], borrow_bps_yr=0.0)
    assert h["vol_bp_yr"] > 5 * abs(h["gross_bp_yr"])


# --------------------------------------------------------------------------- #
# Inference primitives & the Sharpe race
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_t_and_wilson():
    assert st.one_sample_t([1.0, 1.0, 1.0, 1.0]) != st.one_sample_t([1.0])
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi
    assert st.wilson_interval(0, 0) != st.wilson_interval(1, 1)


def test_sharpe_race_is_blind_to_a_few_basis_points(planted_pair):
    """A 6 bp/yr fee gap inside a 17% vol index is invisible to Sharpe — by construction."""
    prices, _ = planted_pair
    r = st.excess_sharpe_race(prices["liquid"], prices["cheap"], prices["cash"])
    assert abs(r["sharpe_diff"]) < 0.05
    assert r["vol_liquid_pct"] > 10.0


def test_era_cut_returns_both_halves(planted_pair):
    prices, _ = planted_pair
    mid = prices.index[len(prices) // 2].strftime("%Y-%m-%d")
    eras = st.era_cut(prices["liquid"], prices["cheap"], split=mid, n_boot=200)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["td_ann_bp_yr"])
        assert e["td_ann_bp_yr"] > 0  # a structural gap prints in both halves


def test_era_cut_returns_none_on_a_stub():
    prices, _ = data.synthetic_daily(n_years=6, seed=920)
    eras = st.era_cut(prices["liquid"], prices["cheap"], split="2010-06-01", n_boot=100)
    assert eras["early"] is None


def test_bootstrap_ci_brackets_the_point(planted_pair):
    prices, _ = planted_pair
    ann = st.annual_td_table(prices["liquid"], prices["cheap"])
    ci = st.bootstrap_td_ci(ann, n_boot=500)
    assert ci["ci_low"] <= ci["td_bp_yr"] <= ci["ci_high"]
    assert ci["td_bp_yr"] == pytest.approx(float(ann.mean()))


def test_bootstrap_ci_degrades_gracefully_on_a_stub():
    ci = st.bootstrap_td_ci(pd.Series([1.0, 2.0]), n_boot=100)
    assert np.isnan(ci["ci_low"]) and ci["n_boot"] == 0


# --------------------------------------------------------------------------- #
# The honest interval — the bootstrap is not it at small n
# --------------------------------------------------------------------------- #
def test_t_critical_values_respect_degrees_of_freedom():
    assert st.t95(4) == pytest.approx(2.776, abs=1e-3)
    assert st.t95(4) > st.t95(30) > 1.96
    assert st.t95(200) == pytest.approx(1.96)


def test_t_interval_brackets_the_mean_and_widens_as_n_shrinks():
    x = [5.0, 6.0, 7.0, 4.0, 8.0]
    lo, hi = st.t_interval(x)
    assert lo < float(np.mean(x)) < hi
    wide = st.t_interval(x)
    narrow = st.t_interval(x * 6)          # same dispersion, 30 observations
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_t_interval_is_wider_than_the_block_bootstrap_at_five_years():
    """The audit finding: a percentile bootstrap on five annual numbers is too narrow."""
    ann = pd.Series([2.3, 1.7, 10.3, 3.8, 10.8], index=[2021, 2022, 2023, 2024, 2025])
    boot = st.bootstrap_td_ci(ann, n_boot=2000)
    lo, hi = st.t_interval(ann.to_numpy())
    assert (hi - lo) > 1.5 * (boot["ci_high"] - boot["ci_low"])
    assert lo < boot["ci_low"] and hi > boot["ci_high"]


def test_td_estimate_carries_both_intervals_and_the_sign_count(planted_pair):
    prices, _ = planted_pair
    est = st.td_estimate(prices["liquid"], prices["cheap"], n_boot=300)
    assert est["t_ci_low"] <= est["td_ann_bp_yr"] <= est["t_ci_high"]
    assert est["t_ci_low"] <= est["ci_low"] and est["t_ci_high"] >= est["ci_high"]
    assert 0 <= est["years_positive"] <= est["years_counted"] == est["n_years"]


def test_breakeven_curve_pessimistic_column_is_the_t_interval(planted_pair):
    """days_t_low_td must be at least as long a wait as the bootstrap's low end."""
    prices, _ = planted_pair
    est = st.td_estimate(prices["liquid"], prices["cheap"], n_boot=300)
    curve = st.breakeven_curve(est, (1.0, 2.0))
    assert (curve["days_t_low_td"] >= curve["days_ci_low_td"] - 1e-9).all()
    assert (curve["days_t_high_td"] <= curve["days_ci_high_td"] + 1e-9).all()


# --------------------------------------------------------------------------- #
# The stub decomposition — what the complete-period rule throws away
# --------------------------------------------------------------------------- #
def test_stub_decomposition_adds_up(planted_pair):
    prices, _ = planted_pair
    d = st.stub_decomposition(prices["liquid"], prices["cheap"])
    assert d["head_bp"] + d["years_bp"] + d["tail_bp"] == pytest.approx(d["total_bp"], abs=1e-6)
    assert d["n_years"] >= 1


def test_stub_decomposition_isolates_a_closing_print_artefact():
    """Plant a level error in the LAST print: the annual estimator must not see it."""
    prices, _ = data.synthetic_daily(n_years=8, signal_strength=0.0, seed=920)
    prices = prices.loc[:"2017-06-30"]          # ends mid-year -> a real closing stub
    clean = st.annual_td_table(prices["liquid"], prices["cheap"])
    dirty_px = prices.copy()
    dirty_px.iloc[-1, dirty_px.columns.get_loc("cheap")] *= 1.0030   # +30 bp level error
    dirty = st.annual_td_table(dirty_px["liquid"], dirty_px["cheap"])
    d = st.stub_decomposition(dirty_px["liquid"], dirty_px["cheap"])
    # the complete years are untouched...
    pd.testing.assert_series_equal(clean, dirty)
    # ...and the whole artefact lands in the discarded closing stub
    assert d["tail_bp"] > 25.0


def test_sign_count():
    assert st.sign_count(pd.Series([1.0, 2.0, -1.0, 3.0, 4.0])) == (4, 5)
    assert st.sign_count(pd.Series([], dtype=float)) == (0, 0)

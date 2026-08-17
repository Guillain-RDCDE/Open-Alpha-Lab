"""Estimator logic, ledger arithmetic and the study's spine — all offline/synthetic.

The spine: on a panel with a *planted* drag spread the fund-vs-fund estimator recovers
that spread inside a tight bootstrap interval, while the fund-vs-coin estimator — fed
the same world through a mis-timed coin close — comes back an order of magnitude wider
and cannot resolve it. On the null (identical wrappers) the fund-vs-fund estimate
collapses to ~0. Costs and borrow only ever reduce the capture trade, and the single
execution lag is enforced.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unstaked import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Estimator mechanics
# --------------------------------------------------------------------------- #
def test_log_diff_is_zero_for_identical_series(planted):
    prices, _ = planted
    d = st.log_diff(prices["cheap"], prices["cheap"])
    assert np.allclose(d.to_numpy(), 0.0)


def test_log_diff_recovers_a_pure_constant_drag():
    n = 500
    idx = pd.bdate_range("2024-07-23", periods=n)
    base = pd.Series(np.linspace(100.0, 180.0, n), index=idx)
    drag = 0.02  # 2%/yr
    lagged = base * np.exp(-drag / 252.0 * np.arange(n))
    d = st.log_diff(lagged, base)
    assert d.mean() * 252 * 100 == pytest.approx(-2.0, abs=1e-6)


def test_tracking_difference_reports_both_measures(planted):
    prices, _ = planted
    td = st.tracking_difference(prices["dear"], prices["cheap"])
    for k in ("ann_td_pct", "ann_td_geo_pct", "iid_t", "hac_t", "monthly_t", "daily_sd_bps"):
        assert np.isfinite(td[k])
    # arithmetic-mean and endpoint measures agree to well inside a percentage point here
    assert abs(td["ann_td_pct"] - td["ann_td_geo_pct"]) < 1.0


def test_tracking_difference_short_sample_returns_nan():
    idx = pd.bdate_range("2024-07-23", periods=10)
    s = pd.Series(np.linspace(100, 101, 10), index=idx)
    td = st.tracking_difference(s, s * 1.01)
    assert np.isnan(td["ann_td_pct"])


def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_and_welch_finite(planted):
    prices, _ = planted
    d = st.log_diff(prices["dear"], prices["cheap"])
    e = st.log_diff(prices["cheap"], prices["coin"])
    assert np.isfinite(st.one_sample_t(d.to_numpy()))
    assert np.isfinite(st.welch_t(d.to_numpy(), e.to_numpy()))


def test_bootstrap_ci_brackets_the_point_estimate(planted):
    prices, _ = planted
    d = st.log_diff(prices["dear"], prices["cheap"])
    ci = st.bootstrap_td_ci(d, n_boot=800, seed=960)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["half_width"] > 0


def test_bootstrap_ci_is_reproducible(planted):
    prices, _ = planted
    d = st.log_diff(prices["dear"], prices["cheap"])
    a = st.bootstrap_td_ci(d, n_boot=500, seed=960)
    b = st.bootstrap_td_ci(d, n_boot=500, seed=960)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]


# --------------------------------------------------------------------------- #
# The ledger — measured tape vs declared assumptions
# --------------------------------------------------------------------------- #
def test_ledger_adds_the_staking_assumption_on_top_of_the_tape():
    """The staking yield is absent from the coin benchmark, so it can only be added."""
    led = st.wrapper_ledger(td_vs_coin_pct=-0.25, expense_ratio=0.0025, staking_yield=0.03)
    assert led["total_vs_unstaked_holder_pct"] == pytest.approx(-0.25)
    assert led["total_vs_staked_holder_pct"] == pytest.approx(-3.25)
    assert led["staking_share_of_declared_cost"] == pytest.approx(3.0 / 3.25)


def test_ledger_unexplained_is_measured_minus_expected_fee():
    led = st.wrapper_ledger(td_vs_coin_pct=-1.40, expense_ratio=0.025, staking_yield=0.03)
    assert led["unexplained_pct"] == pytest.approx(1.10)


def test_staking_sweep_moves_only_the_staking_leg():
    rows = st.staking_sweep(-0.25, 0.0025, yields=(0.02, 0.03, 0.045))
    assert [r["staking_pct_assumed"] for r in rows] == [2.0, 3.0, 4.5]
    assert all(r["td_vs_coin_pct"] == -0.25 for r in rows)
    totals = [r["total_vs_staked_holder_pct"] for r in rows]
    assert totals[0] > totals[1] > totals[2]  # a fatter assumed yield = a deeper shortfall
    shares = [r["staking_share_of_declared_cost"] for r in rows]
    assert shares[0] < shares[1] < shares[2]


def test_fee_sweep_moves_only_the_fee_leg():
    rows = st.fee_sweep(-0.25, 0.03, fees=(0.0012, 0.0025))
    assert rows[0]["staking_share_of_declared_cost"] > rows[1]["staking_share_of_declared_cost"]
    assert rows[0]["total_vs_staked_holder_pct"] == rows[1]["total_vs_staked_holder_pct"]


# --------------------------------------------------------------------------- #
# The capture trade — one lag, costs and borrow only ever hurt
# --------------------------------------------------------------------------- #
def test_capture_trade_has_exactly_one_execution_lag(planted):
    prices, truth = planted
    res = st.long_short_capture(prices["cheap"], prices["dear"], prices["cash"])
    # pct_change drops one bar, the lag drops one more
    assert res["n_days"] == truth["n_days"] - 2


def test_borrow_and_costs_monotonically_reduce_the_net(planted):
    prices, _ = planted
    nets = [r["ann_net_pct"] for r in st.borrow_sweep(prices["cheap"], prices["dear"],
                                                      prices["cash"])]
    assert all(nets[i] > nets[i + 1] for i in range(len(nets) - 1))
    cheap_cost = st.long_short_capture(prices["cheap"], prices["dear"], prices["cash"],
                                       cost_bps=1.0)["ann_net_pct"]
    dear_cost = st.long_short_capture(prices["cheap"], prices["dear"], prices["cash"],
                                      cost_bps=50.0)["ann_net_pct"]
    assert cheap_cost > dear_cost


def test_capture_trade_recovers_roughly_the_planted_spread(planted):
    prices, truth = planted
    res = st.long_short_capture(prices["cheap"], prices["dear"], prices["cash"],
                                borrow_bps_ann=0.0, cost_bps=0.0)
    assert res["ann_gross_pct"] == pytest.approx(truth["spread_planted_pct"], abs=0.6)


# --------------------------------------------------------------------------- #
# Era cut, calendar table, excess-of-cash race
# --------------------------------------------------------------------------- #
def test_era_cut_returns_both_halves(planted):
    prices, _ = planted
    eras = st.era_cut(prices["dear"], prices["cheap"], split="2025-07-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["ann_td_pct"]) and e["ci_low"] < e["ci_high"]


def test_calendar_year_table_shape(planted):
    prices, _ = planted
    tbl = st.calendar_year_table(prices["dear"], prices["cheap"])
    assert {"n_days", "ann_td_pct"}.issubset(tbl.columns)
    assert len(tbl) >= 2


def test_excess_race_is_excess_of_cash_on_both_arms(planted):
    prices, _ = planted
    race = st.excess_sharpe_race(prices["cheap"], prices["coin"], prices["cash"])
    assert np.isfinite(race["excess_sharpe_adv"]) and np.isfinite(race["t_diff"])
    # both arms hold the same asset, so the excess-Sharpe gap must be tiny
    assert abs(race["excess_sharpe_adv"]) < 0.5


# --------------------------------------------------------------------------- #
# The study's spine
# --------------------------------------------------------------------------- #
def test_planted_spread_is_recovered_fund_vs_fund(planted):
    prices, truth = planted
    d = st.synthetic_detect(prices)
    assert d["fund_vs_fund_pct"] == pytest.approx(-truth["spread_planted_pct"], abs=0.5)
    lo, hi = d["fund_vs_fund_ci"]
    assert hi < 0.0  # the interval excludes zero: the drag is genuinely detected


def test_fund_vs_coin_cannot_resolve_the_same_spread(planted):
    """Same world, same truth — the mis-timed coin benchmark loses the signal entirely."""
    prices, truth = planted
    d = st.synthetic_detect(prices)
    lo, hi = d["fund_vs_coin_ci"]
    assert lo < 0.0 < hi                                  # interval spans zero
    assert d["fund_vs_coin_halfwidth"] > 5.0 * d["fund_vs_fund_halfwidth"]
    assert d["fund_vs_coin_halfwidth"] > truth["spread_planted_pct"]


def test_null_fund_vs_fund_estimate_is_quiet(null_panel):
    prices, _ = null_panel
    d = st.synthetic_detect(prices)
    assert abs(d["fund_vs_fund_pct"]) < 0.5


def test_null_across_seeds_is_centred(null_panel):
    ests = np.array([
        st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=960 + s)[0])["fund_vs_fund_pct"]
        for s in range(6)
    ])
    assert abs(ests.mean()) < 0.25
    assert (np.abs(ests) >= 0.5).sum() == 0


def test_resolution_limit_is_ruler_dependent_against_the_coin(planted):
    """The audit finding the study must never re-hide: quoting one block quotes a choice.

    On a mean-reverting difference the bootstrap half-width falls monotonically as the
    block lengthens, so a single 'the tape resolves +/-X' number is a ruler, not a fact.
    """
    prices, _ = planted
    d = st.log_diff(prices["cheap"], prices["coin"])
    sw = st.resolution_sweep(d, blocks=(5, 21, 63))
    widths = [sw["blocks"][b] for b in (5, 21, 63)]
    assert widths[0] > widths[1] > widths[2]
    assert np.isfinite(sw["monthly"]) and sw["monthly"] > 0
    assert sw["max"] > 2.0 * sw["min"]           # the choice is worth more than a factor of two
    assert sw["min"] <= sw["blocks"][21] <= sw["max"]


def test_monthly_halfwidth_is_the_narrow_end_and_still_beats_the_effect(planted):
    """Even the narrowest honest ruler must fail to resolve the planted spread vs the coin."""
    prices, truth = planted
    d = st.log_diff(prices["cheap"], prices["coin"])
    hw = st.monthly_halfwidth(d)
    assert hw > truth["spread_planted_pct"]      # unresolvable on the friendliest ruler too


def test_monthly_halfwidth_is_tight_on_the_same_close_pair(planted):
    """Fund-vs-fund: every ruler must agree, which is what makes that leg quotable."""
    prices, truth = planted
    d = st.log_diff(prices["dear"], prices["cheap"])
    sw = st.resolution_sweep(d, blocks=(5, 21, 63))
    assert sw["max"] < truth["spread_planted_pct"]   # resolvable on the *worst* ruler


def test_capture_trade_reports_a_non_overlapping_monthly_t(planted):
    """HAC is the flattering direction here, so the blunt ruler must be carried too."""
    prices, _ = planted
    res = st.long_short_capture(prices["cheap"], prices["dear"], prices["cash"])
    assert np.isfinite(res["monthly_t_gross"])
    assert res["monthly_n"] > 12
    assert 0 <= res["months_positive"] <= res["monthly_n"]


def test_geometric_reading_is_not_the_holders_benefit(planted):
    """The simple-return difference is level-dependent; the log measure is the wealth one."""
    prices, _ = planted
    td = st.tracking_difference(prices["dear"], prices["cheap"])
    logged = np.log(prices["dear"].iloc[-1] / prices["dear"].iloc[0]) - np.log(
        prices["cheap"].iloc[-1] / prices["cheap"].iloc[0])
    years = td["n_days"] / st.TRADING_DAYS
    assert td["ann_td_pct"] == pytest.approx(float(logged) / years * 100.0, abs=0.05)


def test_resolvable_drag_is_far_larger_against_the_coin(planted):
    prices, _ = planted
    against_coin = st.resolvable_drag(st.log_diff(prices["cheap"], prices["coin"]))
    against_fund = st.resolvable_drag(st.log_diff(prices["dear"], prices["cheap"]))
    assert against_coin > 5.0 * against_fund

"""Book mechanics, inference primitives, and the study's spine — all offline/synthetic.

The spine: an equal-dollar, daily-rebalanced short of both levered legs recovers the
funds' **fee load** and nothing else. On the costly-funds tape the recovered harvest must
match the planted load; on the free-funds null — where the volatility decay is *identical*
and enormous — the harvest must collapse to zero. Borrow scales the harvest away linearly,
costs only reduce it, and the rebalance schedule is the risk dial, not a return dial.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from short_pair import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Rebalance calendars
# --------------------------------------------------------------------------- #
def test_rebalance_flags_frequencies():
    idx = pd.bdate_range("2020-01-01", periods=500)
    assert st.rebalance_flags(idx, "daily").all()
    assert not st.rebalance_flags(idx, "none").any()
    monthly = st.rebalance_flags(idx, "monthly")
    weekly = st.rebalance_flags(idx, "weekly")
    # ~2 years of business days: ~24 month starts, ~100 week starts.
    assert 20 <= monthly.sum() <= 26
    assert 90 <= weekly.sum() <= 110
    # Every month-start is also a week boundary or later — monthly is the sparser schedule.
    assert monthly.sum() < weekly.sum() < len(idx)


def test_rebalance_flags_reject_unknown():
    with pytest.raises(ValueError):
        st.rebalance_flags(pd.bdate_range("2020-01-01", periods=10), "quarterly")


def test_rebalance_flag_never_fires_on_the_first_bar():
    idx = pd.bdate_range("2020-01-01", periods=200)
    for freq in ("weekly", "monthly", "none"):
        assert not st.rebalance_flags(idx, freq)[0]


# --------------------------------------------------------------------------- #
# Book mechanics
# --------------------------------------------------------------------------- #
def test_book_columns_and_no_nans(costly_funds):
    prices, _ = costly_funds
    bt = st.run_short_pair(prices["lev_long"], prices["lev_short"], prices["cash"])
    assert {"r_nav", "r_cash", "e_nav", "gross_short", "turnover"}.issubset(bt.columns)
    assert not bt[["r_nav", "r_cash", "e_nav"]].isna().any().any()


def test_excess_is_nav_minus_cash(costly_funds):
    prices, _ = costly_funds
    bt = st.run_short_pair(prices["lev_long"], prices["lev_short"], prices["cash"])
    assert np.allclose(bt["e_nav"], bt["r_nav"] - bt["r_cash"])


def test_shorting_a_riskless_leg_pair_earns_nothing():
    """Sanity: if both 'funds' simply track cash, the short-pair excess return is 0."""
    idx = pd.bdate_range("2015-01-01", periods=800)
    cash = pd.Series(100 * (1.0 + 0.03 / 252) ** np.arange(800), index=idx)
    bt = st.run_short_pair(cash.copy(), cash.copy(), cash, cost_bps=0.0)
    assert abs(float(bt["e_nav"].mean()) * 252) < 1e-9


def test_daily_reset_recovers_the_planted_fee_load(costly_funds):
    prices, truth = costly_funds
    cmp = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"], cost_bps=0.0)
    assert cmp["ann_mean"] == pytest.approx(truth["expected_harvest_ann"], abs=0.002)
    assert cmp["t_hac"] > 5.0


def test_null_free_funds_harvest_nothing(free_funds):
    """The decay is unchanged and huge; with no fee load there is nothing to collect."""
    prices, truth = free_funds
    assert truth["decay_long_ann"] > 0.05          # the decay is genuinely there
    cmp = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"], cost_bps=0.0)
    assert abs(cmp["ann_mean"]) < 0.003
    assert abs(cmp["t_hac"]) < 2.0


def test_null_quiet_across_seeds():
    means, ts = [], []
    for prices, _ in data.synthetic_panel(n_seeds=6, signal_strength=0.0):
        d = st.synthetic_detect(prices, cost_bps=0.0)
        means.append(d["ann_mean"]); ts.append(d["t_hac"])
    assert abs(np.mean(means)) < 0.003
    assert max(abs(t) for t in ts) < 2.0


def test_planted_recovered_across_seeds():
    for prices, truth in data.synthetic_panel(n_seeds=4, signal_strength=1.0):
        d = st.synthetic_detect(prices, cost_bps=0.0)
        assert d["ann_mean"] == pytest.approx(truth["expected_harvest_ann"], abs=0.003)
        assert d["t_hac"] > 5.0


def test_pair_short_is_market_neutral(costly_funds):
    prices, _ = costly_funds
    cmp = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"],
                     bench=prices["under"], cost_bps=0.0)
    assert abs(cmp["beta"]["beta"]) < 0.05


def test_borrow_reduces_harvest_one_for_one(costly_funds):
    prices, _ = costly_funds
    m0 = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"],
                    borrow_ann=0.0, cost_bps=0.0)["ann_mean"]
    m2 = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"],
                    borrow_ann=0.02, cost_bps=0.0)["ann_mean"]
    # Borrow is charged on 1.0x NAV of gross short, so 2%/yr costs ~2 pp/yr.
    assert m0 - m2 == pytest.approx(0.02, abs=0.002)


def test_no_rebate_costs_the_cash_rate(costly_funds):
    """A retail account that earns nothing on its short proceeds loses the cash rate."""
    prices, truth = costly_funds
    full = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"],
                      rebate=1.0, cost_bps=0.0)["ann_mean"]
    none = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"],
                      rebate=0.0, cost_bps=0.0)["ann_mean"]
    assert full - none == pytest.approx(truth["cash_rate_ann"], abs=0.003)


def test_costs_monotonically_reduce_harvest(costly_funds):
    prices, _ = costly_funds
    means = [r["ann_mean"] for r in st.cost_sweep(prices["lev_long"], prices["lev_short"],
                                                  prices["cash"], grid=(0.0, 2.0, 10.0))]
    assert means[0] > means[1] > means[2]


def test_breakeven_borrow_matches_the_harvest(costly_funds):
    prices, _ = costly_funds
    be = st.breakeven_borrow(prices["lev_long"], prices["lev_short"], prices["cash"],
                             cost_bps=0.0)
    m0 = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"],
                    borrow_ann=0.0, cost_bps=0.0)["ann_mean"]
    assert be == pytest.approx(m0 * 100.0, abs=0.15)


def test_unrebalanced_book_is_the_dangerous_one(costly_funds):
    """Never rebalancing lets the losing leg grow without bound — vol and tail explode."""
    prices, _ = costly_funds
    daily = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"],
                       rebalance="daily", cost_bps=0.0)
    never = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"],
                       rebalance="none", cost_bps=0.0)
    assert never["strategy"]["vol_ann"] > 10 * daily["strategy"]["vol_ann"]
    assert never["max_drawdown"] < daily["max_drawdown"]
    assert never["max_gross"] > 3.0


def test_ruin_flag_survives_the_returned_frame(costly_funds):
    """The ruin flag must reach the caller. ``attrs`` propagation through pandas methods is
    version-dependent, so a lost flag would silently read as 'the book survived'."""
    prices, _ = costly_funds
    never = st.run_short_pair(prices["lev_long"], prices["lev_short"], prices["cash"],
                              rebalance="none", cost_bps=0.0)
    daily = st.run_short_pair(prices["lev_long"], prices["lev_short"], prices["cash"],
                              rebalance="daily", cost_bps=0.0)
    assert never.attrs["ruined"] is True
    assert daily.attrs["ruined"] is False
    # And it must survive the trip through compare()/frequency_race.
    rows = {r["rebalance"]: r for r in st.frequency_race(
        prices["lev_long"], prices["lev_short"], prices["cash"], cost_bps=0.0)}
    assert rows["none"]["ruined"] is True
    assert rows["daily"]["ruined"] is False


def test_leg_shortfalls_average_to_the_daily_harvest(costly_funds):
    """The per-leg decomposition must average to the harvest the book actually earns."""
    prices, truth = costly_funds
    legs = st.implied_leg_shortfall(prices["lev_long"], prices["lev_short"],
                                    prices["cash"], bench=prices["under"])
    cmp = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"], cost_bps=0.0)
    assert legs["average"] == pytest.approx(cmp["ann_mean"], abs=0.001)
    # Each leg's planted load is recovered separately (3x long: expense + 2 spreads;
    # -3x inverse: expense + 4 spreads -> the inverse leg carries the LARGER load here).
    assert legs["shortfall_long"] == pytest.approx(truth["drag_long_ann"], abs=0.002)
    assert legs["shortfall_short"] == pytest.approx(truth["drag_short_ann"], abs=0.002)
    assert 0.0 < legs["share_from_long"] < 1.0


def test_leg_shortfall_requires_the_index_leg(costly_funds):
    prices, _ = costly_funds
    with pytest.raises(ValueError):
        st.implied_leg_shortfall(prices["lev_long"], prices["lev_short"], prices["cash"])


def test_execution_lag_is_exactly_one_day(costly_funds):
    """Perturbing the future must not change any earlier NAV return (causality check)."""
    prices, _ = costly_funds
    a = prices["lev_long"].copy()
    b = prices["lev_short"].copy()
    base = st.run_short_pair(a, b, prices["cash"], rebalance="monthly", cost_bps=0.0)
    a2 = a.copy()
    a2.iloc[1200:] *= 1.5     # perturb only the future
    pert = st.run_short_pair(a2, b, prices["cash"], rebalance="monthly", cost_bps=0.0)
    assert np.allclose(base["r_nav"].iloc[:1199], pert["r_nav"].iloc[:1199])


def test_turnover_ordering_by_schedule(costly_funds):
    prices, _ = costly_funds
    rows = {r["rebalance"]: r for r in st.frequency_race(
        prices["lev_long"], prices["lev_short"], prices["cash"], cost_bps=0.0)}
    assert rows["daily"]["turnover_ann"] > rows["weekly"]["turnover_ann"]
    assert rows["weekly"]["turnover_ann"] > rows["monthly"]["turnover_ann"]
    assert rows["none"]["turnover_ann"] == 0.0


# --------------------------------------------------------------------------- #
# Inference primitives sanity
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_welch_and_one_sample_finite(costly_funds):
    prices, _ = costly_funds
    cmp = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"],
                     bench=prices["under"])
    assert np.isfinite(st.one_sample_t(cmp["e_nav"].to_numpy()))
    assert np.isfinite(st.welch_t(cmp["e_nav"].to_numpy(), cmp["e_bench"].to_numpy()))
    assert np.isfinite(st.sharpe_diff_tstat(cmp["e_nav"], cmp["e_bench"]))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_bootstrap_cis_bracket_their_points(costly_funds):
    prices, _ = costly_funds
    cmp = st.compare(prices["lev_long"], prices["lev_short"], prices["cash"], cost_bps=0.0)
    ci_s = st.bootstrap_sharpe_ci(cmp["e_nav"], n_boot=400, seed=941)
    ci_m = st.bootstrap_mean_ci(cmp["e_nav"], n_boot=400, seed=941)
    assert ci_s["ci_low"] <= ci_s["sharpe"] <= ci_s["ci_high"]
    assert ci_m["ci_low"] <= ci_m["mean_ann"] <= ci_m["ci_high"]
    assert ci_m["ci_low"] > 0.0        # the planted harvest is clear of zero


def test_beta_to_recovers_a_known_exposure():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2016-01-01", periods=1500)
    bench = pd.Series(rng.normal(0.0004, 0.01, 1500), index=idx)
    y = 0.0002 + 0.4 * bench + pd.Series(rng.normal(0, 0.004, 1500), index=idx)
    res = st.beta_to(y, bench)
    assert res["beta"] == pytest.approx(0.4, abs=0.05)
    assert res["alpha_ann"] == pytest.approx(0.0002 * 252, abs=0.02)


def test_era_cut_returns_both_halves(costly_funds):
    prices, _ = costly_funds
    eras = st.era_cut(prices["lev_long"], prices["lev_short"], prices["cash"],
                      split="2018-01-01", cost_bps=0.0)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["ann_mean"]) and e["ann_mean"] > 0.01


def test_calendar_year_table_shape(costly_funds):
    prices, _ = costly_funds
    tbl = st.calendar_year_table(prices["lev_long"], prices["lev_short"], prices["cash"])
    assert {"nav_pct", "cash_pct", "excess_pct", "worst_day_pct"}.issubset(tbl.columns)
    assert len(tbl) > 5


def test_summary_reports_the_tail(costly_funds):
    prices, _ = costly_funds
    s = st.summary(st.run_short_pair(prices["lev_long"], prices["lev_short"],
                                     prices["cash"])["e_nav"])
    for k in ("skew", "excess_kurtosis", "worst_day", "worst_month", "max_drawdown"):
        assert np.isfinite(s[k])
    assert s["worst_day"] < 0

"""Estimators, tests and rules — all offline/synthetic.

The spine: on a planted fee ladder the monthly tracking-difference estimator recovers the
planted gap, the fee-rank test fires, and the pass-through slope lands near −1; on a null
where every wrapper charges the same fee (while the published sheet still shows dispersion)
the spread collapses to zero and the rank test stays quiet. Costs only ever reduce the
chase rule's return; the ownership race's switching rule carries exactly one execution lag.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from which_gold import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_and_welch_finite():
    rng = np.random.default_rng(2)
    a, b = rng.normal(0.1, 1, 500), rng.normal(0.0, 1, 500)
    assert np.isfinite(st.one_sample_t(a))
    assert st.welch_t(a, b) > 0


def test_rankdata_averages_ties():
    r = st.rankdata([1.0, 2.0, 2.0, 5.0])
    assert list(r) == [0.0, 1.5, 1.5, 3.0]


def test_spearman_perfect_and_inverse():
    assert st.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert st.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_block_bootstrap_brackets_the_point():
    rng = np.random.default_rng(3)
    x = pd.Series(0.0002 + rng.normal(0, 0.0006, 96))
    ci = st.block_bootstrap_ci(x, n_boot=1000, seed=961)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def test_monthly_log_returns_are_non_overlapping(planted):
    prices, truth = planted
    mm = st.monthly_log_returns(prices[truth["fund_cols"]])
    # bdate_range has ~261 sessions a year, so n_years * 252 bars spans slightly less
    assert len(mm) == pytest.approx(truth["n_years"] * 12, abs=5)
    assert mm.index.is_monotonic_increasing
    assert not mm.index.duplicated().any()


def test_estimators_agree_on_a_clean_planted_pair(planted):
    """Endpoint, daily and monthly must land on the same planted gap (no premium bias)."""
    prices, truth = planted
    cheap = truth["fund_cols"][int(np.argmin(truth["fees_eff_bps"]))]
    dear = truth["fund_cols"][int(np.argmax(truth["fees_eff_bps"]))]
    gap = truth["fee_spread_bps"]
    ps = st.pair_spread(prices[truth["fund_cols"]], cheap, dear)
    for key in ("spread_bpy", "daily_bpy", "endpoint_bpy"):
        assert ps[key] == pytest.approx(gap, abs=8.0)
    assert ps["tstat"] > 2.0


def test_tracking_table_ordering_matches_the_fee_ladder(planted):
    prices, truth = planted
    funds = truth["fund_cols"]
    fees = {f: v for f, v in zip(funds, truth["fees_bps"])}
    tbl = st.tracking_table(prices[funds], funds, fees=fees)
    assert st.spearman(tbl["fee_bps"], tbl["td_monthly_bpy"]) < -0.8
    # the cohort-relative differences are a cross-section: they must sum to ~zero
    assert abs(float(tbl["td_monthly_bpy"].sum())) < 1.0


def test_all_pairs_is_complete_and_antisymmetric(planted):
    prices, truth = planted
    funds = truth["fund_cols"]
    ap = st.all_pairs(prices[funds], funds)
    assert len(ap) == 10
    for _, row in ap.iterrows():
        back = st.pair_spread(prices[funds], row["b"], row["a"])
        assert back["spread_bpy"] == pytest.approx(-row["spread_bpy"], abs=1e-6)


def test_monthly_estimator_reports_both_ts_and_the_naive_one_is_smaller_when_mean_reverting():
    """On a mean-reverting series the HAC *t* is the ANTI-conservative one — pin that down.

    This is the honesty guard for the whole study: a Newey-West *t* is usually the
    conservative choice, and here it is not. If someone ever quotes ``tstat`` alone as
    "the conservative estimate", this test is the thing that says otherwise.
    """
    rng = np.random.default_rng(11)
    e = rng.normal(0, 1e-3, 400)
    x = pd.Series(2e-4 + e[1:] - 0.5 * e[:-1])          # MA(1) with negative acf1
    mo = st.td_monthly(x, pd.Series(0.0, index=x.index))
    assert mo["acf1"] < -0.2
    assert abs(mo["tstat"]) > abs(mo["t_naive"])         # HAC inflates, it does not shrink
    assert mo["t_naive"] == pytest.approx(st.one_sample_t(x.to_numpy()))


def test_hac_lag_sweep_exposes_the_bandwidth_dependence():
    rng = np.random.default_rng(12)
    e = rng.normal(0, 1e-3, 400)
    x = 2e-4 + e[1:] - 0.5 * e[:-1]
    rows = st.hac_lag_sweep(x)
    assert rows[0]["lags"] == 0 and rows[0]["label"].startswith("naive")
    # with negative dependence the HAC t grows monotonically away from the naive one
    assert all(abs(rows[i]["tstat"]) <= abs(rows[i + 1]["tstat"]) + 1e-9
               for i in range(len(rows) - 1))


def test_pair_spread_reports_the_wider_iid_interval(planted):
    """The block bootstrap inherits the mean reversion; the iid one must also be printed."""
    prices, truth = planted
    funds = truth["fund_cols"]
    ps = st.pair_spread(prices[funds], funds[2], funds[0])
    for k in ("ci_low", "ci_high", "ci_iid_low", "ci_iid_high", "t_hac", "monthly_acf1",
              "pos_years", "n_years"):
        assert k in ps
    assert ps["ci_iid_high"] - ps["ci_iid_low"] > 0
    assert 0 <= ps["pos_years"] <= ps["n_years"]


def test_measurement_floor_is_positive_and_scales(planted):
    prices, truth = planted
    funds = truth["fund_cols"]
    floor = st.measurement_floor(prices[funds], funds)
    assert floor["detectable_bpy"] > 0
    assert floor["sd_month_pair_bp"] > 0
    # a noisier world must have a higher floor
    noisy, tn = data.synthetic_panel(premium_innov_bp=20.0, seed=961)
    floor2 = st.measurement_floor(noisy[tn["fund_cols"]], tn["fund_cols"])
    assert floor2["detectable_bpy"] > floor["detectable_bpy"]


# --------------------------------------------------------------------------- #
# The fee-rank test and pass-through
# --------------------------------------------------------------------------- #
def test_rank_test_p_has_a_hard_floor_at_five_funds():
    rt = st.rank_test([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert rt["exact"] and rt["n_draws"] == 120
    assert rt["spearman"] == pytest.approx(-1.0)
    assert rt["p_perm"] == pytest.approx(2.0 / 120.0, abs=1e-9)
    assert rt["p_floor"] == pytest.approx(2.0 / 120.0, abs=1e-9)


def test_pass_through_recovers_a_unit_slope():
    fees = np.array([40.0, 25.0, 10.0, 17.0, 17.5])
    tds = -(fees - fees.mean())
    pt = st.pass_through(fees, tds)
    assert pt["slope"] == pytest.approx(-1.0, abs=1e-6)
    assert pt["r2"] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# What the gap is worth, and the swept assumptions
# --------------------------------------------------------------------------- #
def test_holding_cost_compounds():
    rows = st.holding_cost(30.0)
    pcts = [r["shortfall_pct"] for r in rows]
    assert pcts == sorted(pcts)
    assert rows[0]["shortfall_pct"] == pytest.approx(0.30, abs=0.01)
    # ten years of 30 bp is close to, and just under, ten times one year
    assert 2.9 < [r for r in rows if r["years"] == 10][0]["shortfall_pct"] < 3.0


def test_spread_break_even_is_monotone_in_cost():
    rows = st.spread_break_even(26.0)
    days = [r["break_even_days"] for r in rows]
    assert days == sorted(days)
    assert days[0] == pytest.approx(0.0)


def test_tax_break_even_grows_with_gain_and_rate():
    rows = st.tax_break_even(26.0)
    by_key = {(r["tax_rate"], r["embedded_gain"]): r["break_even_years"] for r in rows}
    assert by_key[(0.20, 3.00)] > by_key[(0.20, 0.50)]
    assert by_key[(0.238, 0.50)] > by_key[(0.15, 0.50)]
    assert by_key[(0.20, 0.50)] > 10.0  # switching an appreciated holding never pays quickly


def test_long_short_dies_once_borrow_exceeds_the_gross_spread():
    rows = st.long_short_net(26.0, cost_bps=2.0)
    assert rows[0]["alive"]
    assert not rows[-1]["alive"]
    assert all(rows[i]["net_bpy"] > rows[i + 1]["net_bpy"] for i in range(len(rows) - 1))


def test_liquidity_table_scales_with_trade_size(planted):
    prices, truth = planted
    funds = truth["fund_cols"]
    rng = np.random.default_rng(7)
    dv = pd.DataFrame(
        {f: np.exp(rng.normal(np.log(1e8 / (i + 1)), 0.3, len(prices))) for i, f in enumerate(funds)},
        index=prices.index,
    )
    small = st.liquidity_table(dv, funds, trade_usd=10e6)
    big = st.liquidity_table(dv, funds, trade_usd=100e6)
    assert (big["days_of_adv"] > small["days_of_adv"]).all()
    assert big["days_of_adv"].idxmax() == funds[-1]  # the thinnest wrapper


# --------------------------------------------------------------------------- #
# The rules — one execution lag, costs one-way x NAV
# --------------------------------------------------------------------------- #
def _race_frame(prices, truth):
    px = prices[truth["fund_cols"] + ["cash"]].copy()
    return px


def test_ownership_race_arms_and_lag(planted):
    prices, truth = planted
    funds = truth["fund_cols"]
    fees = {f: v for f, v in zip(funds, truth["fees_bps"])}
    race = st.ownership_race(_race_frame(prices, truth), funds, "cash", fees, cost_bps=2.0)
    assert set(race["arms"]) == {"own_cheapest", "own_priciest", "chase_winner"}
    assert race["cheapest"] == funds[int(np.argmin(truth["fees_bps"]))]
    assert race["priciest"] == funds[int(np.argmax(truth["fees_bps"]))]
    # owning the cheapest must beat owning the priciest on a planted ladder
    assert race["sharpe_gap_cheap_minus_dear"] > 0
    assert race["arms"]["own_cheapest"]["cagr"] > race["arms"]["own_priciest"]["cagr"]


def test_chase_rule_costs_reduce_its_return(planted):
    prices, truth = planted
    funds = truth["fund_cols"]
    fees = {f: v for f, v in zip(funds, truth["fees_bps"])}
    px = _race_frame(prices, truth)
    cagrs = []
    for c in (0.0, 5.0, 50.0):
        race = st.ownership_race(px, funds, "cash", fees, cost_bps=c)
        cagrs.append(race["arms"]["chase_winner"]["cagr"])
    assert cagrs[0] >= cagrs[1] >= cagrs[2]


def test_chase_rule_has_a_real_execution_lag_not_a_same_close_fill(planted):
    """Ranked at the last close of year Y, filled at the next close, earning from the one after.

    The failure mode this pins down: assigning the new wrapper's return to the *first*
    session of the new year would mean the position was established at the very close on
    which the signal was formed — a same-day fill dressed up as a lag. So every change of
    holding must sit on the **second** session of a calendar year, never the first.
    """
    prices, truth = planted
    funds = truth["fund_cols"]
    fees = {f: v for f, v in zip(funds, truth["fees_bps"])}
    race = st.ownership_race(_race_frame(prices, truth), funds, "cash", fees)
    h = race["holdings"]
    changes = h.index[(h != h.shift(1)) & h.shift(1).notna()]
    assert len(changes) > 0
    for d in changes:
        i = h.index.get_loc(d)
        assert i >= 2
        prev1, prev2 = h.index[i - 1], h.index[i - 2]
        assert prev1.year == d.year          # the first session of the year is NOT the switch
        assert prev2.year == d.year - 1      # ...it is the session right after the year-end
        assert h.iloc[i - 1] == h.iloc[i - 2]  # that session still earns the OLD wrapper


# --------------------------------------------------------------------------- #
# The study's spine — the machinery is unbiased
# --------------------------------------------------------------------------- #
def test_planted_ladder_is_recovered(planted):
    prices, truth = planted
    d = st.synthetic_detect(prices, truth)
    assert d["pair_spread_bpy"] == pytest.approx(d["planted_gap_bpy"], abs=8.0)
    assert d["pair_t"] > 2.0
    assert d["spearman"] < -0.8
    assert d["p_perm"] <= 0.05
    assert d["pass_through_slope"] == pytest.approx(-1.0, abs=0.25)
    assert d["pass_through_r2"] > 0.9


def test_null_panel_stays_quiet(null_panel):
    prices, truth = null_panel
    d = st.synthetic_detect(prices, truth)
    assert abs(d["pair_spread_bpy"]) < 6.0
    assert abs(d["pair_t"]) < 2.0
    assert d["p_perm"] > 0.05


def test_null_across_seeds_is_centred():
    spreads, ts, ps = [], [], []
    for s in range(8):
        prices, truth = data.synthetic_panel(signal_strength=0.0, seed=961 + s)
        d = st.synthetic_detect(prices, truth)
        spreads.append(d["pair_spread_bpy"]); ts.append(d["pair_t"]); ps.append(d["p_perm"])
    spreads, ts, ps = np.array(spreads), np.array(ts), np.array(ps)
    assert abs(spreads.mean()) < 3.0
    assert (np.abs(ts) >= 2.0).sum() <= 1
    assert (ps < 0.05).sum() <= 1


def test_era_cut_returns_both_halves(planted):
    prices, truth = planted
    funds = truth["fund_cols"]
    cheap = funds[int(np.argmin(truth["fees_bps"]))]
    dear = funds[int(np.argmax(truth["fees_bps"]))]
    eras = st.era_cut(prices[funds], cheap, dear, split="2022-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["spread_bpy"]) and e["spread_bpy"] > 0

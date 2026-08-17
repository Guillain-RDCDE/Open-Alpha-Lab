"""Basket mechanics, beta plumbing, inference primitives and the study's spine.

All offline and synthetic — no cache, no network.

The spine: on a panel with a *planted* laddering premium the race recovers it with
|*t*| >= 2 and lands within a pp of the planted value; on a null panel it reports a gap
indistinguishable from zero across seeds. The basket must equal-weight exactly, costs must
only ever reduce returns, and every estimated weight must be lagged one day.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffer_ladder import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Return plumbing
# --------------------------------------------------------------------------- #
def test_excess_returns_removes_the_cash_leg(planted_ladder):
    prices, _ = planted_ladder
    r = st.to_returns(prices)
    ex = st.excess_returns(r, "cash")
    assert "cash" not in ex.columns
    assert np.allclose((r["ladder"] - r["cash"]).to_numpy(), ex["ladder"].to_numpy())


def test_to_returns_matches_manual_pct_change(planted_ladder):
    prices, _ = planted_ladder
    r = st.to_returns(prices)
    manual = prices["market"].pct_change().dropna()
    assert np.allclose(r["market"].to_numpy(), manual.to_numpy())


# --------------------------------------------------------------------------- #
# The DIY basket
# --------------------------------------------------------------------------- #
def test_basket_equals_the_simple_mean_on_the_reset_day():
    """On a rebalance day the basket earns exactly the equal-weight average return."""
    idx = pd.bdate_range("2021-01-04", periods=300)
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, (300, 4)), index=idx, columns=list("abcd"))
    b = st.equal_weight_basket(r, list("abcd"), rebalance="M", cost_bps=0.0)
    assert b.iloc[0] == pytest.approx(r.iloc[0].mean())


def test_basket_weights_drift_between_rebalances():
    """With no rebalance at all the basket must diverge from the naive daily mean."""
    idx = pd.bdate_range("2021-01-04", periods=500)
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0.0004, 0.012, (500, 4)), index=idx, columns=list("abcd"))
    never = st.equal_weight_basket(r, list("abcd"), rebalance="N", cost_bps=0.0)
    monthly = st.equal_weight_basket(r, list("abcd"), rebalance="M", cost_bps=0.0)
    assert not np.allclose(never.to_numpy(), monthly.to_numpy())
    assert np.allclose(never.iloc[0], monthly.iloc[0])   # identical on day one


def test_basket_costs_monotonically_reduce_return(planted_ladder):
    prices, truth = planted_ladder
    ex = st.excess_returns(st.to_returns(prices), "cash")
    means = [st.equal_weight_basket(ex, truth["vintages"], rebalance="M", cost_bps=c).mean()
             for c in (0.0, 5.0, 25.0)]
    assert means[0] >= means[1] >= means[2]


def test_basket_of_identical_legs_is_that_leg():
    idx = pd.bdate_range("2021-01-04", periods=200)
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.01, 200)
    r = pd.DataFrame({"a": x, "b": x, "c": x}, index=idx)
    b = st.equal_weight_basket(r, ["a", "b", "c"], rebalance="Q", cost_bps=5.0)
    assert np.allclose(b.to_numpy(), x)   # no turnover, so no cost either


# --------------------------------------------------------------------------- #
# Beta plumbing and the single execution lag
# --------------------------------------------------------------------------- #
def test_expanding_beta_recovers_a_known_slope():
    idx = pd.bdate_range("2018-01-02", periods=1500)
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(0, 0.01, 1500), index=idx)
    y = 0.6 * x + pd.Series(rng.normal(0, 0.002, 1500), index=idx)
    b = st.expanding_beta(y, x, min_obs=252).dropna()
    assert abs(b.iloc[-1] - 0.6) < 0.05
    assert abs(st.full_sample_beta(y, x) - 0.6) < 0.05


def test_expanding_beta_is_lagged_one_day():
    """The beta used on day t must not move when day t's returns are perturbed."""
    idx = pd.bdate_range("2018-01-02", periods=1000)
    rng = np.random.default_rng(4)
    x = pd.Series(rng.normal(0, 0.01, 1000), index=idx)
    y = pd.Series(0.5 * x.to_numpy() + rng.normal(0, 0.003, 1000), index=idx)
    b1 = st.expanding_beta(y, x, min_obs=252)
    y2 = y.copy()
    y2.iloc[700:] *= 4.0                       # perturb only from day 700 onward
    b2 = st.expanding_beta(y2, x, min_obs=252)
    # Day 700's beta was estimated through day 699, so it must be unchanged.
    assert b1.iloc[700] == pytest.approx(b2.iloc[700])
    assert b1.iloc[701] != pytest.approx(b2.iloc[701])


def test_expanding_beta_warmup_is_nan():
    idx = pd.bdate_range("2018-01-02", periods=600)
    rng = np.random.default_rng(5)
    x = pd.Series(rng.normal(0, 0.01, 600), index=idx)
    y = pd.Series(rng.normal(0, 0.01, 600), index=idx)
    b = st.expanding_beta(y, x, min_obs=252)
    assert b.iloc[:252].isna().all()
    assert b.iloc[300:].notna().all()


def test_beta_matched_mix_is_beta_times_market():
    idx = pd.bdate_range("2021-01-04", periods=200)
    m = pd.Series(np.linspace(-0.01, 0.01, 200), index=idx)
    mix = st.beta_matched_mix(m, 0.5, cost_bps=0.0, borrow_bps=0.0)
    assert np.allclose(mix.to_numpy(), 0.5 * m.to_numpy())


def test_beta_matched_ladder_lifts_the_beta_to_target():
    idx = pd.bdate_range("2018-01-02", periods=1200)
    rng = np.random.default_rng(6)
    m = pd.Series(rng.normal(0, 0.01, 1200), index=idx)
    basket = pd.Series(0.4 * m.to_numpy() + rng.normal(0, 0.002, 1200), index=idx)
    matched = st.beta_matched_ladder(basket, m, 0.6, 0.4, cost_bps=0.0, borrow_bps=0.0)
    assert abs(st.full_sample_beta(matched, m) - 0.6) < 0.02


def test_borrow_is_charged_when_the_matched_leg_goes_short():
    idx = pd.bdate_range("2021-01-04", periods=200)
    m = pd.Series(np.full(200, 0.001), index=idx)
    free = st.beta_matched_mix(m, -0.2, borrow_bps=0.0)
    paid = st.beta_matched_mix(m, -0.2, borrow_bps=200.0)
    assert paid.mean() < free.mean()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean_and_stays_quiet_on_noise():
    rng = np.random.default_rng(7)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_and_welch_are_finite(planted_ladder):
    prices, truth = planted_ladder
    ex = st.excess_returns(st.to_returns(prices), "cash")
    b = st.equal_weight_basket(ex, truth["vintages"])
    assert np.isfinite(st.one_sample_t(ex["ladder"].to_numpy()))
    assert np.isfinite(st.welch_t(ex["ladder"].to_numpy(), b.to_numpy()))
    assert np.isfinite(st.gap_tstat(ex["ladder"], b))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_summary_reports_negative_drawdown_and_finite_sharpe(planted_ladder):
    prices, _ = planted_ladder
    s = st.summary(st.to_returns(prices)["market"])
    assert s["max_drawdown"] <= 0.0
    assert np.isfinite(s["sharpe"]) and np.isfinite(s["vol_ann"])


def test_max_drawdown_of_a_monotone_rise_is_zero():
    r = pd.Series(np.full(100, 0.001), index=pd.bdate_range("2021-01-04", periods=100))
    assert st.max_drawdown(r) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_gap_ci_brackets_the_point(planted_ladder):
    prices, truth = planted_ladder
    ex = st.excess_returns(st.to_returns(prices), "cash")
    b = st.equal_weight_basket(ex, truth["vintages"])
    ci = st.bootstrap_gap_ci(ex["ladder"], b, n_boot=400, seed=947)
    assert ci["ci_low"] <= ci["gap_ann_pp"] <= ci["ci_high"]


def test_bootstrap_sharpe_gap_ci_brackets_the_point(planted_ladder):
    prices, truth = planted_ladder
    ex = st.excess_returns(st.to_returns(prices), "cash")
    b = st.equal_weight_basket(ex, truth["vintages"])
    ci = st.bootstrap_sharpe_gap_ci(ex["ladder"], b, n_boot=400, seed=947)
    assert ci["ci_low"] <= ci["sharpe_gap"] <= ci["ci_high"]


def test_bootstrap_is_reproducible(planted_ladder):
    prices, truth = planted_ladder
    ex = st.excess_returns(st.to_returns(prices), "cash")
    b = st.equal_weight_basket(ex, truth["vintages"])
    a1 = st.bootstrap_gap_ci(ex["ladder"], b, n_boot=200, seed=1)
    a2 = st.bootstrap_gap_ci(ex["ladder"], b, n_boot=200, seed=1)
    assert a1["ci_low"] == a2["ci_low"] and a1["ci_high"] == a2["ci_high"]


def test_block_sensitivity_covers_the_grid_and_agrees_on_a_planted_premium(planted_ladder):
    """The sweep must visit every (block, seed) cell, and a *real* gap must survive it."""
    prices, truth = planted_ladder
    ex = st.excess_returns(st.to_returns(prices), "cash")
    b = st.equal_weight_basket(ex, truth["vintages"])
    rows = st.bootstrap_block_sensitivity(ex["ladder"], b, blocks=(5, 21, 63),
                                          seeds=(947, 1), n_boot=300)
    assert len(rows) == 6
    assert {r["block"] for r in rows} == {5, 21, 63}
    # The planted premium is large: its CI must exclude zero at EVERY block length —
    # that is what a finding whose significance does not depend on the resampling
    # scheme looks like, and the contrast with the real tape is the point.
    assert all(r["excludes_zero"] for r in rows)


def test_block_sensitivity_flags_a_gap_that_only_one_block_length_calls_significant():
    """A near-zero gap must NOT come out 'excludes zero' at every block length."""
    idx = pd.bdate_range("2021-01-04", periods=1200)
    rng = np.random.default_rng(947)
    a = pd.Series(rng.normal(0.00002, 0.004, len(idx)), index=idx)
    b = pd.Series(np.zeros(len(idx)), index=idx)
    rows = st.bootstrap_block_sensitivity(a, b, blocks=(5, 21, 63), seeds=(947,),
                                          n_boot=300)
    assert not all(r["excludes_zero"] for r in rows)


# --------------------------------------------------------------------------- #
# Entry-point luck
# --------------------------------------------------------------------------- #
def test_dispersion_reports_spread_and_variance_reduction(planted_ladder):
    prices, truth = planted_ladder
    d = st.dispersion_stats(st.to_returns(prices), truth["vintages"])
    assert d["spread_max_pp"] >= d["spread_mean_pp"] > 0
    assert d["sd_basket_pct"] <= d["sd_single_mean_pct"]      # averaging cannot add variance
    assert 0.0 <= d["mean_pairwise_corr"] <= 1.0


def test_independent_legs_diversify_far_more_than_correlated_ones():
    """A sanity anchor for the real-tape finding: variance reduction tracks correlation."""
    corr_prices, corr_truth = data.synthetic_panel(entry_luck_vol=0.01, seed=947)
    indep_prices, indep_truth = data.synthetic_panel(entry_luck_vol=0.30, seed=947)
    d_corr = st.dispersion_stats(st.to_returns(corr_prices), corr_truth["vintages"])
    d_indep = st.dispersion_stats(st.to_returns(indep_prices), indep_truth["vintages"])
    assert d_indep["mean_pairwise_corr"] < d_corr["mean_pairwise_corr"]
    assert d_indep["variance_reduction_pct"] > d_corr["variance_reduction_pct"]


# --------------------------------------------------------------------------- #
# The study's spine — the detector recovers a planted premium and is quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_premium_is_recovered(planted_ladder):
    prices, truth = planted_ladder
    d = st.synthetic_detect(prices, truth)
    assert d["t_hac"] >= 2.0
    assert abs(d["error_pp"]) < 1.0          # within a pp of the planted premium
    assert d["gap_ann_pp"] > 0


def test_null_has_no_spurious_premium(clean_null):
    prices, truth = clean_null
    d = st.synthetic_detect(prices, truth)
    assert abs(d["t_hac"]) < 2.0
    assert abs(d["gap_ann_pp"]) < 1.5


def test_null_is_centred_across_seeds():
    gaps, ts = [], []
    for s in range(8):
        prices, truth = data.synthetic_panel(signal_strength=0.0, extra_fee_ann=0.0,
                                             seed=947 + s)
        d = st.synthetic_detect(prices, truth)
        gaps.append(d["gap_ann_pp"])
        ts.append(d["t_hac"])
    assert abs(np.mean(gaps)) < 1.0
    assert sum(abs(t) >= 2.0 for t in ts) <= 1      # at most one false fire in eight


def test_fee_only_null_recovers_the_fee(null_ladder):
    """With no premium but a planted fee, the measured gap must sit near minus the fee."""
    prices, truth = null_ladder
    d = st.synthetic_detect(prices, truth)
    assert d["gap_ann_pp"] < 1.0
    assert abs(d["error_pp"]) < 1.0


def test_detector_is_monotone_in_the_planted_premium():
    gaps = []
    for ss in (0.0, 0.5, 1.0):
        prices, truth = data.synthetic_panel(signal_strength=ss, seed=947)
        gaps.append(st.synthetic_detect(prices, truth)["gap_ann_pp"])
    assert gaps[0] < gaps[1] < gaps[2]


# --------------------------------------------------------------------------- #
# Race / robustness harness
# --------------------------------------------------------------------------- #
def test_race_reports_both_windows_and_all_arms(planted_ladder):
    prices, truth = planted_ladder
    res = st.race(prices, "ladder", truth["vintages"], "market", "cash")
    assert res["n_days"] > res["n_days_matched"] > 0
    for arm in ("ladder", "diy_basket", "diy_beta_matched", "beta_mix_ladder", "market"):
        assert arm in res["arms"] and np.isfinite(res["summary"][arm]["sharpe"])
    for key in ("vs_diy_basket", "vs_diy_beta_matched", "vs_beta_mix"):
        assert np.isfinite(res["gaps"][key]["gap_ann_pp"])
    assert res["gaps"]["vs_diy_basket"]["n_days"] == res["n_days"]
    assert res["gaps"]["vs_diy_beta_matched"]["n_days"] == res["n_days_matched"]


def test_race_beta_matched_arm_carries_the_wrapper_beta(planted_ladder):
    prices, truth = planted_ladder
    res = st.race(prices, "ladder", truth["vintages"], "market", "cash", cost_bps=0.0)
    m = res["arms"]["market"].reindex(res["arms"]["diy_beta_matched"].index)
    b_matched = st.full_sample_beta(res["arms"]["diy_beta_matched"], m)
    b_ladder = st.full_sample_beta(res["arms"]["ladder"].reindex(m.index), m)
    assert abs(b_matched - b_ladder) < 0.05


def test_era_cut_returns_both_halves(planted_ladder):
    prices, truth = planted_ladder
    eras = st.era_cut(prices, "ladder", truth["vintages"], "market", "cash",
                      split="2021-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["vs_diy_basket_gap_pp"])


def test_cost_sweep_cannot_hurt_the_wrapper(planted_ladder):
    """Costs fall on the DIY arms only, so a higher cost can only widen the wrapper's gap."""
    prices, truth = planted_ladder
    rows = st.cost_sweep(prices, "ladder", truth["vintages"], "market", "cash")
    gaps = [r["gap_vs_basket_pp"] for r in rows]
    assert gaps == sorted(gaps)


def test_fee_sweep_is_monotone_and_additive(planted_ladder):
    """Waiving f pp/yr of the assumed fee must lift the gap by exactly f pp/yr."""
    prices, truth = planted_ladder
    rows = st.fee_sweep(prices, "ladder", truth["vintages"], "market", "cash",
                        fee_grid=(0.0, 0.25))
    delta = rows[1]["gap_vs_basket_pp"] - rows[0]["gap_vs_basket_pp"]
    assert delta == pytest.approx(0.25, abs=0.02)


def test_rebalance_sweep_covers_the_grid(planted_ladder):
    prices, truth = planted_ladder
    rows = st.rebalance_sweep(prices, "ladder", truth["vintages"], "market", "cash")
    assert [r["rebalance"] for r in rows] == ["N", "A", "Q", "M"]
    assert all(np.isfinite(r["gap_vs_basket_pp"]) for r in rows)


def test_calendar_year_table_drops_partial_years(planted_ladder):
    prices, truth = planted_ladder
    tbl = st.calendar_year_table(prices, "ladder", truth["vintages"], "market", "cash")
    assert {"ladder", "diy_basket", "vintage_spread_pp", "market"}.issubset(tbl.columns)
    assert (tbl["vintage_spread_pp"] >= 0).all()
    assert len(tbl) >= 3

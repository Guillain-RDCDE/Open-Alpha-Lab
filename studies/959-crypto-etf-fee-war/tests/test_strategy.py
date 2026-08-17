"""Estimators, inference and rules — all offline/synthetic.

The spine of the study, stated as tests: on a planted fee ladder the fund-versus-fund
monthly spread recovers the planted gap with a large HAC *t* and the cross-sectional
pass-through slope lands near −1; on the null (one fee for everybody, a fee sheet that
still looks dispersed) both stay quiet. Around that: the estimators' invariants, the
execution lag, the cost/borrow/tax arithmetic, and the measurement floor.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fee_war import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_ols_t_at_zero_lags():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 500)
    plain = st.one_sample_t(x)
    assert st.newey_west_t(x, lags=0) == pytest.approx(plain, rel=1e-2)


def test_newey_west_widens_on_positively_autocorrelated_data():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 4000)
    x = np.empty(4000)
    x[0] = e[0]
    for i in range(1, 4000):
        x[i] = 0.8 * x[i - 1] + e[i]
    x = x + 0.5
    assert abs(st.newey_west_t(x, lags=20)) < abs(st.one_sample_t(x))


def test_newey_west_is_nan_on_a_degenerate_series():
    assert not np.isfinite(st.newey_west_t([1.0]))
    assert not np.isfinite(st.newey_west_t(np.zeros(50)))


def test_spearman_endpoints_and_ties():
    assert st.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert st.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    r = st.rankdata([5.0, 5.0, 1.0])
    assert r[0] == r[1] == pytest.approx(1.5)


def test_block_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(10e-4, 5e-4, 60)          # monthly series, ~120 bp/yr
    ci = st.block_bootstrap_ci(x, n_boot=1500, block=3, seed=959)
    assert ci["ci_low"] < ci["point"] < ci["ci_high"]
    assert ci["frac_negative"] < 0.05


def test_block_bootstrap_ci_is_nan_when_too_short():
    ci = st.block_bootstrap_ci([1.0, 2.0], block=3)
    assert not np.isfinite(ci["point"])


# --------------------------------------------------------------------------- #
# The three tracking-difference estimators
# --------------------------------------------------------------------------- #
def _clean_pair(fee_bpy: float, n: int = 600):
    """A noiseless fund shaved by ``fee_bpy`` against a noiseless benchmark."""
    idx = pd.bdate_range("2024-01-11", periods=n)
    years = np.asarray((idx - idx[0]).days, dtype=float) / 365.25
    bench = pd.Series(np.exp(0.3 * years), index=idx)
    fund = pd.Series(np.exp(0.3 * years - fee_bpy * 1e-4 * years), index=idx)
    return fund, bench


def test_td_endpoint_recovers_a_clean_fee():
    fund, bench = _clean_pair(150.0)
    assert st.td_endpoint(fund, bench) == pytest.approx(-150.0, abs=0.5)


def test_td_slope_recovers_a_clean_fee():
    fund, bench = _clean_pair(150.0)
    out = st.td_trend_slope(fund, bench)
    assert out["slope_bpy"] == pytest.approx(-150.0, abs=0.5)


def test_td_monthly_recovers_a_clean_fee():
    fund, bench = _clean_pair(150.0)
    mm = st.monthly_log_returns(pd.DataFrame({"f": fund, "b": bench}))
    out = st.td_monthly(mm["f"], mm["b"])
    assert out["td_bpy"] == pytest.approx(-150.0, abs=3.0)
    assert out["n_months"] >= 20 and out["pos_months"] == 0


def test_td_monthly_is_non_overlapping_and_drops_the_partial_month():
    idx = pd.bdate_range("2024-01-11", periods=300)
    px = pd.DataFrame({"a": np.linspace(100, 200, 300)}, index=idx)
    mm = st.monthly_log_returns(px)
    # one observation per complete calendar month, first one lost to the diff
    assert len(mm) == len(px.resample("ME").last()) - 1
    assert mm.index.is_monotonic_increasing and not mm.index.duplicated().any()


def test_td_monthly_is_nan_on_a_stub():
    out = st.td_monthly(pd.Series([0.01, 0.02]), pd.Series([0.00, 0.01]))
    assert not np.isfinite(out["td_bpy"])


def test_td_monthly_is_algebraically_an_endpoint_estimate():
    """The honest caveat, pinned: log increments telescope.

    ``td_monthly`` is NOT anchor-free — its point estimate equals the endpoint estimate
    taken between the first and last complete month-end. What it adds over ``td_endpoint``
    is a better pair of anchors (the contaminated first session is gone) and a dispersion.
    If anyone ever "improves" this estimator into something that hides that, this fails.
    """
    fund, bench = _clean_pair(150.0)
    px = pd.DataFrame({"f": fund, "b": bench})
    mm = st.monthly_log_returns(px)
    x = (mm["f"] - mm["b"]).dropna()
    me = np.log(px).resample("ME").last()
    ratio = (me["f"] - me["b"]).dropna()
    telescoped = (ratio.iloc[-1] - ratio.iloc[0]) / len(x)
    assert x.mean() == pytest.approx(telescoped, rel=1e-10, abs=1e-14)


def test_td_monthly_survives_trimming_its_own_anchors():
    """A real fee is in every month, so trimming months off both ends barely moves it."""
    fund, bench = _clean_pair(150.0)
    mm = st.monthly_log_returns(pd.DataFrame({"f": fund, "b": bench}))
    x = (mm["f"] - mm["b"]).dropna()
    full = x.mean() * st.MONTHS * 1e4
    for k in (1, 2, 3):
        trimmed = x.iloc[k:len(x) - k].mean() * st.MONTHS * 1e4
        assert trimmed == pytest.approx(full, abs=10.0)


def test_estimators_agree_on_a_clean_tape_and_disagree_under_anchor_noise():
    """A single dislocated first anchor moves the endpoint estimator, not the slope."""
    fund, bench = _clean_pair(150.0)
    fund2 = fund.copy()
    fund2.iloc[0] *= 1.025                      # a 250 bp conversion-day premium
    assert st.td_endpoint(fund2, bench) < st.td_endpoint(fund, bench) - 50.0
    assert st.td_trend_slope(fund2, bench)["slope_bpy"] == pytest.approx(-150.0, abs=15.0)


def test_tracking_table_shape_and_ordering(fee_ladder):
    prices, truth = fee_ladder
    funds = truth["fund_cols"]
    fees = dict(zip(funds, truth["fees_bps"]))
    tbl = st.tracking_table(prices, funds, "bench", fees=fees)
    assert list(tbl.index) == funds
    assert {"td_endpoint_bpy", "td_slope_bpy", "td_monthly_bpy", "td_rel_bpy"}.issubset(tbl.columns)
    # the 150 bp wrapper must be the worst cohort-relative tracker of the panel
    assert tbl["td_rel_bpy"].idxmin() == funds[int(np.argmax(truth["fees_bps"]))]


# --------------------------------------------------------------------------- #
# The measurement floor
# --------------------------------------------------------------------------- #
def test_measurement_floor_ranks_bench_noise_above_peer_noise(fee_ladder):
    prices, truth = fee_ladder
    fl = st.measurement_floor(prices, truth["fund_cols"], "bench")
    assert fl["sd_daily_vs_bench_bp"] > fl["sd_daily_vs_peer_bp"]
    assert fl["ratio"] > 5.0
    assert fl["detectable_vs_bench_bpy"] > fl["detectable_vs_peer_bpy"]


# --------------------------------------------------------------------------- #
# The spine — recover the planted effect, stay quiet on the null
# --------------------------------------------------------------------------- #
def test_pair_spread_recovers_the_planted_gap(fee_ladder):
    prices, truth = fee_ladder
    funds = truth["fund_cols"]
    fees = np.asarray(truth["fees_eff_bps"])
    cheap, dear = funds[int(fees.argmin())], funds[int(fees.argmax())]
    hd = st.pair_spread(prices, cheap, dear)
    planted = float(fees.max() - fees.min())
    assert hd["tstat"] > 4.0
    assert abs(hd["spread_bpy"] - planted) < 0.35 * planted
    assert hd["ci_low"] > 0.0
    assert hd["pos_months"] > 0.7 * hd["n_months"]


def test_pair_spread_is_quiet_on_the_null(flat_fees):
    prices, truth = flat_fees
    funds = truth["fund_cols"]
    hd = st.pair_spread(prices, funds[0], funds[-1])
    assert abs(hd["tstat"]) < 2.0
    assert hd["ci_low"] < 0.0 < hd["ci_high"]


def test_pair_spread_is_antisymmetric(fee_ladder):
    prices, truth = fee_ladder
    funds = truth["fund_cols"]
    a = st.pair_spread(prices, funds[0], funds[-1])
    b = st.pair_spread(prices, funds[-1], funds[0])
    assert a["spread_bpy"] == pytest.approx(-b["spread_bpy"], rel=1e-9)


def test_pass_through_slope_is_near_minus_one_on_the_ladder(fee_ladder):
    prices, truth = fee_ladder
    d = st.synthetic_detect(prices, truth, n_perm=2000)
    assert d["pass_through_slope"] == pytest.approx(-1.0, abs=0.25)
    assert d["pass_through_r2"] > 0.8


def test_pass_through_slope_is_near_zero_on_the_null(flat_fees):
    prices, truth = flat_fees
    d = st.synthetic_detect(prices, truth, n_perm=2000)
    assert abs(d["pass_through_slope"]) < 0.25
    assert abs(d["pair_t"]) < 2.0


def test_null_stays_quiet_across_seeds():
    """Multi-seed null: the pair instrument must never fire, and the rank test rarely."""
    ts, ps = [], []
    for s in range(6):
        prices, truth = data.synthetic_panel(signal_strength=0.0, seed=959 + s)
        d = st.synthetic_detect(prices, truth, n_perm=1500)
        ts.append(abs(d["pair_t"])); ps.append(d["p_perm"])
    assert max(ts) < 2.0
    assert sum(1 for p in ps if p < 0.05) <= 2


# --------------------------------------------------------------------------- #
# The fee-rank test (uses the fee ASSUMPTION)
# --------------------------------------------------------------------------- #
def test_rank_test_fires_on_a_perfect_inverse_ranking():
    fees = [19.0, 20.0, 21.0, 25.0, 30.0, 150.0]
    tds = [-19.0, -20.0, -21.0, -25.0, -30.0, -150.0]
    rt = st.rank_test(fees, tds)
    assert rt["spearman"] == pytest.approx(-1.0)     # dearer fund, worse tracking
    assert rt["exact"] is True and rt["p_perm"] < 0.05


def test_rank_test_is_quiet_on_a_shuffled_ranking():
    rng = np.random.default_rng(3)
    fees = np.arange(1.0, 11.0)
    fired = 0
    for _ in range(30):
        rt = st.rank_test(fees, rng.normal(size=10), n_perm=1500)
        fired += int(rt["p_perm"] < 0.05)
    assert fired <= 5           # nominal 5% of 30 draws, with binomial slack


def test_rank_test_reports_the_attainable_critical_value():
    """A fee sheet with heavy ties cannot generate a fine-grained rank statistic."""
    tied = [25.0] * 8 + [150.0, 150.0]
    rt = st.rank_test(tied, list(np.arange(10.0)), n_perm=2000)
    assert rt["crit_5pct"] > 0.0
    assert abs(rt["spearman"]) <= 1.0


def test_pass_through_is_exact_on_a_noiseless_cross_section():
    fees = np.array([10.0, 20.0, 30.0, 40.0])
    pt = st.pass_through(fees, -fees + 5.0)
    assert pt["slope"] == pytest.approx(-1.0)
    assert pt["intercept"] == pytest.approx(5.0)
    assert pt["r2"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Waiver events & era cut
# --------------------------------------------------------------------------- #
def test_waiver_event_test_finds_a_planted_step():
    """Plant a fee that switches on mid-sample and check the step is recovered."""
    prices, truth = data.synthetic_panel(signal_strength=1.0, seed=959)
    funds = truth["fund_cols"]
    idx = prices.index
    cut = idx[len(idx) // 2]
    extra_bpy = 100.0
    years = np.asarray((idx - idx[0]).days, dtype=float) / 365.25
    years_post = np.clip(years - float((cut - idx[0]).days) / 365.25, 0.0, None)
    px = prices.copy()
    px[funds[0]] = px[funds[0]] * np.exp(-extra_bpy * 1e-4 * years_post)
    ev = st.waiver_event_test(px, funds, {funds[0]: (0.0, str(cut.date()))})
    step = float(ev.loc[funds[0], "step_bpy"])
    assert step < -50.0                      # the fund gets worse after the waiver ends
    assert ev.loc[funds[0], "welch_t"] < 0.0


def test_waiver_event_test_reports_nan_when_a_side_is_too_short(fee_ladder):
    prices, truth = fee_ladder
    funds = truth["fund_cols"]
    ev = st.waiver_event_test(prices, funds, {funds[0]: (0.0, "2024-01-20")})
    assert not np.isfinite(ev.loc[funds[0], "step_bpy"])


def test_era_cut_halves_sum_back_to_the_full_sample(fee_ladder):
    prices, truth = fee_ladder
    funds = truth["fund_cols"]
    cheap, dear = funds[0], funds[-1]
    full = st.pair_spread(prices, cheap, dear)
    split = str(prices.index[len(prices) // 2].date())
    eras = st.era_cut(prices, cheap, dear, split=split)
    n = eras["early"]["n_months"] + eras["late"]["n_months"]
    assert n == full["n_months"]
    blend = (eras["early"]["spread_bpy"] * eras["early"]["n_months"]
             + eras["late"]["spread_bpy"] * eras["late"]["n_months"]) / n
    assert blend == pytest.approx(full["spread_bpy"], rel=1e-6)


# --------------------------------------------------------------------------- #
# The rules — lag, costs, borrow, tax
# --------------------------------------------------------------------------- #
def test_rotation_race_arms_and_lag(fee_ladder):
    prices, truth = fee_ladder
    funds = truth["fund_cols"]
    fees = dict(zip(funds, truth["fees_bps"]))
    race = st.rotation_race(prices, funds, "cash", fees)
    assert race["cheapest"] == funds[int(np.argmin(truth["fees_bps"]))]
    assert race["priciest"] == funds[int(np.argmax(truth["fees_bps"]))]
    assert set(race["arms"]) == {"own_cheapest", "own_priciest", "rotate_winner"}
    for a in race["arms"].values():
        assert np.isfinite(a["excess_sharpe"]) and np.isfinite(a["cagr"])
    # holding the cheapest wrapper must beat holding the priciest on total return
    assert race["arms"]["own_cheapest"]["total_return"] > race["arms"]["own_priciest"]["total_return"]


def test_rotation_holdings_change_only_after_a_quarter_end(fee_ladder):
    prices, truth = fee_ladder
    funds = truth["fund_cols"]
    fees = dict(zip(funds, truth["fees_bps"]))
    race = st.rotation_race(prices, funds, "cash", fees)
    h = race["holdings"]
    changes = h.index[(h != h.shift(1)) & h.shift(1).notna()]
    q_end_next = set(pd.Series(h.index, index=h.index).resample("QE").last().dropna().to_numpy())
    for d in changes:
        prior = h.index[h.index < d]
        assert len(prior) > 0 and prior[-1] in q_end_next   # switched the session AFTER a quarter end


def test_rotation_cost_only_hurts(fee_ladder):
    prices, truth = fee_ladder
    funds = truth["fund_cols"]
    fees = dict(zip(funds, truth["fees_bps"]))
    free = st.rotation_race(prices, funds, "cash", fees, cost_bps=0.0)
    dear = st.rotation_race(prices, funds, "cash", fees, cost_bps=25.0)
    assert free["n_switches"] == dear["n_switches"]
    if free["n_switches"] > 0:
        assert dear["arms"]["rotate_winner"]["total_return"] < free["arms"]["rotate_winner"]["total_return"]
    # the buy-and-hold arms carry no turnover, so cost cannot touch them
    assert dear["arms"]["own_cheapest"]["total_return"] == pytest.approx(
        free["arms"]["own_cheapest"]["total_return"])


def test_long_short_dies_once_borrow_exceeds_the_spread():
    rows = st.long_short_net(140.0)
    assert rows[0]["alive"] is True
    assert rows[-1]["alive"] is False
    nets = [r["net_bpy"] for r in rows]
    assert nets == sorted(nets, reverse=True)          # monotone decreasing in borrow


def test_switch_break_even_scales_with_cost():
    rows = st.switch_break_even(140.0)
    assert rows[0]["break_even_days"] == pytest.approx(0.0)
    assert rows[-1]["break_even_days"] > rows[1]["break_even_days"]
    # a zero or negative spread is never repaid
    assert not np.isfinite(st.switch_break_even(0.0)[1]["break_even_days"])


def test_tax_break_even_grows_with_rate_and_gain():
    rows = {(r["tax_rate"], r["embedded_gain"]): r["break_even_years"] for r in st.tax_break_even(140.0)}
    assert rows[(0.238, 3.00)] > rows[(0.238, 0.20)]
    assert rows[(0.238, 0.50)] > rows[(0.15, 0.50)]
    assert all(np.isfinite(v) for v in rows.values())

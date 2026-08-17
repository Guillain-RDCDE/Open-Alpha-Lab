"""Replicate mechanics, inference primitives, and the study's spine — all offline.

The spine: on a tape where an inverse fund really does leak a planted 3%/yr the harness
recovers a negative gap of about that size; on a costless replicate it measures zero. The
accounting decomposition recovers a planted dividend advantage and financing passthrough
it was never told. The direct-short replicate responds monotonically and with the right
sign to each of its two PROXY inputs (the rebate credit and the stock-loan fee).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inverse_tax import data, strategy as st  # noqa: E402


def frame(prices):
    """The synthetic tape in the column names ``prepare`` expects."""
    return st.synthetic_frame(prices)


def legs_of(prices):
    return st.prepare(frame(prices), fund="FUND", index="IDX")


# --------------------------------------------------------------------------- #
# The direct-short replicate
# --------------------------------------------------------------------------- #
def test_prepare_returns_the_five_legs_and_drops_the_first_bar(fair):
    prices, _ = fair
    legs = legs_of(prices)
    assert {"r_fund", "r_index_tr", "r_index_px", "rf", "r_cash", "days"}.issubset(legs.columns)
    assert not legs.isna().any().any()
    assert len(legs) == len(prices) - 1  # the first bar has no return and no lagged rate


def test_direct_short_is_minus_one_times_the_index_before_financing(fair):
    prices, _ = fair
    legs = legs_of(prices)
    r = st.direct_short(legs, k=-1.0, credit=0.0, borrow_bps=0.0, cost_bps=0.0)
    assert np.allclose(r.to_numpy(), -legs["r_index_tr"].to_numpy())


def test_direct_short_uses_total_return_not_price_return(rich):
    """A share-short owes the dividends, so the replicate must ride the TOTAL-return leg."""
    prices, _ = rich
    legs = legs_of(prices)
    r = st.direct_short(legs, k=-1.0, credit=0.0, borrow_bps=0.0, cost_bps=0.0)
    assert np.allclose(r.to_numpy(), -legs["r_index_tr"].to_numpy())
    assert not np.allclose(r.to_numpy(), -legs["r_index_px"].to_numpy())


def test_leverage_scales_the_replicate(fair):
    prices, _ = fair
    legs = legs_of(prices)
    r1 = st.direct_short(legs, k=-1.0, credit=0.0, borrow_bps=0.0, cost_bps=0.0)
    r2 = st.direct_short(legs, k=-2.0, credit=0.0, borrow_bps=0.0, cost_bps=0.0)
    assert np.allclose(r2.to_numpy(), 2.0 * r1.to_numpy())


def test_more_rebate_credit_helps_the_direct_book_monotonically(fair):
    prices, _ = fair
    legs = legs_of(prices)
    means = [st.direct_short(legs, -1.0, c, 0.0, 0.0).mean() for c in (0.0, 1.0, 2.0)]
    assert means[0] < means[1] < means[2]


def test_borrow_and_cost_only_ever_hurt_the_direct_book(fair):
    prices, _ = fair
    legs = legs_of(prices)
    base = st.direct_short(legs, -1.0, 1.0, 0.0, 0.0).mean()
    assert st.direct_short(legs, -1.0, 1.0, 30.0, 0.0).mean() < base
    assert st.direct_short(legs, -1.0, 1.0, 0.0, 5.0).mean() < base
    # ... and monotonically so.
    fees = [st.direct_short(legs, -1.0, 1.0, b, 0.0).mean() for b in (0.0, 30.0, 100.0)]
    assert fees[0] > fees[1] > fees[2]


def test_rebalance_turnover_is_the_true_releveraging_trade(fair):
    """A constant −k× book must trade |k(k-1)|x|r| of NAV, i.e. 2x|r| at −1x, 6x|r| at −2x."""
    prices, _ = fair
    legs = legs_of(prices)
    for k, factor in [(-1.0, 2.0), (-2.0, 6.0)]:
        gross = st.direct_short(legs, k, 0.0, 0.0, 0.0)
        charged = st.direct_short(legs, k, 0.0, 0.0, 10.0)
        implied = (gross - charged) / (10.0 * 1e-4)
        assert np.allclose(implied.to_numpy(),
                           factor * legs["r_index_tr"].abs().to_numpy())


def test_same_mandate_gap_debits_the_fund_the_dividends(rich):
    """The like-for-like race charges the fund |k| x the realised distribution yield."""
    prices, truth = rich
    legs = legs_of(prices)
    head = st.gap_series(legs, -1.0, 1.0, 30.0, 1.0)
    same = st.same_mandate_gap(legs, -1.0, 1.0, 30.0, 1.0)
    delta = st.annualised_pct(head) - st.annualised_pct(same)
    assert delta == pytest.approx(truth["div_yield_ann"] * 100.0, abs=0.2)
    # ...and it scales with |k|.
    same2 = st.same_mandate_gap(legs, -2.0, 1.0, 30.0, 1.0)
    d2 = st.annualised_pct(st.gap_series(legs, -2.0, 1.0, 30.0, 1.0)) - st.annualised_pct(same2)
    assert d2 == pytest.approx(2.0 * delta, rel=1e-6)


def test_same_mandate_gap_equals_the_headline_when_the_index_pays_nothing(fair):
    """With a zero dividend yield the two races are the same race."""
    prices, _ = fair
    legs = legs_of(prices)
    assert np.allclose(st.same_mandate_gap(legs).to_numpy(), st.gap_series(legs).to_numpy())


def test_hac_lag_profile_starts_at_the_iid_t(taxed):
    prices, _ = taxed
    gap = st.gap_series(legs_of(prices))
    prof = st.hac_lag_profile(gap, lag_grid=(0, 5, 21))
    assert [r["lags"] for r in prof] == [0, 5, 21, None]
    # lags=0 is the i.i.d. t up to the ddof convention (HAC divides by n, not n-1).
    assert prof[0]["t"] == pytest.approx(st.one_sample_t(gap.to_numpy()), rel=1e-3)
    assert all(np.isfinite(r["t"]) for r in prof)


def test_gap_is_linear_in_the_rebate_credit(fair):
    """The break-even solver assumes linearity in c; check it holds exactly."""
    prices, _ = fair
    legs = legs_of(prices)
    g = [st.annualised_pct(st.gap_series(legs, -1.0, c, 30.0, 1.0)) for c in (0.0, 1.0, 2.0)]
    assert g[1] - g[0] == pytest.approx(g[2] - g[1], rel=1e-9)


def test_no_lookahead_the_rate_leg_uses_the_previous_close(fair):
    """Perturbing the rate level on the last bars cannot change any earlier financing."""
    prices, _ = fair
    legs_a = legs_of(prices)
    bumped = prices.copy()
    bumped.iloc[-50:, bumped.columns.get_loc("rate_pct")] *= 3.0
    legs_b = legs_of(bumped)
    assert np.allclose(legs_a["rf"].to_numpy()[:-50], legs_b["rf"].to_numpy()[:-50])


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean_and_stays_quiet_on_noise():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_welch_and_one_sample_finite(fair):
    prices, _ = fair
    legs = legs_of(prices)
    gap = st.gap_series(legs)
    assert np.isfinite(st.one_sample_t(gap.to_numpy()))
    assert np.isfinite(st.welch_t(legs["r_fund"].to_numpy(), legs["r_index_tr"].to_numpy()))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_annualised_pct_matches_hand_arithmetic():
    s = pd.Series([1e-4] * 252)
    assert st.annualised_pct(s) == pytest.approx(2.52)


def test_summary_reports_negative_drawdown_and_finite_sharpe(fair):
    prices, _ = fair
    legs = legs_of(prices)
    s = st.summary(legs["r_fund"])
    assert s["max_drawdown"] <= 0.0
    assert np.isfinite(s["sharpe"]) and np.isfinite(s["vol_ann"])


def test_bootstrap_ci_brackets_the_point_estimate(taxed):
    prices, _ = taxed
    gap = st.gap_series(legs_of(prices))
    ci = st.bootstrap_gap_ci(gap, n_boot=400, seed=942)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["point"] < 0  # the planted tax is a real loss


# --------------------------------------------------------------------------- #
# The study's spine — planted effect recovered, null quiet
# --------------------------------------------------------------------------- #
def test_planted_tax_is_recovered_with_the_right_sign_and_size(taxed):
    prices, truth = taxed
    d = st.synthetic_detect(prices)
    assert d["gap_ann_pct"] < 0
    assert abs(d["gap_ann_pct"] - truth["expected_gap_ann"] * 100.0) < 0.5
    assert d["gap_t"] < -2


def test_null_measures_no_structural_gap(fair):
    prices, truth = fair
    d = st.synthetic_detect(prices)
    assert truth["expected_gap_ann"] == pytest.approx(0.0)
    assert abs(d["gap_ann_pct"]) < 0.5


def test_null_panel_is_centred_on_zero():
    seeds = tuple(942 + s for s in range(12))
    gaps = np.array([st.synthetic_detect(p)["gap_ann_pct"]
                     for p, _ in data.synthetic_panel(seeds=seeds, signal_strength=0.0)])
    ts = np.array([st.synthetic_detect(p)["gap_t"]
                   for p, _ in data.synthetic_panel(seeds=seeds, signal_strength=0.0)])
    assert abs(gaps.mean()) < 0.20
    assert (np.abs(ts) >= 2.0).sum() <= 2  # a well-behaved 5%-ish false-positive rate


def test_planted_panel_recovers_the_sign_on_every_seed():
    seeds = tuple(942 + s for s in range(12))
    gaps = np.array([st.synthetic_detect(p)["gap_ann_pct"]
                     for p, _ in data.synthetic_panel(seeds=seeds, signal_strength=1.0)])
    assert (gaps < 0).all()
    assert abs(gaps.mean() + 3.0) < 0.3


def test_signal_strength_monotonically_worsens_the_gap():
    gaps = [st.synthetic_detect(data.synthetic_daily(signal_strength=s, seed=942)[0])["gap_ann_pct"]
            for s in (0.0, 0.5, 1.0)]
    assert gaps[0] > gaps[1] > gaps[2]


# --------------------------------------------------------------------------- #
# The accounting decomposition
# --------------------------------------------------------------------------- #
def test_decomposition_adds_back_to_the_raw_gap(rich):
    prices, _ = rich
    d = st.decompose(frame(prices), fund="FUND", index="IDX", k=-1.0, expense_ratio_pct=1.0)
    rebuilt = (d["dividends_not_owed_pct"] + d["financing_passthrough_pct"]
               - d["expense_ratio_pct"] + d["residual_pct"])
    assert rebuilt == pytest.approx(d["raw_gap_ann_pct"], abs=1e-9)


def test_decomposition_recovers_planted_dividend_and_passthrough(rich):
    prices, truth = rich
    d = st.decompose(frame(prices), fund="FUND", index="IDX", k=-1.0, expense_ratio_pct=0.0)
    assert d["div_yield_ann_pct"] == pytest.approx(truth["div_yield_ann"] * 100.0, abs=0.2)
    assert d["gamma"] == pytest.approx(truth["passthrough"], abs=0.25)
    assert d["beta"] == pytest.approx(1.0, abs=0.02)
    # The price-leg spec is reported alongside: the funds track the PRICE index, and the
    # financing/residual split is spec-sensitive even though the raw gap is not.
    assert np.isfinite(d["gamma_px"])
    assert d["financing_passthrough_px_pct"] == pytest.approx(
        d["gamma_px"] * d["rf_ann_pct"], rel=1e-9)


def test_breakeven_credit_zeroes_the_gap(taxed):
    prices, _ = taxed
    fr = frame(prices)
    c = st.breakeven_credit(fr, fund="FUND", index="IDX", k=-1.0, borrow_bps=0.0, cost_bps=0.0)
    legs = st.prepare(fr, fund="FUND", index="IDX")
    at_star = st.annualised_pct(st.gap_series(legs, -1.0, c, 0.0, 0.0))
    assert abs(at_star) < 1e-6


# --------------------------------------------------------------------------- #
# Cuts, sweeps and the path-drag table
# --------------------------------------------------------------------------- #
def test_era_cut_returns_both_halves(taxed):
    prices, _ = taxed
    eras = st.era_cut(frame(prices), fund="FUND", index="IDX", k=-1.0, split="2016-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["gap_ann_pct"]) and np.isfinite(e["gap_t"])


def test_rate_regime_cut_puts_a_flat_rate_tape_in_one_bucket(taxed):
    prices, _ = taxed  # rate_ann defaults to 3%, so everything is "normal"
    rr = st.rate_regime_cut(frame(prices), fund="FUND", index="IDX", k=-1.0, threshold_pct=1.0)
    assert rr["zirp"] is None
    assert rr["normal"] is not None and rr["normal"]["n_days"] > 1000


def test_sweeps_have_the_expected_shape_and_monotonicity(fair):
    prices, _ = fair
    fr = frame(prices)
    rows = st.credit_borrow_sweep(fr, fund="FUND", index="IDX", k=-1.0,
                                  credits=(0.0, 1.0), borrows=(0.0, 100.0))
    assert len(rows) == 4
    by = {(r["credit"], r["borrow_bps"]): r["gap_ann_pct"] for r in rows}
    assert by[(0.0, 0.0)] > by[(1.0, 0.0)]      # crediting the shorter narrows the gap
    assert by[(0.0, 100.0)] > by[(0.0, 0.0)]    # charging borrow widens it
    costs = st.cost_sweep(fr, fund="FUND", index="IDX", k=-1.0, cost_grid=(0.0, 10.0))
    assert costs[1]["gap_ann_pct"] > costs[0]["gap_ann_pct"]


def test_path_drag_table_shape_and_the_two_x_reset_drag(taxed):
    prices, _ = taxed
    fr = frame(prices)
    rows1 = st.path_drag(fr, fund="FUND", index="IDX", k=-1.0, horizons=(21, 63))
    rows2 = st.path_drag(fr, fund="FUND", index="IDX", k=-2.0, horizons=(21, 63))
    assert [r["horizon_days"] for r in rows1] == [21, 63]
    assert all(r["n_windows"] > 10 for r in rows1)
    # The pure reset drag is a constant-leverage property and grows with |k| and horizon.
    for a, b in zip(rows1, rows2):
        assert b["reset_minus_static_median_pp"] < a["reset_minus_static_median_pp"]
    assert (rows2[1]["reset_minus_static_median_pp"]
            < rows2[0]["reset_minus_static_median_pp"])


def test_financing_crosscheck_agrees_on_a_synthetic_tape(fair):
    prices, truth = fair
    fr = frame(prices)
    cc = st.financing_crosscheck(fr)
    assert abs(cc["irx_ann_pct"] - truth["rate_ann"] * 100.0) < 0.2
    assert abs(cc["diff_ann_pct"]) < 0.2


def test_race_reports_excess_of_cash_on_both_arms(fair):
    prices, _ = fair
    r = st.race(frame(prices), fund="FUND", index="IDX", k=-1.0)
    legs = r["legs"]
    assert np.allclose(r["e_fund"].to_numpy(),
                       (legs["r_fund"] - legs["r_cash"]).to_numpy())
    assert np.allclose(r["e_direct"].to_numpy(),
                       (r["r_direct"] - legs["r_cash"]).to_numpy())
    assert np.isfinite(r["sharpe_diff"])

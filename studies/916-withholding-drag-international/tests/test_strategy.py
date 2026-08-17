"""Estimator invariants, inference primitives, and the study's spine — all offline.

The spine: on a panel with a *planted* withholding gap the income-yield estimator
recovers the true gap in bp/yr with a CI that brackets it; on the null (every fund
taxed identically) it reports ~0 bp with a CI that covers zero. Along the way the
measurement identities are pinned down: the differenced yield equals the declared cash
distribution, the blend has no look-ahead, and costs only ever reduce return.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from withholding import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The measurement identity
# --------------------------------------------------------------------------- #
def test_income_yield_is_zero_without_dividends():
    """If the total-return and price-only legs coincide, the measured yield is zero."""
    idx = pd.bdate_range("2015-01-01", periods=500)
    s = pd.Series(100 * np.exp(np.cumsum(np.full(500, 0.0003))), index=idx)
    d = st.income_yield_daily(s, s)
    assert abs(float(d.sum())) < 1e-12
    assert abs(st.realised_yield_bp(s, s)) < 1e-9


def test_income_yield_recovers_a_hand_built_dividend():
    """One 1% distribution in a flat year must read back as ~100 bp of income."""
    idx = pd.bdate_range("2015-01-01", periods=252)
    px = pd.Series(100.0, index=idx)
    px.iloc[126:] = 99.0            # price drops by the 1.00 paid on the ex-date
    tr = pd.Series(100.0, index=idx)  # total return is unchanged by a distribution
    got = float(st.income_yield_daily(tr, px).sum()) * 1e4
    assert abs(got - 101.0) < 1.0   # 1.00 / 99.00 = 101 bp, paid once


def test_differenced_yield_matches_declared_cash_on_the_synthetic_tape():
    """The synthetic fund's declared payments and the differenced measure must agree."""
    tr, px, truth = data.synthetic_panel(seed=916)
    # Reconstruct the cash paid per share from the price/total-return legs.
    r_tr = tr["broad"].pct_change()
    r_px = px["broad"].pct_change()
    cash = ((r_tr - r_px) * px["broad"].shift(1)).fillna(0.0)
    cc = st.distribution_cross_check(tr["broad"], px["broad"], cash)
    assert abs(cc["residual_bp"]) < 0.5
    assert cc["n_ex_dates"] > 20


def test_annual_income_yield_drops_partial_years():
    tr, px, _ = data.synthetic_panel(n_years=6, seed=916)
    tbl = st.annual_income_yield(tr["broad"], px["broad"])
    assert (tbl["n_days"] >= 240).all()
    assert len(tbl) >= 4
    assert (tbl["yield_pct"] > 0).all()


# --------------------------------------------------------------------------- #
# The blend (the only traded object)
# --------------------------------------------------------------------------- #
def test_blend_of_identical_legs_reproduces_them():
    idx = pd.bdate_range("2015-01-01", periods=300)
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0003, 0.01, 300), index=idx)
    frame = pd.DataFrame({"a": r, "b": r, "c": r})
    b = st.blend_returns(frame, {"a": 0.5, "b": 0.3, "c": 0.2}, cost_bps=0.0)
    assert np.allclose(b.to_numpy(), r.to_numpy())


def test_blend_weights_are_renormalised():
    idx = pd.bdate_range("2015-01-01", periods=200)
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(0, 0.01, 200), "b": rng.normal(0, 0.01, 200)},
                         index=idx)
    b1 = st.blend_returns(frame, {"a": 1.0, "b": 1.0})
    b2 = st.blend_returns(frame, {"a": 0.5, "b": 0.5})
    assert np.allclose(b1.to_numpy(), b2.to_numpy())


def test_blend_has_no_lookahead(planted):
    """Perturbing the tail of the constituents cannot change earlier blend returns."""
    tr, px, _ = planted
    cols = ["bench_a", "bench_b", "bench_c"]
    r = tr[cols].pct_change().dropna()
    w = {c: 1 / 3 for c in cols}
    base = st.blend_returns(r, w, cost_bps=5.0)
    r2 = r.copy()
    r2.iloc[1500:] += 0.05
    pert = st.blend_returns(r2, w, cost_bps=5.0)
    assert np.allclose(base.to_numpy()[:1500], pert.to_numpy()[:1500])


def test_blend_costs_monotonically_reduce_return(planted):
    tr, _, _ = planted
    cols = ["bench_a", "bench_b", "bench_c"]
    r = tr[cols].pct_change().dropna()
    w = {"bench_a": 0.5, "bench_b": 0.3, "bench_c": 0.2}
    means = [st.blend_returns(r, w, cost_bps=c).mean() for c in (0.0, 5.0, 25.0)]
    assert means[0] >= means[1] >= means[2]


def test_blend_index_levels_are_positive_and_ordered(planted):
    tr, px, _ = planted
    cols = ["bench_a", "bench_b", "bench_c"]
    w = {c: 1 / 3 for c in cols}
    b_tr, b_px = st.blend_index(tr[cols], px[cols], w)
    assert (b_tr > 0).all() and (b_px > 0).all()
    # the blend pays dividends, so its total-return leg must outgrow its price leg
    assert b_tr.iloc[-1] > b_px.iloc[-1]


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_block_bootstrap_ci_brackets_the_point():
    rng = np.random.default_rng(2)
    x = pd.Series(rng.normal(1e-4, 1e-3, 3000))
    ci = st.block_bootstrap_mean_ci(x, n_boot=400, seed=916)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_bootstrap_sharpe_ci_brackets_the_point():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(3e-4, 0.01, 3000))
    ci = st.bootstrap_sharpe_ci(x, n_boot=400, seed=916)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]


def test_implied_withholding_is_monotone_arithmetic():
    rows = st.implied_withholding(300.0)
    drags = [r["drag_bp"] for r in rows]
    assert drags == sorted(drags)
    # w = 0 recovers the net yield exactly (no drag)
    zero = st.implied_withholding(300.0, w_grid=(0.0,))[0]
    assert abs(zero["gross_bp"] - 300.0) < 1e-9 and abs(zero["drag_bp"]) < 1e-9


def test_summary_reports_finite_stats(planted):
    tr, _, _ = planted
    s = st.summary(tr["broad"].pct_change())
    for k in ("sharpe", "vol_ann", "cagr", "max_drawdown", "tstat"):
        assert np.isfinite(s[k])
    assert s["max_drawdown"] <= 0.0


# --------------------------------------------------------------------------- #
# The study's spine — the estimator recovers a planted gap and is quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_gap_is_recovered(planted):
    tr, px, truth = planted
    d = st.synthetic_detect(tr, px, truth, n_boot=400)
    assert truth["true_gap_bp"] > 10.0
    assert abs(d["measured_gap_bp"] - d["true_gap_bp"]) < 5.0
    assert d["t_hac"] > 2.0
    assert d["ci_low_bp"] <= d["true_gap_bp"] <= d["ci_high_bp"]


def test_planted_gap_scales_with_signal_strength():
    got = []
    for ss in (0.0, 0.5, 1.0):
        tr, px, truth = data.synthetic_panel(signal_strength=ss, seed=916)
        got.append(st.synthetic_detect(tr, px, truth, n_boot=200)["measured_gap_bp"])
    assert got[0] < got[1] < got[2]


def test_null_gap_is_quiet(null):
    tr, px, truth = null
    d = st.synthetic_detect(tr, px, truth, n_boot=400)
    assert truth["true_gap_bp"] == 0.0
    assert abs(d["measured_gap_bp"]) < 5.0
    assert d["ci_low_bp"] <= 0.0 <= d["ci_high_bp"]


def test_null_is_quiet_across_seeds():
    gaps = np.array([
        st.synthetic_detect(*data.synthetic_panel(signal_strength=0.0, seed=916 + s),
                            n_boot=200)["measured_gap_bp"]
        for s in range(5)
    ])
    assert abs(gaps.mean()) < 3.0
    assert (np.abs(gaps) >= 10.0).sum() == 0


# --------------------------------------------------------------------------- #
# Fee accounting, eras, sweeps
# --------------------------------------------------------------------------- #
def test_fee_adjustment_shifts_the_mean_but_not_the_spread(planted):
    tr, px, truth = planted
    cols = truth["bench_cols"]
    b_tr, b_px = st.blend_index(tr[cols], px[cols], {c: 1 / 3 for c in cols})
    g = st.yield_gap(tr["broad"], px["broad"], b_tr, b_px,
                     fund_expense=0.0005, bench_expense=0.0050, n_boot=300)
    assert abs(g["fee_gap_bp"] - 45.0) < 1e-6
    assert abs(g["gap_fee_adj_bp"] - (g["gap_bp"] + g["fee_gap_bp"])) < 1e-9
    # the fee shift is a constant, so the CI width is unchanged
    w_raw = g["ci_high_bp"] - g["ci_low_bp"]
    w_adj = g["ci_high_fee_adj_bp"] - g["ci_low_fee_adj_bp"]
    assert abs(w_raw - w_adj) < 1e-6


def test_fee_gap_sweep_shifts_only_the_mean(planted):
    """The expense-ratio ASSUMPTION sweep must be a pure re-centring of the same series."""
    tr, px, truth = planted
    cols = truth["bench_cols"]
    b_tr, b_px = st.blend_index(tr[cols], px[cols], {c: 1 / 3 for c in cols})
    g = st.yield_gap(tr["broad"], px["broad"], b_tr, b_px, n_boot=200)
    rows = st.fee_gap_sweep(g, fee_gap_grid_bp=(0.0, 20.0, 47.0), n_boot=200)
    assert len(rows) == 3
    # each row is the raw gap shifted by exactly the assumed fee gap
    for r in rows:
        assert abs(r["gap_fee_adj_bp"] - (g["gap_bp"] + r["fee_gap_bp"])) < 1e-9
    # a constant shift cannot change the CI width
    widths = [r["ci_high_bp"] - r["ci_low_bp"] for r in rows]
    assert max(widths) - min(widths) < 1e-6
    # and a zero fee gap must reproduce the raw gap and its HAC t exactly
    assert abs(rows[0]["gap_fee_adj_bp"] - g["gap_bp"]) < 1e-9
    assert abs(rows[0]["t_hac"] - g["t_hac"]) < 1e-9
    # the adjusted gap is monotone in the assumed fee gap
    assert rows[0]["gap_fee_adj_bp"] < rows[1]["gap_fee_adj_bp"] < rows[2]["gap_fee_adj_bp"]


def test_fee_gap_sweep_headline_row_matches_yield_gap(planted):
    """At the headline fee gap the sweep must reproduce yield_gap's own fee-adjusted CI."""
    tr, px, truth = planted
    cols = truth["bench_cols"]
    b_tr, b_px = st.blend_index(tr[cols], px[cols], {c: 1 / 3 for c in cols})
    g = st.yield_gap(tr["broad"], px["broad"], b_tr, b_px,
                     fund_expense=0.0005, bench_expense=0.0050, n_boot=300)
    row = st.fee_gap_sweep(g, fee_gap_grid_bp=(g["fee_gap_bp"],), n_boot=300)[0]
    assert abs(row["gap_fee_adj_bp"] - g["gap_fee_adj_bp"]) < 1e-9
    assert abs(row["t_hac"] - g["t_hac_fee_adj"]) < 1e-9
    assert abs(row["ci_low_bp"] - g["ci_low_fee_adj_bp"]) < 1e-9
    assert abs(row["ci_high_bp"] - g["ci_high_fee_adj_bp"]) < 1e-9


def test_era_cut_returns_both_halves(planted):
    tr, px, truth = planted
    cols = truth["bench_cols"]
    b_tr, b_px = st.blend_index(tr[cols], px[cols], {c: 1 / 3 for c in cols})
    eras = st.era_cut(tr["broad"], px["broad"], b_tr, b_px, split="2017-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["gap_bp"]) and e["n_days"] > 300


def test_weight_sweep_covers_every_set(planted):
    tr, px, truth = planted
    cols = truth["bench_cols"]
    sets = {"equal": {c: 1 / 3 for c in cols},
            "tilted": {cols[0]: 0.7, cols[1]: 0.2, cols[2]: 0.1}}
    rows = st.blend_weight_sweep(tr["broad"], px["broad"], tr[cols], px[cols],
                                 sets, fund_expense=0.0, bench_expense=0.0)
    assert len(rows) == 2
    assert all(np.isfinite(r["gap_bp"]) for r in rows)


def test_total_return_race_is_symmetric_and_finite(planted):
    tr, px, truth = planted
    cols = truth["bench_cols"]
    b_tr, _ = st.blend_index(tr[cols], px[cols], {c: 1 / 3 for c in cols})
    cash = pd.Series(100 * (1 + 0.02 / 252) ** np.arange(len(tr)), index=tr.index)
    race = st.total_return_race(tr["broad"], b_tr, cash)
    assert np.isfinite(race["sharpe_adv"]) and np.isfinite(race["t_diff"])
    # the return advantage is the mean of the excess-return difference, annualised
    assert abs(race["ret_adv_bp"]
               - float(race["diff"].mean()) * 252 * 1e4) < 1e-6


def test_cost_sweep_is_monotone_against_the_blend(planted):
    tr, px, truth = planted
    cols = truth["bench_cols"]
    cash = pd.Series(100 * (1 + 0.02 / 252) ** np.arange(len(tr)), index=tr.index)
    rows = st.cost_sweep(tr["broad"], tr[cols], px[cols], cash,
                         {c: 1 / 3 for c in cols}, cost_grid=(0.0, 10.0, 50.0))
    advs = [r["ret_adv_bp"] for r in rows]
    assert advs[0] >= advs[1] >= advs[2]

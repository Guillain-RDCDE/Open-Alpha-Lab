"""Estimator logic, backtest invariants, and the study's spine — all offline/synthetic.

The spine: on a panel with a planted constant drag the trend estimator recovers it (and
recovers a planted fee on the spot wrapper, which is the real tape's ruler calibration);
on a panel with a planted compression the era test recovers the compression; on the null
— a large but *constant* basis straight through the event — the era test stays quiet.
The harvest charges borrow monotonically and respects its single execution lag.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etf_basis import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Tracking-difference plumbing
# --------------------------------------------------------------------------- #
def _straight_line(drag_ann=0.05, n=756, seed=0, noise=0.0):
    """A wrapper that bleeds ``drag_ann`` a year against a random-walk reference."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    ref = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, n))), index=idx)
    elapsed = np.asarray((idx - idx[0]).days, dtype=float) / 365.25
    wrap = ref * np.exp(-drag_ann * elapsed)
    if noise:
        wrap = wrap * np.exp(rng.normal(0, noise, n))
    return wrap, ref


def test_cumulative_diff_starts_at_zero_and_is_negative_for_a_bleeding_wrapper():
    wrap, ref = _straight_line()
    c = st.cumulative_diff(wrap, ref)
    assert abs(c.iloc[0]) < 1e-12
    assert c.iloc[-1] < 0


def test_trend_drag_recovers_a_known_constant_bleed():
    wrap, ref = _straight_line(drag_ann=0.05)
    res = st.trend_drag(wrap, ref)
    assert abs(res["drag_pct"] - (-5.0)) < 0.05
    assert res["t"] < -5


def test_naive_and_trend_agree_without_timestamp_noise():
    wrap, ref = _straight_line(drag_ann=0.03)
    assert abs(st.naive_drag(wrap, ref)["drag_pct"] - st.trend_drag(wrap, ref)["drag_pct"]) < 0.2


def test_trend_beats_naive_when_the_reference_is_offset_stamped():
    """With an offset-stamped reference the endpoint estimator is far noisier."""
    errs_naive, errs_trend = [], []
    for seed in range(12):
        wrap, ref = _straight_line(drag_ann=0.05, seed=seed, noise=0.03)
        errs_naive.append(abs(st.naive_drag(wrap, ref)["drag_pct"] + 5.0))
        errs_trend.append(abs(st.trend_drag(wrap, ref)["drag_pct"] + 5.0))
    assert np.mean(errs_trend) < np.mean(errs_naive)


def test_monthly_drag_recovers_a_known_bleed_without_hac():
    """The non-overlapping cross-check: same target, ordinary *t*, no HAC needed.

    ``noise=0.004`` is the *matched-close* case (BITO vs IBIT, same trading hours). With
    the fat offset of a differently-stamped reference this estimator is much weaker —
    which is exactly what it reads on the coin-referenced real tape, and the reason the
    verdict leans on the matched-close pairs instead.
    """
    wrap, ref = _straight_line(drag_ann=0.05, noise=0.004)
    mo = st.monthly_drag(wrap, ref)
    assert abs(mo["drag_pct"] - (-5.0)) < 1.0
    assert mo["t"] < -2 and mo["n_months"] >= 30

    noisy, ref2 = _straight_line(drag_ann=0.05, noise=0.03)
    assert abs(st.monthly_drag(noisy, ref2)["t"]) < abs(mo["t"])


def test_monthly_drag_is_less_certain_than_the_trend_slope(compressed):
    """It must be the CONSERVATIVE ruler — it throws information away on purpose."""
    prices, _ = compressed
    mo = st.monthly_drag(prices["futures_etf"], prices["spot"])
    tr = st.trend_drag(prices["futures_etf"], prices["spot"])
    assert abs(mo["t"]) < abs(tr["t"])
    assert np.sign(mo["drag_pct"]) == np.sign(tr["drag_pct"])


def test_residual_diagnostics_separate_a_stationary_from_a_wandering_residual():
    """A trend + i.i.d. noise must look stationary; a trend + random walk must not."""
    rng = np.random.default_rng(958)
    n = 900
    idx = pd.bdate_range("2021-01-04", periods=n)
    ref = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, n))), index=idx)
    elapsed = np.asarray((idx - idx[0]).days, dtype=float) / 365.25
    clean = ref * np.exp(-0.05 * elapsed + rng.normal(0, 0.02, n))
    wander = ref * np.exp(-0.05 * elapsed + np.cumsum(rng.normal(0, 0.004, n)))
    good = st.trend_residual_diagnostics(clean, ref)
    bad = st.trend_residual_diagnostics(wander, ref)
    assert good["df_t"] < -5.0 and good["ar1"] < 0.5
    assert bad["df_t"] > good["df_t"] and bad["ar1"] > 0.9


def test_monthly_drag_degrades_gracefully():
    wrap, ref = _straight_line(n=40)
    assert not np.isfinite(st.monthly_drag(wrap, ref)["drag_pct"])


def test_cycle_regression_finds_a_planted_state_dependence():
    """A drag deliberately made to depend on the reference's trailing return."""
    rng = np.random.default_rng(958)
    n = 1000
    idx = pd.bdate_range("2021-01-04", periods=n)
    r = rng.normal(0.0, 0.02, n)
    ref = pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx)
    trail = pd.Series(r, index=idx).rolling(126).sum().fillna(0.0).to_numpy()
    daily_drag = -(0.05 + 0.30 * trail) / 252.0          # richer carry after rallies
    wrap = ref * np.exp(np.cumsum(daily_drag))
    res = st.cycle_regression(wrap, ref)
    assert res["slope"] < 0 and res["corr"] < 0
    assert abs(res["t"]) < abs(res["t_ols"])              # HAC must PUNISH the overlap
    assert res["n_eff"] < 10.0                            # and the sample is tiny


def test_cycle_regression_is_quiet_on_a_constant_drag():
    wrap, ref = _straight_line(drag_ann=0.05, n=1000, noise=0.02)
    res = st.cycle_regression(wrap, ref)
    assert abs(res["t"]) < 3.0


def test_cycle_regression_nan_on_a_short_series():
    wrap, ref = _straight_line(n=200)
    assert not np.isfinite(st.cycle_regression(wrap, ref)["t"])


def test_align_windows_and_intersects():
    wrap, ref = _straight_line(n=500)
    w, r = st.align(wrap, ref.iloc[100:], lo=wrap.index[200])
    assert w.index.equals(r.index) and w.index[0] == wrap.index[200]


def test_trend_drag_returns_nan_on_a_stub():
    wrap, ref = _straight_line(n=10)
    assert not np.isfinite(st.trend_drag(wrap, ref)["drag_pct"])


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean_and_is_quiet_on_noise():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_hac_ols_matches_ols_point_estimates():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 400)
    y = 1.5 + 2.0 * x + rng.normal(0, 0.5, 400)
    X = np.column_stack([np.ones_like(x), x])
    beta, se = st.hac_ols(X, y)
    assert abs(beta[1] - 2.0) < 0.1 and (se > 0).all()


def test_implied_basis_inverts_the_decomposition():
    ib = st.implied_basis(drag_pct=-6.0, fee=0.0095, cash_rate=0.045)
    assert abs(ib["basis_pct"] - (4.5 - 0.95 + 6.0)) < 1e-9
    # the excess-of-cash basis must not depend on the cash assumption
    other = st.implied_basis(drag_pct=-6.0, fee=0.0095, cash_rate=0.01)
    assert abs(ib["excess_basis_pct"] - other["excess_basis_pct"]) < 1e-9


def test_fee_sweep_moves_the_basis_one_for_one():
    rows = st.fee_sweep(-6.0, cash_rate=0.045, fee_grid=(0.0075, 0.0095))
    assert abs((rows[0]["basis_pct"] - rows[1]["basis_pct"]) - 0.20) < 1e-9


def test_bootstrap_ci_brackets_the_point_estimate():
    wrap, ref = _straight_line(drag_ann=0.05, noise=0.02)
    ci = st.bootstrap_drag_ci(wrap, ref, n_boot=400, seed=958)
    assert ci["ci_low"] <= ci["drag_pct"] <= ci["ci_high"]


# --------------------------------------------------------------------------- #
# The era test — planted compression vs the null
# --------------------------------------------------------------------------- #
def test_era_test_recovers_a_planted_compression(compressed):
    prices, truth = compressed
    res = st.piecewise_drag(prices["futures_etf"], prices["spot"], split=truth["event_date"])
    assert abs(res["change_pct"] - truth["drag_change_pct"]) < 1.5
    assert res["t"] > 3  # a genuine compression is a positive change in slope


def test_era_test_is_quiet_on_the_null(uncompressed):
    prices, truth = uncompressed
    res = st.piecewise_drag(prices["futures_etf"], prices["spot"], split=truth["event_date"])
    assert abs(res["change_pct"]) < 2.0


def test_null_era_test_centred_across_seeds():
    changes = np.array([
        st.piecewise_drag(p["futures_etf"], p["spot"], split=t["event_date"])["change_pct"]
        for p, t in (data.synthetic_panel(signal_strength=0.0, seed=958 + s) for s in range(6))
    ])
    assert abs(changes.mean()) < 1.0
    assert (np.abs(changes) >= 2.0).sum() <= 1


def test_era_test_preserves_the_level_in_both_halves(uncompressed):
    """On the null both halves must read the same (large) planted drag."""
    prices, truth = uncompressed
    res = st.piecewise_drag(prices["futures_etf"], prices["spot"], split=truth["event_date"])
    assert abs(res["pre_pct"] - truth["drag_fut_pre_pct"]) < 2.0
    assert abs(res["post_pct"] - truth["drag_fut_post_pct"]) < 2.0


def test_piecewise_nan_when_the_split_is_outside_the_window(compressed):
    prices, _ = compressed
    res = st.piecewise_drag(prices["futures_etf"], prices["spot"], split="1999-01-01")
    assert not np.isfinite(res["change_pct"])


def test_matched_window_sweep_agrees_on_the_sign_of_a_planted_break(compressed):
    """Every symmetric width must see a planted compression as a compression."""
    prices, truth = compressed
    tbl = st.matched_window_sweep(prices["futures_etf"], prices["spot"],
                                  split=truth["event_date"], months=(6, 12, 18))
    assert len(tbl) == 3 and {"pre_pct", "post_pct", "change_pct", "t"}.issubset(tbl.columns)
    assert (tbl["change_pct"] > 0).all()


def test_matched_window_sweep_is_quiet_on_the_null(uncompressed, compressed):
    """On the null every width must be far smaller than the planted break.

    Note the *narrowest* window is the noisiest — 252 sessions of a fat timestamp offset
    buy little precision — which is precisely why the whole family is published on the
    real tape rather than one hand-picked width.
    """
    prices, truth = uncompressed
    tbl = st.matched_window_sweep(prices["futures_etf"], prices["spot"],
                                  split=truth["event_date"], months=(6, 12, 18))
    planted, t_p = compressed
    tbl_p = st.matched_window_sweep(planted["futures_etf"], planted["spot"],
                                    split=t_p["event_date"], months=(6, 12, 18))
    assert tbl["change_pct"].abs().max() < 0.5 * tbl_p["change_pct"].min()
    assert abs(tbl["change_pct"].mean()) < 2.5


def test_placebo_sweep_returns_many_dates_and_ranks(compressed):
    prices, truth = compressed
    sweep = st.placebo_split_sweep(prices["futures_etf"], prices["spot"], freq="MS")
    assert len(sweep) > 10 and {"split", "change_pct", "t"}.issubset(sweep.columns)
    real = st.piecewise_drag(prices["futures_etf"], prices["spot"], split=truth["event_date"])
    rank = st.placebo_rank(sweep, real["t"])
    # a genuinely planted break at the true date should top the placebo distribution
    assert rank["rank"] <= 3 and rank["n"] == len(sweep)


def test_annual_drag_table_shape(compressed):
    prices, _ = compressed
    tbl = st.annual_drag_table(prices["futures_etf"], prices["spot"])
    assert {"drag_pct", "t", "n"}.issubset(tbl.columns)
    assert len(tbl) >= 3 and (tbl["n"] >= 60).all()


# --------------------------------------------------------------------------- #
# The ruler calibration — a spot wrapper must read its own fee
# --------------------------------------------------------------------------- #
def test_spot_wrapper_drag_recovers_the_planted_fee(compressed):
    prices, truth = compressed
    res = st.trend_drag(prices["spot_etf"], prices["spot"])
    assert abs(res["drag_pct"] - truth["drag_spot_etf_pct"]) < 0.25


def test_synthetic_detect_reports_every_planted_quantity(compressed):
    prices, truth = compressed
    d = st.synthetic_detect(prices, truth)
    for key in ("spot_etf_drag_pct", "fut_drag_trend_pct", "era_change_pct", "era_t"):
        assert np.isfinite(d[key])
    assert d["era_change_pct"] > 0 and d["planted_change_pct"] > 0


# --------------------------------------------------------------------------- #
# The harvest — borrow, costs and the single execution lag
# --------------------------------------------------------------------------- #
def test_pair_trade_columns_and_no_nans(compressed):
    prices, _ = compressed
    pr = st.pair_trade(prices["spot_etf"], prices["futures_etf"])
    assert {"r_long", "r_short", "w_long", "w_short", "gross", "borrow", "cost",
            "net"}.issubset(pr.columns)
    assert not pr.isna().any().any()


def test_pair_trade_harvests_the_planted_carry(compressed):
    prices, _ = compressed
    s = st.pair_summary(st.pair_trade(prices["spot_etf"], prices["futures_etf"],
                                      borrow_ann=0.0, cost_bps=0.0), "gross")
    assert s["ann_pct"] > 0 and s["t"] > 2


def test_borrow_and_costs_monotonically_reduce_the_harvest(compressed):
    prices, _ = compressed
    anns = [st.pair_summary(st.pair_trade(prices["spot_etf"], prices["futures_etf"],
                                          borrow_ann=b, cost_bps=5.0), "net")["ann_pct"]
            for b in (0.0, 0.02, 0.05)]
    assert anns[0] > anns[1] > anns[2]
    cheap = st.pair_summary(st.pair_trade(prices["spot_etf"], prices["futures_etf"],
                                          borrow_ann=0.02, cost_bps=0.0), "net")["ann_pct"]
    dear = st.pair_summary(st.pair_trade(prices["spot_etf"], prices["futures_etf"],
                                         borrow_ann=0.02, cost_bps=25.0), "net")["ann_pct"]
    assert cheap > dear


def test_pair_weights_are_lagged_one_day(compressed):
    """Day one is traded at unit weights; the reset formed at t is only live at t+1."""
    prices, _ = compressed
    pr = st.pair_trade(prices["spot_etf"], prices["futures_etf"], rebalance_days=5,
                       borrow_ann=0.0, cost_bps=0.0)
    assert pr["w_long"].iat[0] == 1.0 and pr["w_short"].iat[0] == -1.0
    # after a reset on session i, the fresh weights appear on session i+1, not on i
    assert pr["w_long"].iat[5] == 1.0
    assert pr["w_long"].iat[4] != 1.0


def test_pair_gross_matches_the_weighted_legs(compressed):
    prices, _ = compressed
    pr = st.pair_trade(prices["spot_etf"], prices["futures_etf"], borrow_ann=0.0, cost_bps=0.0)
    recomputed = pr["w_long"] * pr["r_long"] + pr["w_short"] * pr["r_short"]
    assert np.allclose(recomputed.to_numpy(), pr["gross"].to_numpy())


def test_borrow_sweep_and_pair_by_year_shapes(compressed):
    prices, _ = compressed
    rows = st.borrow_sweep(prices["spot_etf"], prices["futures_etf"],
                           borrow_grid=(0.0, 0.02), cost_grid=(0.0, 5.0))
    assert len(rows) == 4 and all(np.isfinite(r["ann_pct"]) for r in rows)
    tbl = st.pair_by_year(prices["spot_etf"], prices["futures_etf"])
    assert {"gross_pct", "net_pct", "sharpe"}.issubset(tbl.columns) and len(tbl) >= 3


def test_bootstrap_sharpe_ci_brackets_point(compressed):
    prices, _ = compressed
    pr = st.pair_trade(prices["spot_etf"], prices["futures_etf"])
    ci = st.bootstrap_sharpe_ci(pr["net"], n_boot=300, seed=958)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]


@pytest.mark.parametrize("n", [0, 5, 20])
def test_estimators_degrade_gracefully_on_short_series(n):
    idx = pd.bdate_range("2024-01-02", periods=max(n, 1))
    s = pd.Series(np.linspace(100, 101, max(n, 1)), index=idx)
    assert not np.isfinite(st.trend_drag(s.iloc[:n], s.iloc[:n])["drag_pct"])
    assert not np.isfinite(st.naive_drag(s.iloc[:n], s.iloc[:n])["drag_pct"])

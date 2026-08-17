"""Fit mechanics, backtest invariants, and the study's spine — all offline/synthetic.

The spine: on a fund with a *planted* alpha and a *planted* delta ratchet the harness
recovers the true static weights, reports a positive out-of-sample alpha with |t| >= 2,
and finds a positive convexity smile; on the null — a fund that is exactly a static mix
plus noise — it reports neither. Costs and fees move the residual in the documented
direction, and the weights are fitted on in-sample data only (no look-ahead).
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from convert_repl import data, strategy as st  # noqa: E402

FACTORS = ("eq", "growth", "credit")


def _is_end(prices):
    return str(prices.index[len(prices) // 2].date())


# --------------------------------------------------------------------------- #
# Projection & constrained least squares
# --------------------------------------------------------------------------- #
def test_project_simplex_sums_to_one_and_is_nonneg():
    v = np.array([0.7, -0.3, 1.4, 0.2])
    w = st.project_simplex(v)
    assert (w >= -1e-12).all()
    assert abs(w.sum() - 1.0) < 1e-9


def test_project_capped_leaves_feasible_points_alone():
    v = np.array([0.2, 0.3, 0.1])
    assert np.allclose(st.project_capped(v), v)


def test_project_capped_enforces_budget():
    w = st.project_capped(np.array([0.9, 0.8, -0.4]), max_sum=1.0)
    assert (w >= -1e-12).all()
    assert w.sum() <= 1.0 + 1e-9


def test_constrained_ls_recovers_known_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.01, (4000, 3))
    w_true = np.array([0.4, 0.2, 0.25])
    y = 0.0002 + X @ w_true + rng.normal(0, 0.002, 4000)
    a, w = st.constrained_ls(y, X)
    assert np.allclose(w, w_true, atol=0.02)
    assert abs(a - 0.0002) < 5e-4


def test_constrained_ls_respects_the_budget():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 0.01, (2000, 3))
    y = X @ np.array([1.2, 0.8, -0.5]) + rng.normal(0, 0.001, 2000)
    _, w = st.constrained_ls(y, X, nonneg=True, max_sum=1.0)
    assert (w >= -1e-9).all() and w.sum() <= 1.0 + 1e-9


def test_unconstrained_ls_matches_plain_ols():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 0.01, (1500, 2))
    y = X @ np.array([1.5, -0.7]) + rng.normal(0, 0.001, 1500)
    _, w = st.constrained_ls(y, X, nonneg=False)
    Z = np.column_stack([np.ones(len(X)), X])
    ols = np.linalg.pinv(Z.T @ Z) @ (Z.T @ y)
    assert np.allclose(w, ols[1:], atol=1e-8)


# --------------------------------------------------------------------------- #
# Replica path invariants
# --------------------------------------------------------------------------- #
def test_replica_is_exactly_the_weighted_excess_when_frictionless(planted):
    prices, _ = planted
    rets = prices.pct_change().dropna()
    w = np.array([0.3, 0.2, 0.35])
    got = st.replica_excess(rets[list(FACTORS)], rets["cash"], w, cost_bps=0.0, fee_ann=0.0)
    want = rets[list(FACTORS)].sub(rets["cash"], axis=0).to_numpy() @ w
    # the first day is exactly on target; drift only accumulates afterwards
    assert abs(got.iloc[0] - want[0]) < 1e-12
    assert np.corrcoef(got.to_numpy(), want)[0, 1] > 0.999


def test_costs_and_fees_only_reduce_the_replica(planted):
    prices, _ = planted
    rets = prices.pct_change().dropna()
    w = np.array([0.3, 0.2, 0.35])
    base = st.replica_excess(rets[list(FACTORS)], rets["cash"], w, cost_bps=0.0, fee_ann=0.0)
    costed = st.replica_excess(rets[list(FACTORS)], rets["cash"], w, cost_bps=10.0, fee_ann=0.0)
    feed = st.replica_excess(rets[list(FACTORS)], rets["cash"], w, cost_bps=0.0, fee_ann=0.005)
    assert costed.mean() < base.mean()
    assert feed.mean() < base.mean()
    assert abs(feed.mean() - (base.mean() - 0.005 / 252)) < 1e-12


def test_cash_only_replica_has_zero_excess(planted):
    prices, _ = planted
    rets = prices.pct_change().dropna()
    z = st.replica_excess(rets[list(FACTORS)], rets["cash"], np.zeros(3),
                          cost_bps=5.0, fee_ann=0.0)
    assert float(np.abs(z).max()) < 1e-12


def test_rebalance_is_lagged_one_day_past_the_month_end(planted):
    """The month-end drift must be carried into the first day of the new month.

    If the replica were traded back to target at the *month-end* close (the zero-lag
    variant), the first day of every new month would be earned on exactly the target
    weights. It must not be: the study claims one execution lag, so that day's return has
    to come from the drifted book.
    """
    prices, _ = planted
    rets = prices.pct_change().dropna()
    w = np.array([0.3, 0.2, 0.35])
    got = st.replica_excess(rets[list(FACTORS)], rets["cash"], w,
                            cost_bps=0.0, fee_ann=0.0)
    static = pd.Series(rets[list(FACTORS)].sub(rets["cash"], axis=0).to_numpy() @ w,
                       index=rets.index)
    months = rets.index.to_period("M")
    first_of_month = pd.Series(months, index=rets.index).ne(
        pd.Series(months, index=rets.index).shift(1))
    first_of_month.iloc[0] = False
    gap = (got - static)[first_of_month]
    assert len(gap) > 100
    assert float(gap.abs().max()) > 1e-6, "first day of the month is on target => no lag"


def test_borrow_is_charged_on_a_short_leg(planted):
    prices, _ = planted
    rets = prices.pct_change().dropna()
    w = np.array([0.6, -0.3, 0.2])
    with_borrow = st.replica_excess(rets[list(FACTORS)], rets["cash"], w,
                                    cost_bps=0.0, fee_ann=0.0, borrow_bps_ann=200.0)
    free = st.replica_excess(rets[list(FACTORS)], rets["cash"], w,
                             cost_bps=0.0, fee_ann=0.0, borrow_bps_ann=0.0)
    assert with_borrow.mean() < free.mean()


# --------------------------------------------------------------------------- #
# Fit / hold-out mechanics — no look-ahead
# --------------------------------------------------------------------------- #
def test_fit_uses_in_sample_only(planted):
    """Scrambling the post-IS tape must not change the fitted weights."""
    prices, _ = planted
    cut = _is_end(prices)
    fit_a = st.fit_replication(prices, "fund", FACTORS, "cash", cut)
    tampered = prices.copy()
    mask = tampered.index > pd.Timestamp(cut)
    tampered.loc[mask, "fund"] = tampered.loc[mask, "fund"] * 3.0
    fit_b = st.fit_replication(tampered, "fund", FACTORS, "cash", cut)
    assert np.allclose(fit_a["w_vec"], fit_b["w_vec"], atol=1e-6)


def test_holdout_windows_do_not_overlap(planted):
    prices, _ = planted
    cut = _is_end(prices)
    res = st.holdout(prices, "fund", FACTORS, "cash", cut)
    assert res["is"]["n_days"] + res["oos"]["n_days"] == len(res["diff"])
    assert pd.Timestamp(res["oos"]["start"]) > pd.Timestamp(cut)
    assert pd.Timestamp(res["is"]["end"]) <= pd.Timestamp(cut)


def test_holdout_weights_are_feasible(planted):
    prices, _ = planted
    res = st.holdout(prices, "fund", FACTORS, "cash", _is_end(prices))
    w = res["fit"]["w_vec"]
    assert (w >= -1e-9).all() and w.sum() <= 1.0 + 1e-9
    assert abs(res["fit"]["w_cash"] - (1.0 - w.sum())) < 1e-9


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_hac_ols_recovers_planted_curvature():
    rng = np.random.default_rng(3)
    m = rng.normal(0, 0.04, 3000)
    y = 0.001 + 0.5 * m + 2.0 * m ** 2 + rng.normal(0, 0.005, 3000)
    fit = st.hac_ols(y, np.column_stack([m, m ** 2]))
    assert abs(fit["beta"][2] - 2.0) < 0.3
    assert fit["t"][2] > 3


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    rng = np.random.default_rng(4)
    x = pd.Series(0.0002 + rng.normal(0, 0.008, 3000))
    a = st.block_bootstrap_mean_ci(x, n_boot=400, seed=953)
    b = st.block_bootstrap_mean_ci(x, n_boot=400, seed=953)
    assert a["ci_low"] <= a["mean_ann"] <= a["ci_high"]
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]


def test_welch_and_one_sample_finite():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.1, 1, 300), rng.normal(0, 1, 300)
    assert np.isfinite(st.welch_t(a, b)) and np.isfinite(st.one_sample_t(a))


def test_to_monthly_compounds():
    idx = pd.bdate_range("2020-01-01", periods=44)
    s = pd.Series(0.001, index=idx)
    m = st.to_monthly(s)
    assert len(m) == len(set(idx.to_period("M")))
    assert abs(m.iloc[0] - ((1.001 ** int((idx.month == 1).sum())) - 1.0)) < 1e-12


# --------------------------------------------------------------------------- #
# The study's spine — the harness recovers what is planted, and only that
# --------------------------------------------------------------------------- #
def test_planted_weights_are_recovered(planted):
    prices, truth = planted
    res = st.holdout(prices, "fund", FACTORS, "cash", _is_end(prices))
    got = np.array([res["fit"]["weights"][c] for c in FACTORS])
    assert np.allclose(got, np.array(truth["w_true"]), atol=0.06)


def test_planted_alpha_is_detected(planted):
    prices, _ = planted
    d = st.synthetic_detect(prices)
    assert d["alpha_oos_ann"] > 0.01
    assert d["t_oos"] > 2.0


def test_planted_convexity_is_detected(planted):
    prices, _ = planted
    d = st.synthetic_detect(prices)
    assert d["tm_gamma"] > 0.0
    assert d["t_tm_gamma"] > 2.0
    assert d["smile_bps"] > 0.0


def test_null_shows_no_alpha(null):
    prices, _ = null
    d = st.synthetic_detect(prices)
    assert abs(d["t_oos"]) < 2.0


def test_null_shows_no_convexity(null):
    prices, _ = null
    d = st.synthetic_detect(prices)
    assert abs(d["t_tm_gamma"]) < 2.0


def test_null_is_quiet_across_seeds():
    ts = np.array([
        [d["t_oos"], d["t_tm_gamma"]]
        for d in (st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=953 + s)[0])
                  for s in range(6))
    ])
    assert abs(ts[:, 0].mean()) < 1.0
    assert (np.abs(ts[:, 0]) >= 2.0).sum() <= 1
    assert (np.abs(ts[:, 1]) >= 2.0).sum() <= 1


def test_null_replication_is_tight(null):
    """With no planted effect the replica should track the fund closely (high R^2)."""
    prices, _ = null
    res = st.holdout(prices, "fund", FACTORS, "cash", _is_end(prices))
    assert res["fit"]["r2_is"] > 0.8
    assert res["oos"]["corr"] > 0.9


# --------------------------------------------------------------------------- #
# Robustness helpers
# --------------------------------------------------------------------------- #
def test_convexity_test_shape(planted):
    prices, _ = planted
    res = st.holdout(prices, "fund", FACTORS, "cash", _is_end(prices))
    rets = prices.pct_change().dropna()
    cx = st.convexity_test(res["ex_fund"], res["replica"], rets["eq"] - rets["cash"],
                           mask=res["oos_mask"])
    assert cx["n_months"] > 40
    for k in ("mean_low_bps", "mean_high_bps", "up_capture", "down_capture", "tm_gamma"):
        assert np.isfinite(cx[k])


def test_era_cut_returns_both_halves(planted):
    prices, _ = planted
    res = st.holdout(prices, "fund", FACTORS, "cash", _is_end(prices))
    mid = res["diff"][res["oos_mask"]]
    split = str(mid.index[len(mid) // 2].date())
    eras = st.era_cut(res, split=split)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["alpha_ann"]) and np.isfinite(e["t_hac"])


def test_cost_sweep_alpha_rises_with_the_replica_fee(planted):
    prices, _ = planted
    rows = st.cost_sweep(prices, "fund", FACTORS, "cash", _is_end(prices),
                         cost_grid=(3.0,), fee_grid=(0.0, 0.002))
    assert rows[1]["alpha_ann"] > rows[0]["alpha_ann"]  # the replica's fee flatters the fund


def test_cost_sweep_alpha_rises_with_rebalance_cost(planted):
    prices, _ = planted
    rows = st.cost_sweep(prices, "fund", FACTORS, "cash", _is_end(prices),
                         cost_grid=(0.0, 25.0), fee_grid=(0.0,))
    assert rows[1]["alpha_ann"] >= rows[0]["alpha_ann"]


def test_factor_set_sweep_shrinks_the_residual_as_the_kit_grows(planted):
    """A richer kit must explain more in-sample — that is the hindsight the sweep exposes."""
    prices, _ = planted
    rows = st.factor_set_sweep(prices, "fund", [("eq",), ("eq", "growth", "credit")],
                               "cash", _is_end(prices))
    assert len(rows) == 2
    assert rows[1]["r2_is"] >= rows[0]["r2_is"]
    for r in rows:
        assert np.isfinite(r["alpha_ann"]) and np.isfinite(r["t_hac"])


def test_split_sweep_covers_every_boundary(planted):
    prices, _ = planted
    cuts = [str(prices.index[i].date()) for i in (len(prices) // 3, len(prices) // 2)]
    rows = st.split_sweep(prices, "fund", FACTORS, "cash", cuts)
    assert [r["is_end"] for r in rows] == cuts
    assert rows[0]["n_oos"] > rows[1]["n_oos"]  # an earlier cut leaves a longer hold-out
    for r in rows:
        assert np.isfinite(r["alpha_ann"]) and np.isfinite(r["t_hac"])


def test_rolling_refit_runs_and_is_forward_only(planted):
    prices, _ = planted
    first = str(prices.index[len(prices) // 3].date())
    rr = st.rolling_refit(prices, "fund", FACTORS, "cash", first)
    assert rr["n_days"] > 200
    assert pd.Timestamp(rr["start"]) > pd.Timestamp(first).replace(month=1, day=1)
    assert np.isfinite(rr["alpha_ann"]) and np.isfinite(rr["t_hac"])


def test_summary_and_drawdown(planted):
    prices, _ = planted
    ex = st.excess_returns(prices, "cash")["fund"]
    s = st.summary(ex)
    assert s["n_days"] > 1000 and np.isfinite(s["sharpe"])
    assert -1.0 < st.max_drawdown(ex) <= 0.0

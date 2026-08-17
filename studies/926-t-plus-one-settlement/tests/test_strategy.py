"""Decomposition identity, DiD estimator behaviour, and the study's spine — all offline.

The spine: on a panel with a *planted* post-switch break in the treated leg, the
difference-in-difference must recover it (right sign, |t| >= 2, point estimate close to
the planted size); on a panel where nothing changes at the switch it must stay quiet
across seeds. Costs and borrow only ever reduce the overlay's return, and the overlay's
position carries exactly one execution lag.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t_plus_one import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The night/day identity
# --------------------------------------------------------------------------- #
def test_decompose_identity_is_exact(planted):
    panel, _ = planted
    d = st.decompose(panel["treated"])
    lhs = (1.0 + d["r_on"]) * (1.0 + d["r_id"])
    rhs = 1.0 + d["r_cc"]
    assert np.allclose(lhs.to_numpy(), rhs.to_numpy(), atol=1e-12)


def test_decompose_is_case_insensitive(planted):
    panel, _ = planted
    upper = panel["treated"].rename(columns=str.capitalize)
    a = st.decompose(panel["treated"])
    b = st.decompose(upper)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_daily_outcomes_units_and_bounds(planted):
    panel, _ = planted
    y = st.daily_outcomes(panel["treated"])
    assert {"r_on", "r_id", "abs_on", "abs_cc", "on_var_share"}.issubset(y.columns)
    share = y["on_var_share"].dropna()
    assert (share >= 0.0).all() and (share <= 1.0).all()
    assert np.allclose(y["abs_on"].to_numpy(), y["r_on"].abs().to_numpy())


# --------------------------------------------------------------------------- #
# Calendar masks
# --------------------------------------------------------------------------- #
def test_turn_of_month_mask_picks_month_boundaries():
    idx = pd.bdate_range("2021-01-04", periods=500)
    tom = st.turn_of_month_mask(idx, n_last=1, n_first=3)
    assert 0.10 < tom.mean() < 0.25          # ~4 of ~21 trading days
    # every day whose month differs from the next day's is a month-end and must be flagged
    months = idx.to_period("M").astype(str).to_numpy()
    is_month_end = np.append(months[:-1] != months[1:], True)
    assert tom.to_numpy()[is_month_end].all()
    # and the first trading day of each month must be flagged too
    is_month_start = np.append(True, months[1:] != months[:-1])
    assert tom.to_numpy()[is_month_start].all()


def test_quarter_turn_mask_is_a_subset_of_turn_of_month():
    idx = pd.bdate_range("2021-01-04", periods=500)
    tom = st.turn_of_month_mask(idx)
    q = st.quarter_turn_mask(idx)
    assert (q & ~tom).sum() == 0
    assert 0 < q.sum() < tom.sum()


def test_month_ranks_reset_each_month():
    idx = pd.bdate_range("2022-01-03", periods=200)
    fwd, bwd = st.month_ranks(idx)
    assert fwd[0] == 0 and bwd[-1] == 0
    assert (fwd >= 0).all() and (bwd >= 0).all()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_hac_ols_recovers_a_planted_step():
    rng = np.random.default_rng(2)
    n = 2000
    post = np.zeros(n); post[n // 2:] = 1.0
    y = 0.5 + 2.0 * post + rng.normal(0, 1.0, n)
    fit = st.hac_ols(y, np.column_stack([np.ones(n), post]))
    assert abs(fit["beta"][1] - 2.0) < 0.2
    assert fit["t"][1] > 5


def test_hac_ols_quiet_without_a_step():
    rng = np.random.default_rng(3)
    n = 2000
    post = np.zeros(n); post[n // 2:] = 1.0
    y = rng.normal(0, 1.0, n)
    fit = st.hac_ols(y, np.column_stack([np.ones(n), post]))
    assert abs(fit["t"][1]) < 3


def test_welch_and_one_sample_finite(planted):
    panel, _ = planted
    y = st.daily_outcomes(panel["treated"])["r_on"].to_numpy()
    assert np.isfinite(st.one_sample_t(y))
    assert np.isfinite(st.welch_t(y[:500], y[500:]))


def test_block_bootstrap_ci_brackets_point(planted):
    panel, _ = planted
    y = st.daily_outcomes(panel["treated"])["abs_on"]
    ci = st.block_bootstrap_ci(y, lambda a: float(a.mean()), n_boot=400, seed=926)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


# --------------------------------------------------------------------------- #
# The DiD estimator
# --------------------------------------------------------------------------- #
def test_symmetric_windows_are_equal_length_and_straddle_the_switch(planted):
    panel, truth = planted
    idx = panel["treated"].index
    asof = str(idx[-1].date())
    win = st.symmetric_windows(idx, truth["switch_date"], asof)
    assert len(win["pre"]) == len(win["post"]) == win["n_each"]
    assert win["pre_end"] < pd.Timestamp(truth["switch_date"]) <= win["post_start"]
    assert win["post_end"] <= pd.Timestamp(asof)


def test_did_recovers_the_planted_overnight_drift(planted):
    panel, truth = planted
    d = st.synthetic_detect(panel, truth)
    # The planted shift is the constant bump plus the turn-of-month bump on ~19% of days.
    expected = truth["planted_on_drift_bps"] + truth["planted_tom_bps"] * truth["tom_frac"]
    assert d["did_r_on_bps"] > 0
    assert abs(d["did_r_on_bps"] - expected) < 0.5 * expected
    assert d["t_r_on"] > 2.0


def test_did_recovers_the_planted_overnight_vol_lift(planted):
    panel, truth = planted
    d = st.synthetic_detect(panel, truth)
    assert d["did_abs_on_bps"] > 0
    assert d["t_abs_on"] > 2.0


def test_did_quiet_on_the_null(null_world):
    panel, truth = null_world
    d = st.synthetic_detect(panel, truth)
    for key in ("t_r_on", "t_abs_on", "t_on_var_share", "t_abs_cc"):
        assert abs(d[key]) < 2.0, key


def test_did_quiet_on_the_null_across_seeds():
    fires = 0
    for s in range(6):
        panel, truth = data.synthetic_panel(signal_strength=0.0, seed=926 + s)
        d = st.synthetic_detect(panel, truth)
        fires += int(abs(d["t_r_on"]) >= 2.0)
    assert fires <= 1


def test_did_sign_flips_with_the_planted_sign():
    panel, truth = data.synthetic_panel(signal_strength=1.0, seed=926,
                                        d_on_drift_bps=-8.0, d_tom_bps=-45.0)
    d = st.synthetic_detect(panel, truth)
    assert d["did_r_on_bps"] < 0


def test_did_bootstrap_ci_brackets_point(planted):
    panel, truth = planted
    tab = st.did_table(panel, "treated", "control", truth["switch_date"],
                       str(panel["treated"].index[-1].date()))
    ci = st.did_bootstrap_ci(tab["r_on"], n_boot=400, seed=926)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert abs(ci["point"] - tab["r_on"]["did"]) < 1e-9


def test_window_length_sweep_shapes(planted):
    panel, truth = planted
    rows = st.window_length_sweep(panel, "treated", "control", truth["switch_date"],
                                  str(panel["treated"].index[-1].date()),
                                  outcome="r_on", half_windows=(63, 126, 252))
    assert len(rows) == 3
    assert all(np.isfinite(r["did_t"]) for r in rows)


def test_placebo_switches_returns_a_distribution(planted):
    panel, truth = planted
    p = st.placebo_switches(panel, "treated", "control", truth["switch_date"],
                            str(panel["treated"].index[-1].date()),
                            outcome="r_on", step_days=126, exclude_days=252)
    assert p["n_placebo"] >= 3
    assert np.isfinite(p["placebo_median_abs_t"])
    assert 0.0 <= p["pct_below"] <= 1.0


# --------------------------------------------------------------------------- #
# The tradable arm — lag, costs, borrow
# --------------------------------------------------------------------------- #
def test_overlay_weight_is_lagged_one_day(planted):
    panel, _ = planted
    close_l = panel["treated"]["close"]
    close_s = panel["control"]["close"]
    bt = st.tom_spread_backtest(close_l, close_s, cost_bps=0.0, borrow_bps_ann=0.0)
    tom = st.turn_of_month_mask(close_l.index).astype(float)
    expect = tom.shift(1).reindex(bt.index)
    assert np.allclose(bt["w"].to_numpy(), expect.to_numpy())


def test_overlay_no_lookahead_future_prices_do_not_move_past_weights(planted):
    """Perturbing the tail of the price tape must not change earlier positions."""
    panel, _ = planted
    a = panel["treated"]["close"].copy()
    b = a.copy()
    b.iloc[1500:] *= 2.5
    w_a = st.tom_spread_backtest(a, panel["control"]["close"])["w"]
    w_b = st.tom_spread_backtest(b, panel["control"]["close"])["w"]
    n = 1400
    assert np.allclose(w_a.iloc[:n].to_numpy(), w_b.iloc[:n].to_numpy())


def test_overlay_is_flat_outside_the_turn_of_month(planted):
    panel, _ = planted
    bt = st.tom_spread_backtest(panel["treated"]["close"], panel["control"]["close"],
                                cost_bps=0.0, borrow_bps_ann=0.0)
    off = bt[bt["w"] == 0.0]
    assert len(off) > len(bt) * 0.5
    assert np.allclose(off["r_gross"].to_numpy(), 0.0)


def test_costs_and_borrow_monotonically_reduce_return(planted):
    panel, _ = planted
    means = []
    for c, b in ((0.0, 0.0), (3.0, 50.0), (10.0, 100.0)):
        bt = st.tom_spread_backtest(panel["treated"]["close"], panel["control"]["close"],
                                    cost_bps=c, borrow_bps_ann=b)
        means.append(bt["r_net"].mean())
    assert means[0] >= means[1] >= means[2]


def test_borrow_sweep_covers_the_grid(planted):
    panel, truth = planted
    rows = st.borrow_sweep(panel["treated"]["close"], panel["control"]["close"],
                           truth["switch_date"], str(panel["treated"].index[-1].date()),
                           cost_grid=(0.0, 3.0), borrow_grid=(0.0, 50.0))
    assert len(rows) == 4
    assert all(np.isfinite(r["net_post_bps"]) for r in rows)
    zero = [r for r in rows if r["cost_bps"] == 0.0 and r["borrow_bps_ann"] == 0.0][0]
    full = [r for r in rows if r["cost_bps"] == 3.0 and r["borrow_bps_ann"] == 50.0][0]
    assert zero["net_post_bps"] >= full["net_post_bps"]


def test_overlay_recovers_the_planted_month_end_bump():
    """A large planted turn-of-month bump must show up in the overlay's post-period."""
    panel, truth = data.synthetic_panel(signal_strength=1.0, seed=928, d_tom_bps=45.0)
    r = st.tom_pre_post(panel["treated"]["close"], panel["control"]["close"],
                        truth["switch_date"], str(panel["treated"].index[-1].date()),
                        cost_bps=0.0, borrow_bps_ann=0.0)
    assert r["gross_post_bps"] > r["gross_pre_bps"]
    assert r["did_gross_bps"] > 0


def test_excess_sharpe_pre_post_runs(planted):
    panel, truth = planted
    e = st.excess_sharpe_pre_post(panel["treated"]["close"], panel["cash"]["close"],
                                  truth["switch_date"], str(panel["treated"].index[-1].date()))
    assert np.isfinite(e["pre"]["sharpe"]) and np.isfinite(e["post"]["sharpe"])
    assert e["pre"]["n_days"] > 100 and e["post"]["n_days"] > 100


def test_summary_and_drawdown_agree(planted):
    panel, _ = planted
    r = panel["treated"]["close"].pct_change(fill_method=None).dropna()
    s = st.summary(r)
    assert abs(s["max_drawdown"] - st.max_drawdown(r)) < 1e-12
    assert s["n_days"] == len(r)

"""Estimator logic, replication invariants, and the study's spine — offline/synthetic.

The spine: on a tape whose financing rate is *planted* at benchmark + 75 bp the estimator
recovers 75 bp; on a null tape where the wrappers borrow at exactly the benchmark rate and
charge nothing it recovers ~0 bp. Because the two worlds share the identical index path
and tracking noise for a given seed, the *difference* of the two recovered spreads is the
planted effect exactly — noise cancels — so the recovery test is a hard equality, not a
tolerance. The tolerance tests then check the level, seed by seed.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lev_financing import data, strategy as st  # noqa: E402


def _rets(prices):
    return prices["index"].pct_change().dropna(), {
        L: prices[f"fund_{L}"].pct_change().dropna() for L in (2, 3)
    }


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_ols_hac_recovers_planted_line():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.01, 4000)
    y = -0.0001 + 2.5 * x + rng.normal(0, 0.0005, 4000)
    fit = st.ols_hac(y, x)
    assert abs(fit["beta"] - 2.5) < 0.02
    assert abs(fit["alpha"] + 0.0001) < 3e-5
    assert fit["alpha_t"] < -3 and fit["r2"] > 0.99


def test_ols_hac_handles_short_input():
    fit = st.ols_hac(np.zeros(4), np.zeros(4))
    assert np.isnan(fit["beta"])


def test_auto_lags_grows_with_sample():
    assert st.auto_lags(100) < st.auto_lags(10000)


def test_block_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(3)
    x = pd.Series(0.0002 + rng.normal(0, 0.01, 3000))
    ci = st.block_bootstrap_ci(x, n_boot=400, seed=945)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["n_obs"] == 3000


def test_block_bootstrap_ci_short_input_is_nan():
    ci = st.block_bootstrap_ci(pd.Series([0.1, 0.2]), n_boot=10)
    assert np.isnan(ci["point"])


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
def test_estimator_recovers_leverage(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    for L in (2, 3):
        e = st.implied_financing(funds[L], r_idx, L,
                                 expense_ratio=truth["expense_ratio_pct"],
                                 rate_pct=prices["rate"])
        assert abs(e["beta"] - L) < 0.02
        assert abs(e["beta_t_vs_L"]) < 3.0


def test_estimator_recovers_planted_spread(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    for L in (2, 3):
        d = st.synthetic_detect(prices, truth, leverage=L)
        # tolerance ~2.5 HAC standard errors of the intercept, scaled by (L-1)
        tol = 2.5 * d["alpha_se_ann_pct"] / (L - 1)
        assert abs(d["spread_error_pct"]) < max(tol, 0.1)
        assert d["alpha_t"] < -4.0        # the drag itself is unmistakable


def test_null_recovers_zero_spread(null_tape):
    prices, truth = null_tape
    for L in (2, 3):
        d = st.synthetic_detect(prices, truth, leverage=L)
        tol = 2.5 * d["alpha_se_ann_pct"] / (L - 1)
        assert abs(d["spread_over_rate_pct"]) < max(tol, 0.1)


def test_planted_minus_null_is_the_effect_exactly():
    """Shared seed => shared noise => the recovered difference IS the planted spread."""
    p1, t1 = data.synthetic_panel(signal_strength=1.0, seed=945)
    p0, t0 = data.synthetic_panel(signal_strength=0.0, seed=945)
    for L in (2, 3):
        s1 = st.synthetic_detect(p1, t1, leverage=L)["spread_over_rate_pct"]
        s0 = st.synthetic_detect(p0, t0, leverage=L)["spread_over_rate_pct"]
        assert s1 - s0 == pytest.approx(t1["planted_spread_pct"], abs=1e-6)


def test_null_across_seeds_is_centred():
    errs = []
    for s in range(6):
        p, t = data.synthetic_panel(signal_strength=0.0, seed=945 + s)
        errs.append(st.synthetic_detect(p, t, leverage=3)["spread_over_rate_pct"])
    assert abs(np.mean(errs)) < 0.25


def test_expense_ratio_shifts_financing_by_one_over_l_minus_one(planted):
    """A 10 bp error in the assumed fee moves implied financing by 10 bp / (L-1)."""
    prices, truth = planted
    r_idx, funds = _rets(prices)
    for L in (2, 3):
        rows = st.expense_ratio_sweep(funds[L], r_idx, prices["rate"], L,
                                      expense_ratios=(0.80, 0.90))
        delta = rows[0]["implied_financing_pct"] - rows[1]["implied_financing_pct"]
        assert delta == pytest.approx(0.10 / (L - 1), abs=1e-9)


def test_spread_t_is_the_claim_not_the_intercept_t(planted):
    """The spread's own HAC t must be reported, and it must be far weaker than alpha's.

    The intercept t only says "the drag is non-zero", which the expense ratio guarantees.
    Quoting it as evidence for a mark-up over the benchmark rate is the overstatement this
    test exists to prevent, so it pins the arithmetic: spread_t = spread / (alpha_se/(L-1)),
    and |spread_t| < |alpha_t| whenever a fee is being stripped out.
    """
    prices, truth = planted
    r_idx, funds = _rets(prices)
    for L in (2, 3):
        e = st.implied_financing(funds[L], r_idx, L, truth["expense_ratio_pct"],
                                 rate_pct=prices["rate"])
        assert e["spread_se_ann_pct"] == pytest.approx(
            e["drag_se_ann_pct"] / (L - 1), rel=1e-12)
        assert e["spread_t"] == pytest.approx(
            e["spread_over_rate_pct"] / e["spread_se_ann_pct"], rel=1e-12)
        assert e["all_in_spread_t"] == pytest.approx(
            e["all_in_spread_over_rate_pct"] / e["spread_se_ann_pct"], rel=1e-12)
        # stripping a real fee always leaves a weaker claim than "the drag is non-zero"
        assert abs(e["spread_t"]) < abs(e["alpha_t"])
        # and the all-in claim (no fee assumption) is the stronger of the two
        assert e["all_in_spread_t"] > e["spread_t"]


def test_null_tape_spread_t_is_inside_the_bar(null_tape):
    """On the null the spread t must NOT clear |t| = 2 — the bar has to be losable."""
    prices, truth = null_tape
    r_idx, funds = _rets(prices)
    for L in (2, 3):
        e = st.implied_financing(funds[L], r_idx, L, truth["expense_ratio_pct"],
                                 rate_pct=prices["rate"])
        assert abs(e["spread_t"]) < 2.0


def test_bench_fee_addback_increases_drag(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    a = st.implied_financing(funds[2], r_idx, 2, truth["expense_ratio_pct"], bench_fee=0.0)
    b = st.implied_financing(funds[2], r_idx, 2, truth["expense_ratio_pct"], bench_fee=0.10)
    assert b["drag_ann_pct"] > a["drag_ann_pct"]


def test_constrained_and_free_beta_agree_when_beta_is_l(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    for L in (2, 3):
        free = st.implied_financing(funds[L], r_idx, L, truth["expense_ratio_pct"])
        pinned = st.constrained_drag_series(funds[L], r_idx, L).mean() * 252 * 100
        assert abs(pinned - free["drag_ann_pct"]) < 0.35


def test_bootstrap_spread_ci_excludes_zero_on_planted(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    ci = st.bootstrap_spread(funds[3], r_idx, 3, truth["expense_ratio_pct"],
                             mean_rate_pct=truth["mean_rate_pct"], n_boot=400, seed=945)
    assert ci["ci_low"] > 0.0
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


# --------------------------------------------------------------------------- #
# Rolling estimate, pass-through, cuts
# --------------------------------------------------------------------------- #
def test_rolling_financing_tracks_the_rate_cycle(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    roll = st.rolling_financing(funds[3], r_idx, prices["rate"], 3,
                                truth["expense_ratio_pct"], window=252)
    assert len(roll) > 1000
    early = roll["implied_financing"].iloc[:200].mean()
    late = roll["implied_financing"].iloc[-200:].mean()
    assert late - early > 3.0            # the planted 0.15% -> 5.00% step is visible
    assert abs(roll["spread"].mean() - truth["planted_spread_pct"]) < 0.4


def test_rolling_is_causal(planted):
    """Perturbing the tail must not change any earlier rolling estimate."""
    prices, truth = planted
    r_idx, funds = _rets(prices)
    base = st.rolling_financing(funds[2], r_idx, prices["rate"], 2,
                                truth["expense_ratio_pct"], window=252)
    f2 = funds[2].copy()
    f2.iloc[-300:] = f2.iloc[-300:] - 0.01
    pert = st.rolling_financing(f2, r_idx, prices["rate"], 2,
                                truth["expense_ratio_pct"], window=252)
    n_safe = len(base) - 300 - 252
    assert n_safe > 100
    assert np.allclose(base["implied_financing"].iloc[:n_safe],
                       pert["implied_financing"].iloc[:n_safe])


def test_rolling_short_sample_returns_empty(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    roll = st.rolling_financing(funds[2].iloc[:50], r_idx.iloc[:50], prices["rate"], 2,
                                truth["expense_ratio_pct"], window=252)
    assert len(roll) == 0


def test_pass_through_slope_is_about_one(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    roll = st.rolling_financing(funds[3], r_idx, prices["rate"], 3,
                                truth["expense_ratio_pct"], window=252)
    pt = st.pass_through_regression(roll)
    assert abs(pt["slope"] - 1.0) < 0.25
    assert pt["corr"] > 0.9


def test_rate_regime_cut_splits_and_agrees(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    cut = st.rate_regime_cut(funds[3], r_idx, prices["rate"], 3,
                             truth["expense_ratio_pct"], threshold=1.0)
    assert cut["zirp"] is not None and cut["hiked"] is not None
    # the same planted spread must show in both rate worlds
    assert abs(cut["zirp"]["spread_over_rate_pct"]
               - cut["hiked"]["spread_over_rate_pct"]) < 0.6


def test_era_cut_returns_both_halves(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    split = str(prices.index[len(prices) // 2].date())
    eras = st.era_cut(funds[2], r_idx, prices["rate"], 2,
                      truth["expense_ratio_pct"], split=split)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["implied_financing_pct"])


# --------------------------------------------------------------------------- #
# The replication and the practical race
# --------------------------------------------------------------------------- #
def test_replication_is_lagged_not_look_ahead(planted):
    """Perturbing the rate on the final day cannot change any earlier replication return."""
    prices, truth = planted
    r_idx, _ = _rets(prices)
    rate = prices["rate"]
    base = st.replicate(r_idx, rate, 2, 1.0, cost_bps=1.0)
    bumped = rate.copy()
    bumped.iloc[-1] += 5.0
    pert = st.replicate(r_idx, bumped, 2, 1.0, cost_bps=1.0)
    assert np.allclose(base.iloc[:-1], pert.iloc[:-1])


def test_replication_costs_and_spread_only_hurt(planted):
    prices, truth = planted
    r_idx, _ = _rets(prices)
    means = [st.replicate(r_idx, prices["rate"], 2, m, cost_bps=c).mean()
             for m, c in ((0.0, 0.0), (0.0, 2.0), (2.0, 2.0), (5.0, 5.0))]
    assert means[0] > means[1] > means[2] > means[3]


def test_replication_beta_is_l(planted):
    prices, truth = planted
    r_idx, _ = _rets(prices)
    for L in (2, 3):
        rep = st.replicate(r_idx, prices["rate"], L, 1.0, cost_bps=0.0)
        fit = st.ols_hac(rep.to_numpy(), r_idx.reindex(rep.index).to_numpy())
        assert abs(fit["beta"] - L) < 0.15


def test_breakeven_spread_makes_the_race_a_tie(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    cash = prices["cash"].pct_change().dropna()
    for L in (2, 3):
        be = st.breakeven_margin_spread(funds[L], r_idx, prices["rate"], L, cost_bps=1.0)
        race = st.replication_race(funds[L], r_idx, cash, prices["rate"], L,
                                   margin_spread=be, cost_bps=1.0)
        assert abs(race["diff_ann_pct"]) < 0.02


def test_wrapper_wins_above_breakeven_and_loses_below(planted):
    prices, truth = planted
    r_idx, funds = _rets(prices)
    cash = prices["cash"].pct_change().dropna()
    be = st.breakeven_margin_spread(funds[2], r_idx, prices["rate"], 2, cost_bps=1.0)
    hi = st.replication_race(funds[2], r_idx, cash, prices["rate"], 2, be + 2.0, 1.0)
    lo = st.replication_race(funds[2], r_idx, cash, prices["rate"], 2, max(be - 2.0, 0.0), 1.0)
    assert hi["diff_ann_pct"] > 0 > lo["diff_ann_pct"]


def test_summary_shapes(planted):
    prices, _ = planted
    s = st.summary(prices["index"].pct_change().dropna())
    assert {"sharpe", "cagr", "vol_ann", "max_drawdown", "tstat"}.issubset(s)
    assert np.isfinite(s["sharpe"]) and s["max_drawdown"] <= 0.0
    assert np.isnan(st.summary(pd.Series([0.01]))["sharpe"])

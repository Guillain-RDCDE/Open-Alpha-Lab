"""Measurement logic, inference primitives, and the study's spine — offline/synthetic.

The spine: on a planted cross-section the scorecard recovers the true capture spreads and
the Henriksson-Merton convexity flags the convex fund positive and the concave ones
negative; on the null it stays quiet at roughly the nominal rate. A pure low-beta fund —
the thing income funds actually are — must score a capture spread of **zero**, not a
positive one. The traded arm has exactly one execution lag, costs only ever hurt, and the
short leg's borrow only ever hurts.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture_ratio import data, strategy as st  # noqa: E402


def _linear_pair(beta=0.6, n=240, seed=0):
    """A fund that is exactly ``beta`` times the benchmark plus noise (true spread = 0)."""
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n, freq="M").to_timestamp(how="end").normalize()
    b = pd.Series(rng.normal(0.007, 0.045, n), index=idx)
    f = beta * b + pd.Series(rng.normal(0, 0.005, n), index=idx)
    return f, b


# --------------------------------------------------------------------------- #
# Capture-ratio arithmetic
# --------------------------------------------------------------------------- #
def test_exact_scaling_gives_capture_equal_to_beta_and_zero_spread():
    """A fund that is exactly 0.6x the index has up = down = 0.6 and spread = 0."""
    idx = pd.period_range("2000-01", periods=60, freq="M").to_timestamp(how="end").normalize()
    b = pd.Series(np.tile([0.03, -0.02, 0.01, -0.04], 15), index=idx)
    cap = st.capture_ratios(0.6 * b, b)
    assert abs(cap["up"] - 0.6) < 1e-12
    assert abs(cap["down"] - 0.6) < 1e-12
    assert abs(cap["spread"]) < 1e-12


def test_low_beta_alone_is_not_convexity():
    """Low beta is a smaller position, not skill: the spread must stay ~0 for any beta."""
    for beta in (0.3, 0.6, 0.9):
        f, b = _linear_pair(beta=beta, seed=3)
        assert abs(st.capture_ratios(f, b)["spread"]) < 0.15
        assert abs(st.hm_regression(f, b)["convexity"]) < 0.10


def test_planted_kink_is_recovered_by_both_measures():
    idx = pd.period_range("2000-01", periods=300, freq="M").to_timestamp(how="end").normalize()
    rng = np.random.default_rng(7)
    b = pd.Series(rng.normal(0.007, 0.045, 300), index=idx)
    f = 0.8 * b.clip(lower=0) + 0.4 * b.clip(upper=0)      # up-beta 0.8, down-beta 0.4
    cap = st.capture_ratios(f, b)
    hm = st.hm_regression(f, b)
    assert abs(cap["up"] - 0.8) < 0.02 and abs(cap["down"] - 0.4) < 0.02
    assert abs(hm["convexity"] - 0.4) < 0.02
    assert hm["t_convexity"] > 5


def test_benchmark_sign_partitions_the_months():
    f, b = _linear_pair(seed=11)
    cap = st.capture_ratios(f, b)
    assert cap["n_up"] + cap["n_down"] == cap["n_months"] - int((b == 0).sum())
    assert cap["den_up"] > 0 > cap["den_down"]


def test_geometric_and_arithmetic_agree_in_sign_on_a_strong_kink():
    idx = pd.period_range("2000-01", periods=300, freq="M").to_timestamp(how="end").normalize()
    rng = np.random.default_rng(4)
    b = pd.Series(rng.normal(0.007, 0.04, 300), index=idx)
    f = 0.9 * b.clip(lower=0) + 0.3 * b.clip(upper=0)
    a = st.capture_ratios(f, b)["spread"]
    g = st.geometric_capture_ratios(f, b)["spread"]
    assert a > 0 and g > 0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.004 + rng.normal(0, 0.01, 4000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 4000))) < 3


def test_hac_ols_matches_ols_point_estimates():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 500)
    y = 0.3 + 1.7 * x + rng.normal(0, 0.1, 500)
    beta, cov = st.hac_ols(y, np.column_stack([np.ones(500), x]))
    assert abs(beta[0] - 0.3) < 0.05 and abs(beta[1] - 1.7) < 0.05
    assert cov.shape == (2, 2) and (np.diag(cov) > 0).all()


def test_bonferroni_bar_is_stricter_than_nominal():
    assert abs(st.bonferroni_z(1) - 1.96) < 0.01
    assert st.bonferroni_z(14) > 2.8
    assert st.bonferroni_z(50) > st.bonferroni_z(14)


def test_spearman_sanity():
    rho, p = st.spearman([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7])
    assert rho > 0.99 and p < 0.05
    rho2, _ = st.spearman([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1])
    assert rho2 < -0.99


def test_bootstrap_ci_brackets_the_point_spread():
    f, b = _linear_pair(seed=5)
    boot = st.bootstrap_capture_spread(f, b, n_boot=600, seed=948)
    assert boot["ci_low"] <= boot["spread"] <= boot["ci_high"]
    assert 0.0 <= boot["frac_positive"] <= 1.0
    assert boot["n_boot"] > 100


def test_bootstrap_is_reproducible():
    f, b = _linear_pair(seed=6)
    a1 = st.bootstrap_capture_spread(f, b, n_boot=400, seed=948)
    a2 = st.bootstrap_capture_spread(f, b, n_boot=400, seed=948)
    a3 = st.bootstrap_capture_spread(f, b, n_boot=400, seed=949)
    assert a1["ci_low"] == a2["ci_low"] and a1["ci_high"] == a2["ci_high"]
    assert a1["ci_low"] != a3["ci_low"]


def test_summary_is_finite_and_sharpe_scales():
    f, _ = _linear_pair(seed=8)
    s = st.summary(f)
    for k in ("sharpe", "vol_ann", "cagr", "max_drawdown", "tstat"):
        assert np.isfinite(s[k])
    assert s["max_drawdown"] <= 0.0


# --------------------------------------------------------------------------- #
# The traded arm — the one execution lag, costs and borrow
# --------------------------------------------------------------------------- #
def test_traded_spread_has_exactly_one_lag(planted_panel):
    """Perturbing only the future must not change any earlier traded return."""
    rets, _ = planted_panel
    f, b, c = rets["CONVEX"], rets["BENCH"], rets["CASH"]
    base = st.traded_beta_neutral(f, b, c, cost_bps=0.0, borrow_bps_ann=0.0)
    f2 = f.copy()
    f2.iloc[200:] *= 5.0
    pert = st.traded_beta_neutral(f2, b, c, cost_bps=0.0, borrow_bps_ann=0.0)
    common = base.index.intersection(pert.index)
    # beta_hat is estimated through month t and applied at t+1, so month 200 is the first
    # month that can move (its own return changed); months < 200 must be untouched.
    early = common[common < rets.index[200]]
    assert len(early) > 50
    assert np.allclose(base.loc[early].to_numpy(), pert.loc[early].to_numpy())


def test_traded_spread_costs_and_borrow_only_ever_hurt(planted_panel):
    rets, _ = planted_panel
    f, b, c = rets["CONCAVE_A"], rets["BENCH"], rets["CASH"]
    gross = st.traded_beta_neutral(f, b, c, cost_bps=0.0, borrow_bps_ann=0.0).mean()
    costed = st.traded_beta_neutral(f, b, c, cost_bps=5.0, borrow_bps_ann=0.0).mean()
    borrowed = st.traded_beta_neutral(f, b, c, cost_bps=5.0, borrow_bps_ann=100.0).mean()
    heavy = st.traded_beta_neutral(f, b, c, cost_bps=25.0, borrow_bps_ann=200.0).mean()
    assert gross > costed > borrowed > heavy


def test_traded_sweep_is_monotone_in_both_frictions(planted_panel):
    rets, _ = planted_panel
    rows = st.traded_sweep(rets["FLAT_A"], rets["BENCH"], rets["CASH"])
    assert len(rows) > 1
    by = {(r["cost_bps"], r["borrow_bps"]): r["mean_bps"] for r in rows}
    assert by[(0.0, 0.0)] >= by[(5.0, 0.0)] >= by[(25.0, 0.0)]
    assert by[(0.0, 0.0)] >= by[(0.0, 100.0)] >= by[(0.0, 200.0)]


def test_traded_spread_is_roughly_beta_neutral(planted_panel):
    rets, _ = planted_panel
    r = st.traded_beta_neutral(rets["FLAT_C"], rets["BENCH"], rets["CASH"],
                               cost_bps=0.0, borrow_bps_ann=0.0)
    eb = (rets["BENCH"] - rets["CASH"]).reindex(r.index)
    assert abs(float(np.corrcoef(r.to_numpy(), eb.to_numpy())[0, 1])) < 0.35


# --------------------------------------------------------------------------- #
# Scorecard, era cut, rank persistence
# --------------------------------------------------------------------------- #
def test_scorecard_shape_and_columns(planted_panel):
    rets, truth = planted_panel
    sc = st.scorecard(rets, truth["bench_map"], cash_col="CASH", n_boot=200)
    assert len(sc) == len(truth["fund_names"])
    for col in ("up", "down", "spread", "convexity", "t_convexity", "beta",
                "alpha_bps", "sharpe_gap", "ci_low", "ci_high"):
        assert col in sc.columns and np.isfinite(sc[col]).all()


def test_force_bench_overrides_the_assumption(planted_panel):
    rets, truth = planted_panel
    a = st.scorecard(rets, truth["bench_map"], cash_col="CASH", n_boot=1)
    b = st.scorecard(rets, truth["bench_map"], cash_col="CASH", n_boot=1,
                     force_bench="BENCH")
    assert (a["bench"] == "BENCH").all() and (b["bench"] == "BENCH").all()
    assert np.allclose(a["spread"].to_numpy(), b["spread"].to_numpy())


def test_era_cut_and_rank_persistence_run(planted_panel):
    rets, truth = planted_panel
    split = str(rets.index[len(rets) // 2].date())
    era = st.era_spreads(rets, truth["bench_map"], split=split)
    assert len(era) == len(truth["fund_names"])
    assert {"spread_early", "spread_late"}.issubset(era.columns)
    rp = st.rank_persistence(era)
    assert rp["n"] == len(era) and np.isfinite(rp["rho"])
    # planted structure IS a fund property, so it must persist across the two halves
    assert rp["rho"] > 0.5


def test_era_cut_drops_short_histories(planted_panel):
    rets, truth = planted_panel
    late = str(rets.index[-12].date())
    era = st.era_spreads(rets, truth["bench_map"], split=late, min_months=24)
    assert len(era) == 0


# --------------------------------------------------------------------------- #
# The study's spine — the machinery is unbiased
# --------------------------------------------------------------------------- #
def test_planted_cross_section_is_recovered(planted_panel):
    rets, truth = planted_panel
    d = st.synthetic_detect(rets, truth, n_boot=200)
    assert d["corr_measured_true"] > 0.85
    assert d["n_hits_positive"] >= 1       # the CONVEX fund is flagged convex
    assert d["n_hits_negative"] >= 2       # the CONCAVE_* funds are flagged concave


def test_planted_convex_fund_scores_highest(planted_panel):
    rets, truth = planted_panel
    sc = st.scorecard(rets, truth["bench_map"], cash_col="CASH", n_boot=1)
    assert sc["convexity"].idxmax() == "CONVEX"
    assert sc["convexity"].idxmin().startswith("CONCAVE")


def test_null_panel_fires_at_about_the_nominal_rate(null_panel):
    rets, truth = null_panel
    d = st.synthetic_detect(rets, truth, n_boot=200)
    assert d["n_hits_positive"] + d["n_hits_negative"] <= 2


def test_null_convexity_is_unbiased_across_seeds():
    nb = st.null_spread_distribution(
        data.synthetic_panel(signal_strength=0.0, seed=948 + s) for s in range(6))
    assert nb["n_obs"] == 6 * len(data.PLANTED_FUNDS)
    assert abs(nb["convexity_mean"]) < 0.05           # the regression twin is unbiased


def test_raw_capture_spread_is_upward_biased_under_the_null():
    """The study's methodological punchline: the industry ratio invents convexity.

    Under a truth of exactly zero the arithmetic capture spread has a *positive* mean —
    it is a difference of ratios whose down-leg denominator is a small negative number.
    The HM convexity coefficient does not carry that bias. If this ever stops being true
    the study's central caveat has to be rewritten.
    """
    nb = st.null_spread_distribution(
        data.synthetic_panel(signal_strength=0.0, seed=948 + s) for s in range(6))
    assert nb["spread_mean"] > nb["convexity_mean"] + 0.03
    assert nb["spread_frac_positive"] > 0.6


# --------------------------------------------------------------------------- #
# The fund-matched null — the bias measured on a fund's OWN sample
# --------------------------------------------------------------------------- #
def test_matched_null_detects_a_genuinely_convex_fund(planted_panel):
    """Beta-only null: the planted convex fund clears it, the concave ones sit at the tail."""
    rets, truth = planted_panel
    conv = st.matched_null_spread(rets["CONVEX"], rets["BENCH"], rets["CASH"], n_sims=400)
    assert conv["p_value"] < 0.05
    assert conv["excess_spread"] > 0.15
    for name in ("CONCAVE_A", "CONCAVE_C"):
        c = st.matched_null_spread(rets[name], rets["BENCH"], rets["CASH"], n_sims=400)
        assert c["p_value"] > 0.90 and c["excess_spread"] < 0.0
    assert truth["funds"]["CONVEX"]["true_spread"] > 0


def test_matched_null_bias_is_smaller_than_the_unconditional_one(null_panel):
    """Conditioning on the realised benchmark path removes most of the ratio bias.

    This is why the generic synthetic bias must NOT be subtracted from a real-tape spread:
    the real spread is computed on ONE fixed benchmark path, and on a fixed path the
    estimator's bias is a fraction of its unconditional size.
    """
    rets, truth = null_panel
    mn = st.matched_null_panel(rets, truth["bench_map"], cash_col="CASH", n_sims=300)
    nb = st.null_spread_distribution(
        data.synthetic_panel(signal_strength=0.0, seed=948 + s) for s in range(6))
    assert float(mn["null_mean"].abs().median()) < nb["spread_mean"]


def test_alpha_alone_manufactures_a_positive_capture_spread():
    """A perfectly LINEAR fund with a positive alpha reads as 'convex' on the scorecard."""
    f, b = _linear_pair(beta=0.6, n=300, seed=11)
    f_alpha = f + 0.004                       # +40 bps/month, no convexity whatsoever
    assert abs(st.capture_ratios(f, b)["spread"]) < 0.08
    assert st.capture_ratios(f_alpha, b)["spread"] > 0.15
    # the regression twin is not fooled: convexity stays ~0 either way
    assert abs(st.hm_regression(f_alpha, b)["convexity"]) < 0.10


def test_alpha_beta_twin_reproduces_the_observed_spread(planted_panel):
    """include_alpha=True is a decomposition, not a test: it re-reads the same number.

    An OLS line fitted to a convex fund answers with a large positive intercept, and that
    intercept reproduces the capture spread. The spread therefore carries no information
    beyond (alpha, beta) — which is exactly why the study leans on Henriksson-Merton.
    """
    rets, _ = planted_panel
    for name in ("CONVEX", "FLAT_A", "CONCAVE_C"):
        m = st.matched_null_spread(rets[name], rets["BENCH"], rets["CASH"],
                                   n_sims=400, include_alpha=True)
        assert abs(m["excess_spread"]) < 0.05


def test_matched_null_is_reproducible_and_short_samples_return_nan():
    f, b = _linear_pair(beta=0.7, n=120, seed=5)
    cash = pd.Series(0.0015, index=b.index)
    a = st.matched_null_spread(f, b, cash, n_sims=200, seed=3)
    c = st.matched_null_spread(f, b, cash, n_sims=200, seed=3)
    assert a["p_value"] == c["p_value"] and a["null_mean"] == c["null_mean"]
    short = st.matched_null_spread(f.iloc[:18], b.iloc[:18], cash.iloc[:18], n_sims=50)
    assert short["n_sims"] == 0 and np.isnan(short["p_value"])


@pytest.mark.parametrize("seed", [948, 951, 954])
def test_scorecard_is_deterministic_given_a_seed(seed):
    rets, truth = data.synthetic_panel(signal_strength=1.0, seed=seed)
    a = st.scorecard(rets, truth["bench_map"], cash_col="CASH", n_boot=200, seed=1)
    b = st.scorecard(rets, truth["bench_map"], cash_col="CASH", n_boot=200, seed=1)
    assert np.allclose(a["ci_low"].to_numpy(), b["ci_low"].to_numpy())

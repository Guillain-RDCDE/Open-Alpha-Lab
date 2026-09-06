"""Strategy tests for Study 1012 — how much of alpha is the benchmark."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from benchmark import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The regression
# --------------------------------------------------------------------------- #
def test_ols_recovers_a_planted_intercept_and_slope():
    rng = np.random.default_rng(1012)
    x = rng.normal(0, 0.01, 3000)
    y = 0.0002 + 1.5 * x + rng.normal(0, 0.003, 3000)
    d = st.ols_with_hac(y, x.reshape(-1, 1))
    assert d["alpha"] == pytest.approx(0.0002 * 252, rel=0.25)
    assert d["betas"][0] == pytest.approx(1.5, abs=0.05)


def test_a_perfect_fit_has_r2_of_one():
    x = np.linspace(-0.01, 0.01, 500)
    d = st.ols_with_hac(2 * x, x.reshape(-1, 1))
    assert d["r2"] == pytest.approx(1.0)
    assert d["alpha"] == pytest.approx(0.0, abs=1e-9)


def test_hac_standard_errors_exceed_naive_ones_under_autocorrelation():
    """Which is why HAC is the default here, not an option."""
    rng = np.random.default_rng(0)
    n = 3000
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.6 * e[t - 1] + rng.normal(0, 0.004)
    x = rng.normal(0, 0.01, n)
    y = 1.0 * x + e
    a = st.ols_with_hac(y, x.reshape(-1, 1), lags=0)
    b = st.ols_with_hac(y, x.reshape(-1, 1), lags=20)
    assert b["alpha_se"] > a["alpha_se"]


def test_ols_declines_on_too_little_data():
    assert st.ols_with_hac(np.arange(10.0), np.arange(10.0).reshape(-1, 1)) == {}


def test_single_factor_alpha_reports_beta_and_tracking_error():
    px = data.load_prices()
    R = px.pct_change()
    d = st.single_factor_alpha(R["XLK"], R[data.MARKET])
    assert 0.5 < d["beta"] < 2.0
    assert d["tracking_error"] > 0
    assert np.isfinite(d["information_ratio"])


def test_a_fund_benchmarked_against_itself_has_zero_alpha():
    px = data.load_prices()
    R = px.pct_change()
    d = st.single_factor_alpha(R[data.MARKET], R[data.MARKET])
    assert abs(d["alpha"]) < 1e-9
    assert d["beta"] == pytest.approx(1.0)
    assert d["r2"] == pytest.approx(1.0)


def test_multi_factor_alpha_names_its_loadings():
    px = data.load_prices()
    R = px.pct_change()
    d = st.multi_factor_alpha(R["XLK"], R[[data.MARKET, data.SMALL]])
    assert set(d["loadings"]) == {data.MARKET, data.SMALL}


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
def test_the_grid_covers_the_panel():
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    g = st.alpha_grid(funds, benches, rf)
    assert len(g) > 30
    assert set(g.columns) >= {"fund", "benchmark", "alpha", "alpha_t", "r2"}


def test_a_fund_is_never_benchmarked_against_itself():
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    g = st.alpha_grid(funds, benches, rf)
    assert (g["fund"] != g["benchmark"]).all()


def test_alpha_moves_a_lot_across_benchmarks():
    """The headline."""
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    r = st.alpha_range(st.alpha_grid(funds, benches, rf))
    assert r["alpha_spread"].median() > 0.02


def test_the_spread_exceeds_the_standard_error_for_most_funds():
    """Specification uncertainty dominates sampling uncertainty. Nobody reports the first."""
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    r = st.alpha_range(st.alpha_grid(funds, benches, rf))
    assert (r["spread_over_se"] > 1.0).mean() > 0.5


def test_some_funds_flip_the_sign_of_their_alpha():
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    r = st.alpha_range(st.alpha_grid(funds, benches, rf))
    assert r["sign_flips"].any()


def test_alpha_range_declines_on_a_thin_grid():
    assert st.alpha_range(pd.DataFrame()).empty
    thin = pd.DataFrame({"fund": ["A"], "alpha": [0.01], "alpha_se": [0.01],
                         "alpha_t": [1.0], "r2": [0.5], "benchmark": ["B"]})
    assert st.alpha_range(thin).empty


# --------------------------------------------------------------------------- #
# Choosing a benchmark
# --------------------------------------------------------------------------- #
def test_best_fit_reports_how_many_candidates_were_searched():
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    b = st.best_fit_benchmark(funds["XLK"], benches, rf)
    assert b["n_candidates"] >= 5
    assert b["best_r2"] > 0.3


def test_the_best_fitting_benchmark_is_not_the_most_flattering_one():
    """Which is precisely why 'we used the best-fitting index' is not a defence."""
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    gains = []
    for f in ("XLK", "XLU", "XLE", "VNQ"):
        if f not in funds.columns:
            continue
        b = st.best_fit_benchmark(funds[f], benches, rf)
        if b:
            gains.append(b["cherry_picking_gain"])
    assert max(gains) > 0.0


def test_best_fit_declines_on_an_empty_candidate_set():
    px = data.load_prices()
    funds, _, rf = _panel(px)
    assert st.best_fit_benchmark(funds["XLK"], pd.DataFrame(), rf) == {}


def test_the_ladder_shrinks_alpha_as_factors_are_added():
    """Each factor absorbs something previously called skill."""
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    L = st.specification_ladder(funds["XLK"], benches, rf)
    assert L["r2"].is_monotonic_increasing
    assert abs(L["alpha"].iloc[-1]) <= abs(L["alpha"].iloc[0]) + 0.05


def test_the_ladder_always_improves_r2():
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    for f in ("XLU", "VNQ"):
        if f not in funds.columns:
            continue
        L = st.specification_ladder(funds[f], benches, rf)
        assert L["r2"].is_monotonic_increasing


def test_the_ladder_handles_missing_benchmarks():
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    L = st.specification_ladder(funds["XLK"], benches[[data.MARKET]], rf)
    assert len(L) >= 1


# --------------------------------------------------------------------------- #
# Can the data choose?
# --------------------------------------------------------------------------- #
def test_the_bootstrap_reports_a_win_share_per_benchmark():
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    c = st.can_the_data_choose(funds["XLK"], benches, rf, n_boot=40)
    assert c
    assert abs(sum(c["win_share"].values()) - 1.0) < 1e-9
    assert 0 <= c["modal_share"] <= 1


def test_a_fund_that_IS_a_benchmark_is_identified_decisively():
    """The calibration: when the answer is obvious the method must say so."""
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    c = st.can_the_data_choose(funds["IWM"], benches, rf, n_boot=40)
    assert c["modal_benchmark"] == "IWM" or c["modal_share"] > 0.8


def test_can_the_data_choose_declines_on_a_short_series():
    px = data.load_prices()
    funds, benches, rf = _panel(px)
    assert st.can_the_data_choose(funds["XLK"].iloc[:100], benches, rf) == {}


def test_the_encompassing_test_detects_a_redundant_benchmark():
    px = data.load_prices()
    R = px.pct_change()
    rf = R[data.BILLS] if data.BILLS in R.columns else None
    e = st.encompassing_test(R["IWM"], R["IWM"], R[data.MARKET], rf)
    assert e["a_encompasses_b"] or e["beta_a"] > e["beta_b"]


def test_the_encompassing_test_can_say_both_are_needed():
    px = data.load_prices()
    R = px.pct_change()
    rf = R[data.BILLS] if data.BILLS in R.columns else None
    e = st.encompassing_test(R["QQQ"], R[data.MARKET], R[data.SMALL], rf)
    assert e["both_needed"] or e["a_encompasses_b"] or e["b_encompasses_a"]


def test_the_encompassing_test_declines_on_too_little_data():
    px = data.load_prices()
    R = px.pct_change().iloc[:50]
    assert st.encompassing_test(R["XLK"], R[data.MARKET], R[data.SMALL]) == {}


# --------------------------------------------------------------------------- #
# The control, where the truth is planted
# --------------------------------------------------------------------------- #
def test_the_correct_benchmark_recovers_a_planted_alpha():
    for true_a in (0.0, 0.03):
        w = st.synthetic_fund(n_days=6000, true_alpha=true_a,
                              loadings={"F1": 1.0, "F2": 0.5})
        d = st.multi_factor_alpha(w["fund"], w["factors"])
        assert d["alpha"] == pytest.approx(true_a, abs=0.012)


def test_the_correct_benchmark_recovers_the_planted_loadings():
    w = st.synthetic_fund(n_days=6000, loadings={"F1": 1.2, "F2": 0.4})
    d = st.multi_factor_alpha(w["fund"], w["factors"])
    assert d["loadings"]["F1"] == pytest.approx(1.2, abs=0.06)
    assert d["loadings"]["F2"] == pytest.approx(0.4, abs=0.06)


def test_a_wrong_benchmark_manufactures_alpha_from_nothing():
    """The study's central claim, on data where the true alpha is exactly zero."""
    d = st.mis_specification_damage(true_alpha=0.0, factor_corr_grid=(0.3,),
                                    n_days=4000, n_reps=10)
    assert abs(d.loc[0.3, "alpha_correct"]) < 0.02
    assert d.loc[0.3, "error_only_f1"] > 0.02
    assert abs(d.loc[0.3, "alpha_correct"]) < abs(d.loc[0.3, "error_only_f1"]) / 2


def test_the_bias_requires_the_omitted_factor_to_have_EARNED_something():
    """The mechanism, and the one this study initially got wrong.

    Omitting a **zero-mean** factor does not bias the intercept at all — omitted-variable bias
    lands on the slope and reaches alpha only through the omitted factor's mean. A benchmark
    that misses a factor manufactures alpha exactly when that factor earned a premium, which
    in practice size, value and momentum all did.
    """
    def false_alpha(premium):
        out = []
        for k in range(8):
            w = st.synthetic_fund(4000, 0.0, {"F1": 1.0, "F2": 0.5},
                                  factor_means={"F1": 0.07, "F2": premium},
                                  factor_corr=0.3, seed=1012 + k)
            a = st.single_factor_alpha(w["fund"], w["factors"]["F1"])
            if a:
                out.append(a["alpha"])
        return float(np.mean(out))
    assert abs(false_alpha(0.0)) < 0.02
    assert false_alpha(0.12) > 0.03


def test_the_damage_is_larger_when_the_omitted_factor_is_less_similar():
    """Correlated benchmarks are NOT interchangeable — the damage is worst when they differ."""
    d = st.mis_specification_damage(true_alpha=0.0,
                                    factor_corr_grid=(0.1, 0.9),
                                    n_days=4000, n_reps=6)
    assert abs(d.loc[0.1, "error_only_f1"]) > abs(d.loc[0.9, "error_only_f1"])


def test_a_wrong_benchmark_produces_SIGNIFICANT_false_alpha():
    """A bias is a nuisance. A significant bias is a machine for publishing findings.

    Measured on a long sample, because the bias is constant while the standard error shrinks:
    a false alpha becomes MORE significant the more data you have, which inverts the usual
    comfort taken from a long track record.
    """
    s = st.false_alpha_significance(true_alpha=0.0, n_days=8000, n_reps=60)
    assert s["mean_false_alpha"] > 0.01
    assert s["share_significant_wrong_benchmark"] > \
        s["share_significant_right_benchmark"]


def test_the_correct_benchmark_keeps_its_false_positive_rate_near_nominal():
    s = st.false_alpha_significance(true_alpha=0.0, n_days=8000, n_reps=60)
    assert s["share_significant_right_benchmark"] < 0.20


def test_the_hac_estimator_has_the_right_size_on_iid_data():
    """The control for the control: if the test over-rejected, nothing above would mean much."""
    rng = np.random.default_rng(0)
    sig = 0
    for _ in range(200):
        x = rng.normal(0, 0.01, 3000)
        y = 1.0 * x + rng.normal(0, 0.005, 3000)
        if abs(st.ols_with_hac(y, x.reshape(-1, 1), lags=5)["alpha_t"]) > 2:
            sig += 1
    assert sig / 200 < 0.12


def test_the_synthetic_factors_have_the_correlation_they_claim():
    w = st.synthetic_fund(n_days=40000, factor_corr=0.7,
                          loadings={"F1": 1.0, "F2": 0.5})
    assert w["factors"].corr().iloc[0, 1] == pytest.approx(0.7, abs=0.03)


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def _panel(px):
    R = px.pct_change()
    fcols = [c for c in data.FUNDS if c in R.columns
             and R[c].dropna().shape[0] > 1500]
    bcols = [c for c in data.BENCHMARKS if c in R.columns
             and R[c].dropna().shape[0] > 1500]
    rf = R[data.BILLS] if data.BILLS in R.columns else None
    return R[fcols], R[bcols], rf


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_funds": 17, "n_benchmarks": 9, "median_spread": 0.061,
         "median_spread_over_se": 3.4, "share_sign_flip": 0.71,
         "share_both_significant": 0.29, "worst_fund": "XLE",
         "worst_spread": 0.118, "ladder_first": 0.041, "ladder_last": 0.008,
         "share_decisive": 0.24, "false_alpha": 0.021, "false_sig_rate": 0.56,
         "true_sig_rate": 0.08, "cherry_pick_gain": 0.037}
    h.update(over)
    return h


def test_verdict_signal_compares_the_spread_against_the_standard_error():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(median_spread_over_se=1.2))["signal"] == "Weak"
    assert st.verdict(_headline(median_spread_over_se=0.3))["signal"] == "None"


def test_verdict_tradability_keys_off_whether_the_data_can_decide():
    assert st.verdict(_headline())["trad"] == "Mirage"
    assert st.verdict(_headline(share_decisive=0.5))["trad"] == "Partial"
    assert st.verdict(_headline(share_decisive=0.9))["trad"] == "Useful"


def test_verdict_prose_names_the_false_positive_rate():
    v = st.verdict(_headline())
    assert "no error bar at all" in v["signal_why"]
    assert "true alpha of zero" in v["trad_why"]
    assert "Ask for the **grid**" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

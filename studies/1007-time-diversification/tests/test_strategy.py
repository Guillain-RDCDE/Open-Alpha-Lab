"""Strategy tests for Study 1007 — three definitions of risk, and the horizon."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from timediv import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def test_horizon_windows_compound_correctly():
    idx = pd.bdate_range("2010-01-01", periods=1000)
    r = pd.Series(np.full(1000, 0.001), index=idx)
    w = st.horizon_windows(r, 1.0, step=252)
    assert np.allclose(w, 252 * np.log1p(0.001))


def test_horizon_windows_are_empty_when_the_horizon_exceeds_the_data():
    r = pd.Series(np.full(300, 0.001), index=pd.bdate_range("2020-01-01", periods=300))
    assert st.horizon_windows(r, 5.0).size == 0


def test_a_longer_horizon_yields_fewer_windows():
    r = st.synthetic_iid(n_days=6000)
    assert len(st.horizon_windows(r, 10)) < len(st.horizon_windows(r, 2))


def test_the_effective_sample_is_reported_honestly():
    """Thirty-three years contains three independent thirty-year periods, not hundreds."""
    r = st.synthetic_iid(n_days=8400)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    m = st.horizon_metrics(r, c, years_grid=(1, 30))
    assert m.loc[30, "n_windows"] > 30            # dozens of overlapping windows...
    assert m.loc[30, "effective_n"] < 2.0         # ...worth barely one independent one
    assert m.loc[1, "effective_n"] > 30


# --------------------------------------------------------------------------- #
# The three risks
# --------------------------------------------------------------------------- #
def test_annualised_dispersion_falls_with_horizon():
    r = st.synthetic_iid(n_days=8400)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    m = st.horizon_metrics(r, c, years_grid=(1, 3, 10, 20))
    assert m["annualised_sd"].is_monotonic_decreasing


def test_terminal_wealth_dispersion_rises_with_horizon():
    """Same windows, opposite direction. This is the entire argument.

    Measured on a LONG series. On a 33-year tape the same statistic peaks and then falls —
    see `test_short_samples_make_log_dispersion_appear_to_shrink`, which is an artefact rather
    than a fact about markets.
    """
    r = st.synthetic_iid(n_days=60000)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    m = st.horizon_metrics(r, c, years_grid=(1, 3, 10, 20))
    assert m["log_sd"].is_monotonic_increasing
    assert m["terminal_sd"].is_monotonic_increasing


def test_short_samples_make_log_dispersion_appear_to_shrink():
    """A statistic that MUST rise appears to fall, on a sample the length of the real one.

    Log dispersion of terminal wealth grows like sqrt(T) under i.i.d. returns — there is no
    parameter choice that makes it fall. On 33 years of i.i.d. data it peaks around a decade
    and declines, because the long-horizon windows are worth roughly one independent
    observation each and their sample standard deviation is badly downward-biased.
    """
    r = st.synthetic_iid(n_days=8400)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    m = st.horizon_metrics(r, c, years_grid=(1, 3, 5, 10, 15, 20, 25, 30))
    assert not m["log_sd"].is_monotonic_increasing
    assert m["log_sd"].idxmax() < 20
    assert m.loc[30, "log_sd"] < m.loc[10, "log_sd"]


def test_both_directions_hold_on_the_real_tape_too():
    px = data.load_prices()
    r, c = _pair(px)
    m = st.horizon_metrics(r, c, years_grid=(1, 5, 10, 20))
    assert m["annualised_sd"].iloc[-1] < m["annualised_sd"].iloc[0]
    assert m["log_sd"].iloc[-1] > m["log_sd"].iloc[0]


def test_shortfall_probability_falls_with_horizon():
    px = data.load_prices()
    r, c = _pair(px)
    m = st.horizon_metrics(r, c, years_grid=(1, 5, 10, 20))
    assert m["shortfall_vs_cash"].iloc[-1] < m["shortfall_vs_cash"].iloc[0]


def test_the_worst_outcome_gets_worse_even_as_shortfall_falls():
    """The trade the glide-path argument leaves out: less likely, more costly."""
    px = data.load_prices()
    r, c = _pair(px)
    m = st.horizon_metrics(r, c, years_grid=(1, 10, 20))
    assert m.loc[20, "shortfall_vs_cash"] < m.loc[1, "shortfall_vs_cash"]
    assert m.loc[20, "log_sd"] > m.loc[1, "log_sd"]


def test_horizon_metrics_skips_horizons_with_too_little_data():
    r = st.synthetic_iid(n_days=1000)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    m = st.horizon_metrics(r, c, years_grid=(1, 50))
    assert 50 not in m.index


# --------------------------------------------------------------------------- #
# Arithmetic versus economics
# --------------------------------------------------------------------------- #
def test_iid_returns_converge_at_one_over_root_t_GIVEN_ENOUGH_DATA():
    """The control. Nothing here can mean-revert, and it still narrows — at -0.5 eventually."""
    slopes = []
    for k in range(4):
        r = st.synthetic_iid(n_days=120000, seed=1007 + k)
        c = pd.Series(np.zeros(len(r)), index=r.index)
        slopes.append(st.excess_convergence(
            st.horizon_metrics(r, c, (1, 2, 3, 5, 7, 10, 15)))["slope"])
    assert np.mean(slopes) == pytest.approx(-0.5, abs=0.08)


def test_a_realistic_sample_measures_far_STEEPER_than_the_truth():
    """The study's central methodological finding, on data that cannot mean-revert.

    Thirty-three years of i.i.d. returns yields a convergence slope near -0.7, not -0.5. Anyone
    measuring -0.7 on real equities and concluding "mean reversion" is reading the length of
    their sample, not a property of the market.
    """
    slopes = []
    for k in range(4):
        r = st.synthetic_iid(n_days=8400, seed=1007 + k)
        c = pd.Series(np.zeros(len(r)), index=r.index)
        e = st.excess_convergence(st.horizon_metrics(r, c, (1, 2, 3, 5, 7, 10, 15)))
        slopes.append(e["slope"])
    assert np.mean(slopes) < -0.60


def test_the_bias_shrinks_as_the_sample_grows():
    b = st.small_sample_bias(n_days_grid=(8400, 30000, 100000), n_reps=2)
    assert b["mean_slope"].is_monotonic_increasing
    assert abs(b["bias_vs_half"].iloc[-1]) < abs(b["bias_vs_half"].iloc[0])
    assert abs(b["bias_vs_half"].iloc[-1]) < 0.06


def test_the_sqrt_t_benchmark_is_anchored_at_one_year():
    r = st.synthetic_iid(n_days=20000)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    b = st.sqrt_t_benchmark(st.horizon_metrics(r, c, years_grid=(1, 4, 9)))
    assert b.loc[1, "ratio_annualised"] == pytest.approx(1.0)
    assert b.loc[4, "iid_annualised_sd"] == pytest.approx(
        b.loc[1, "iid_annualised_sd"] / 2)


def test_mean_reverting_returns_converge_faster_than_root_t():
    """And the machinery detects it — otherwise a null result would prove nothing."""
    r = st.synthetic_mean_reverting(n_days=40000, phi=-0.05)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    e = st.excess_convergence(st.horizon_metrics(r, c,
                                                 years_grid=(1, 2, 3, 5, 7, 10, 15)))
    assert e["slope"] < -0.5


def test_excess_convergence_declines_on_too_few_horizons():
    r = st.synthetic_iid(n_days=6000)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    assert st.excess_convergence(st.horizon_metrics(r, c, years_grid=(1, 2))) == {}


# --------------------------------------------------------------------------- #
# Variance ratios
# --------------------------------------------------------------------------- #
def test_variance_ratio_is_one_for_iid_returns():
    r = st.synthetic_iid(n_days=40000)
    v = st.variance_ratio(r, 252)
    assert v["vr"] == pytest.approx(1.0, abs=0.12)
    assert not v["mean_reverting"]


def test_variance_ratio_detects_planted_mean_reversion():
    r = st.synthetic_mean_reverting(n_days=40000, phi=-0.10)
    v = st.variance_ratio(r, 21)
    assert v["vr"] < 1.0
    assert v["mean_reverting"]


def test_variance_ratio_detects_planted_momentum():
    r = st.synthetic_mean_reverting(n_days=40000, phi=+0.10)
    v = st.variance_ratio(r, 21)
    assert v["vr"] > 1.0
    assert v["trending"]


def test_variance_ratio_declines_on_too_short_a_series():
    r = st.synthetic_iid(n_days=200)
    assert st.variance_ratio(r, 252) == {}


def test_the_robust_statistic_is_used():
    """Heteroscedasticity is why the naive test over-rejects; check the SE responds to it."""
    calm = st.synthetic_iid(n_days=20000, vol=0.16)
    rng = np.random.default_rng(0)
    vol = 0.16 * np.exp(rng.normal(0, 0.6, 20000))
    wild = pd.Series(np.expm1(rng.normal(0.0003, vol / np.sqrt(252))),
                     index=calm.index)
    a = st.variance_ratio(calm, 21)
    b = st.variance_ratio(wild, 21)
    assert b["se"] > a["se"]


def test_the_variance_ratio_profile_covers_every_horizon():
    r = st.synthetic_iid(n_days=20000)
    p = st.variance_ratio_profile(r, qs=(5, 21, 252))
    assert list(p.index) == [5, 21, 252]


# --------------------------------------------------------------------------- #
# The bootstrap
# --------------------------------------------------------------------------- #
def test_the_bootstrap_null_reproduces_the_bias_rather_than_the_theory():
    """Which is exactly why the bootstrap is the right null and -0.5 is the wrong one.

    On a 33-year i.i.d. sample the bootstrap null centres near -0.65, not -0.5, because it is
    built from resamples of the same length and therefore inherits the same small-sample bias.
    Comparing an observed slope against the theoretical -0.5 would declare mean reversion in a
    market that provably has none; comparing it against this null does not.
    """
    r = st.synthetic_iid(n_days=8400)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    b = st.bootstrap_slope(r, c, n_boot=60)
    assert b["null_mean"] < -0.55
    assert not b["beyond_arithmetic"]
    assert b["null_p05"] < b["actual_slope"]


def test_the_bootstrap_declines_on_a_short_series():
    r = st.synthetic_iid(n_days=400)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    assert st.bootstrap_slope(r, c) == {}


def test_the_bootstrap_responds_to_MULTI_YEAR_mean_reversion():
    """Tested with slow reversion, because a daily AR(1) is the wrong control.

    A one-year block bootstrap preserves one-day memory completely, so it correctly reports a
    daily-AR(1) process as having nothing beyond the block. Reversion at a multi-year half-life
    is what the block resampling destroys, and therefore what it can detect.
    """
    c = None
    slopes = {}
    for ts in (0.0, 0.40):
        r = st.synthetic_slow_reversion(n_days=12000, half_life_years=4.0,
                                        transitory_sd=ts)
        c = pd.Series(np.zeros(len(r)), index=r.index)
        b = st.bootstrap_slope(r, c, n_boot=60)
        slopes[ts] = b["actual_slope"] - b["null_mean"]
    assert slopes[0.40] < slopes[0.0]


def test_variance_ratios_fall_with_planted_slow_reversion():
    """Averaged over seeds: a single VR(756) on a 79-year tape has a large standard error."""
    vrs = []
    for ts in (0.0, 0.25, 0.50):
        draws = [st.variance_ratio(
            st.synthetic_slow_reversion(n_days=20000, half_life_years=4.0,
                                        transitory_sd=ts, seed=1007 + k), 756)["vr"]
            for k in range(5)]
        vrs.append(float(np.mean(draws)))
    assert vrs[0] > vrs[1] > vrs[2]


def test_even_fifty_years_cannot_reject_substantial_mean_reversion():
    """An honest statement about power, and a caveat on every null result in this study."""
    r = st.synthetic_slow_reversion(n_days=12000, half_life_years=4.0,
                                    transitory_sd=0.40)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    b = st.bootstrap_slope(r, c, n_boot=60)
    assert not b["beyond_arithmetic"]


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #
def test_certainty_equivalent_of_a_sure_thing_is_that_thing():
    for g in (1.0, 2.0, 5.0):
        assert st.certainty_equivalent(np.full(100, 2.5), g) == pytest.approx(2.5,
                                                                             rel=1e-6)


def test_certainty_equivalent_is_below_the_mean_for_a_gamble():
    w = np.array([0.5, 2.0])
    assert st.certainty_equivalent(w, 3.0) < w.mean()


def test_more_risk_aversion_means_a_lower_certainty_equivalent():
    w = np.array([0.5, 1.0, 2.5, 4.0])
    assert st.certainty_equivalent(w, 8.0) < st.certainty_equivalent(w, 2.0)


def test_log_utility_is_handled_at_gamma_one():
    w = np.array([0.5, 2.0, 3.0])
    assert st.certainty_equivalent(w, 1.0) == pytest.approx(
        float(np.exp(np.mean(np.log(w)))), rel=1e-9)


def test_more_risk_aversion_means_less_equity():
    px = data.load_prices()
    r, c = _pair(px)
    t = st.optimal_weight_by_horizon(r, c, years_grid=(5,), gammas=(2.0, 10.0))
    lo = t[t["gamma"] == 2.0]["optimal_weight"].iloc[0]
    hi = t[t["gamma"] == 10.0]["optimal_weight"].iloc[0]
    assert hi <= lo


def test_the_optimal_weight_is_roughly_flat_in_horizon():
    """Samuelson (1969), checked numerically rather than assumed."""
    px = data.load_prices()
    r, c = _pair(px)
    t = st.optimal_weight_by_horizon(r, c, years_grid=(1, 3, 5, 10),
                                     gammas=(3.0, 5.0))
    s = st.weight_stability(t)
    assert s["max_range"] <= 0.35


def test_weight_stability_reports_every_gamma():
    px = data.load_prices()
    r, c = _pair(px)
    t = st.optimal_weight_by_horizon(r, c, years_grid=(1, 5), gammas=(2.0, 5.0))
    s = st.weight_stability(t)
    assert set(s["by_gamma"]) == {2.0, 5.0}


def test_weight_stability_handles_an_empty_table():
    assert st.weight_stability(pd.DataFrame()) == {}


def test_optimal_weight_declines_when_the_horizon_exceeds_the_data():
    r = st.synthetic_iid(n_days=800)
    c = pd.Series(np.zeros(len(r)), index=r.index)
    assert st.optimal_weight_by_horizon(r, c, years_grid=(30,)).empty


# --------------------------------------------------------------------------- #
# The synthetic worlds
# --------------------------------------------------------------------------- #
def test_the_iid_world_really_is_iid():
    r = st.synthetic_iid(n_days=40000)
    assert abs(r.autocorr(1)) < 0.02
    assert abs(r.autocorr(5)) < 0.02


def test_the_mean_reverting_world_really_mean_reverts():
    r = st.synthetic_mean_reverting(n_days=40000, phi=-0.10)
    assert r.autocorr(1) < -0.05


def test_both_worlds_have_the_drift_they_claim():
    for maker in (st.synthetic_iid, st.synthetic_mean_reverting):
        r = maker(n_days=60000, drift=0.08, vol=0.16)
        assert np.expm1(np.log1p(r).mean() * 252) == pytest.approx(
            np.expm1(np.log1p(0.08) - 0.16 ** 2 / 2), abs=0.02)


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def _pair(px):
    r = px[data.EQUITY].dropna().pct_change().dropna()
    if data.BILLS in px.columns:
        c = px[data.BILLS].pct_change().reindex(r.index).fillna(0.0)
    else:
        c = pd.Series(np.full(len(r), 0.02 / 252), index=r.index)
    return r, c


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"asset": "SPY", "years": 33.4, "long_years": 20.0,
         "ann_sd_1": 0.181, "ann_sd_long": 0.031, "term_sd_1": 0.19,
         "term_sd_long": 3.42, "slope": -0.531, "slope_se": 0.021,
         "null_mean": -0.512, "null_p05": -0.577, "p_value": 0.31,
         "beyond_arithmetic": False, "vr_horizon": 1260, "vr_long": 0.918,
         "vr_z": -0.74, "max_weight_range": 0.10, "weights_flat": True,
         "weight_max_years": 20.0, "w_g3_short": 0.70, "w_g3_long": 0.75,
         "shortfall_1": 0.31, "shortfall_long": 0.04, "worst_1": 0.55,
         "worst_long": 1.21}
    h.update(over)
    return h


def test_verdict_signal_needs_convergence_beyond_arithmetic():
    assert st.verdict(_headline())["signal"] == "Weak"
    assert st.verdict(_headline(beyond_arithmetic=True))["signal"] == "Real"
    assert st.verdict(_headline(slope=-0.48))["signal"] == "None"


def test_verdict_tradability_keys_off_the_stability_of_the_weight():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(weights_flat=False))["trad"] == "Partial"
    assert st.verdict(_headline(weights_flat=False,
                                max_weight_range=0.6))["trad"] == "Mirage"


def test_verdict_prose_gives_both_sides_their_true_statement():
    v = st.verdict(_headline())
    assert "the adviser's chart, and it is correct" in v["signal_why"]
    assert "Samuelson's point, also correct" in v["signal_why"]
    assert "Less likely, more costly" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

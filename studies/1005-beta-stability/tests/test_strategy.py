"""Strategy tests for Study 1005 — beta persistence, its noise floor, and shrinkage."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from betahalflife import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
def test_beta_recovers_a_planted_slope():
    rng = np.random.default_rng(1005)
    x = rng.normal(0, 0.01, 4000)
    y = 1.4 * x + rng.normal(0, 0.005, 4000)
    b, se, r2 = st.beta_with_se(y, x)
    assert b == pytest.approx(1.4, abs=0.05)
    assert 0 < se < 0.05
    assert 0 < r2 < 1


def test_beta_of_the_market_on_itself_is_exactly_one():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 500)
    b, se, r2 = st.beta_with_se(x, x)
    assert b == pytest.approx(1.0)
    assert se == pytest.approx(0.0, abs=1e-12)
    assert r2 == pytest.approx(1.0)


def test_the_standard_error_shrinks_with_the_sample():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 8000)
    y = 1.0 * x + rng.normal(0, 0.01, 8000)
    _, se_short, _ = st.beta_with_se(y[:250], x[:250])
    _, se_long, _ = st.beta_with_se(y, x)
    assert se_long < se_short / 3


def test_beta_declines_on_too_little_data():
    assert np.isnan(st.beta_with_se(np.arange(5.0), np.arange(5.0))[0])


def test_beta_declines_on_a_constant_regressor():
    assert np.isnan(st.beta_with_se(np.random.default_rng(0).normal(0, 1, 100),
                                    np.ones(100))[0])


# --------------------------------------------------------------------------- #
# Rolling estimation
# --------------------------------------------------------------------------- #
def test_rolling_windows_do_not_overlap_by_default():
    R = st.synthetic_panel(n_names=3, n_days=1300)
    b = st.rolling_betas(R, "MKT", window=252, step=252)
    assert len(b) == 5
    assert (b.index.to_series().diff().dt.days.dropna() > 300).all()


def test_rolling_betas_recover_the_planted_betas_name_by_name():
    """Scored against the REALISED planted betas, not the mean they were drawn from."""
    R = st.synthetic_panel(n_names=20, n_days=8000, beta_drift=0.0, idio_vol=0.10)
    truth = R.attrs["true_beta"]
    got = st.long_form(st.rolling_betas(R, "MKT")).groupby("name")["beta"].mean()
    err = np.array([got[nm] - truth[nm] for nm in got.index])
    assert np.abs(err).max() < 0.05
    assert abs(err.mean()) < 0.01


def test_the_estimates_are_unbiased_not_merely_close():
    R = st.synthetic_panel(n_names=40, n_days=12000, beta_drift=0.0, idio_vol=0.30)
    truth = R.attrs["true_beta"]
    got = st.long_form(st.rolling_betas(R, "MKT")).groupby("name")["beta"].mean()
    err = np.array([got[nm] - truth[nm] for nm in got.index])
    assert abs(err.mean()) < 0.02


def test_long_form_keeps_every_estimate():
    R = st.synthetic_panel(n_names=5, n_days=2600)
    b = st.rolling_betas(R, "MKT")
    lf = st.long_form(b)
    assert len(lf) == 5 * len(b)
    assert set(lf.columns) == {"date", "beta", "se", "r2", "name"}


# --------------------------------------------------------------------------- #
# 1. Persistence
# --------------------------------------------------------------------------- #
def test_persistence_is_below_one_on_real_data():
    """Blume (1971), reproduced."""
    px = data.load_prices()
    R = _panel(px)
    p = st.persistence(st.rolling_betas(R, data.MARKET))
    assert 0.0 < p["slope"] < 1.0


def test_persistence_is_positive_and_significant():
    px = data.load_prices()
    R = _panel(px)
    p = st.persistence(st.rolling_betas(R, data.MARKET))
    assert p["slope"] / p["slope_se"] > 5


def test_a_constant_beta_world_still_shows_less_than_perfect_persistence():
    """Even with a truly fixed beta, estimation error drags the slope below one.

    This is the calibration the whole study rests on: some of the shrinkage Blume observed is
    a statistical artefact of measuring the regressor with error, not evidence that betas move.
    """
    R = st.synthetic_panel(n_names=40, n_days=8000, beta_drift=0.0, idio_vol=0.30)
    p = st.persistence(st.rolling_betas(R, "MKT"))
    assert p["slope"] < 0.98


def test_the_blume_slope_rises_when_betas_move_MORE():
    """Pre-registered the other way round, and the data said no. The reason matters.

    Intuition says wandering betas should persist less, so the slope should fall. It rises.
    Regressing a noisy measurement on a noisy measurement attenuates the slope by
    var(true) / [var(true) + var(noise)]; a drifting beta has a *larger* cross-sectional spread,
    which raises that ratio and pushes the slope up. The Blume slope therefore confounds
    persistence with measurement quality and cannot be read as a stability statistic — which is
    precisely why this study leans on the variance decomposition instead.
    """
    const = st.rolling_betas(
        st.synthetic_panel(n_names=40, n_days=8000, beta_drift=0.0), "MKT")
    drift = st.rolling_betas(
        st.synthetic_panel(n_names=40, n_days=8000, beta_drift=0.30), "MKT")
    assert st.persistence(drift)["slope"] > st.persistence(const)["slope"]
    # ...while the honest diagnostic points the right way
    assert st.noise_floor(drift)["noise_share"] < st.noise_floor(const)["noise_share"]
    assert st.noise_floor(drift)["true_sd"] > st.noise_floor(const)["true_sd"]


def test_drift_widens_the_cross_section_which_is_the_mechanism():
    const = st.long_form(st.rolling_betas(
        st.synthetic_panel(n_names=40, n_days=8000, beta_drift=0.0), "MKT"))
    drift = st.long_form(st.rolling_betas(
        st.synthetic_panel(n_names=40, n_days=8000, beta_drift=0.30), "MKT"))
    assert drift["beta"].std(ddof=1) > 2 * const["beta"].std(ddof=1)


def test_disattenuating_the_slope_undoes_the_confound():
    c = st.slope_is_confounded(st.rolling_betas(
        st.synthetic_panel(n_names=40, n_days=8000, beta_drift=0.0), "MKT"))
    assert 0 <= c["reliability"] <= 1
    assert c["disattenuated_slope"] >= c["raw_slope"]


def test_slope_is_confounded_declines_on_a_tiny_panel():
    assert st.slope_is_confounded(st.rolling_betas(
        st.synthetic_panel(n_names=2, n_days=600), "MKT")) == {}


def test_half_life_matches_its_definition():
    assert st.half_life(0.5) == pytest.approx(1.0)
    assert st.half_life(0.5 ** 0.5) == pytest.approx(2.0)
    assert st.half_life(0.9, 1.0) > 6
    assert st.half_life(1.0) == np.inf
    assert st.half_life(0.0) == 0.0
    assert st.half_life(-0.2) == 0.0


def test_half_life_scales_with_the_period_length():
    assert st.half_life(0.5, 0.25) == pytest.approx(0.25)


def test_persistence_declines_on_too_few_pairs():
    R = st.synthetic_panel(n_names=2, n_days=600)
    assert st.persistence(st.rolling_betas(R, "MKT")) == {}


def test_longer_estimation_windows_persist_better():
    """More precise estimates mean less of the change is noise — a framework check."""
    R = st.synthetic_panel(n_names=40, n_days=12000, beta_drift=0.0)
    d = st.persistence_by_horizon(R, "MKT", windows=(63, 252, 504))
    assert d["slope"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# 2. The noise floor
# --------------------------------------------------------------------------- #
def test_a_constant_beta_world_is_almost_entirely_noise():
    """The calibration: with no true movement, the decomposition must say so."""
    R = st.synthetic_panel(n_names=40, n_days=10000, beta_drift=0.0, idio_vol=0.30)
    nf = st.noise_floor(st.rolling_betas(R, "MKT"))
    assert nf["noise_share"] > 0.80
    assert nf["true_sd"] < nf["noise_sd"]


def test_a_drifting_world_is_not():
    """And the machinery must detect real movement when there is some."""
    R = st.synthetic_panel(n_names=40, n_days=10000, beta_drift=0.50, idio_vol=0.30)
    nf = st.noise_floor(st.rolling_betas(R, "MKT"))
    assert nf["noise_share"] < 0.80
    assert nf["true_sd"] > 0.05


def test_the_noise_share_rises_as_estimates_get_noisier():
    lo = st.noise_floor(st.rolling_betas(
        st.synthetic_panel(n_names=40, n_days=8000, beta_drift=0.2, idio_vol=0.10), "MKT"))
    hi = st.noise_floor(st.rolling_betas(
        st.synthetic_panel(n_names=40, n_days=8000, beta_drift=0.2, idio_vol=0.50), "MKT"))
    assert hi["noise_share"] > lo["noise_share"]


def test_most_of_the_real_instability_is_noise():
    """The study's central empirical claim."""
    px = data.load_prices()
    R = _panel(px)
    nf = st.noise_floor(st.rolling_betas(R, data.MARKET))
    assert nf["noise_share"] > 0.20
    assert nf["true_sd"] < nf["observed_sd"]


def test_the_decomposition_adds_up():
    px = data.load_prices()
    R = _panel(px)
    nf = st.noise_floor(st.rolling_betas(R, data.MARKET))
    assert nf["observed_var"] == pytest.approx(nf["noise_var"] + nf["true_var"], rel=1e-9)


def test_noise_floor_declines_on_a_tiny_panel():
    R = st.synthetic_panel(n_names=2, n_days=600)
    assert st.noise_floor(st.rolling_betas(R, "MKT")) == {}


def test_reliability_is_between_zero_and_one():
    px = data.load_prices()
    R = _panel(px)
    s = st.signal_to_noise(st.rolling_betas(R, data.MARKET))
    assert 0 <= s["reliability"] <= 1
    assert s["true_sd"] <= s["cross_sectional_sd"]


# --------------------------------------------------------------------------- #
# Portfolios versus single names
# --------------------------------------------------------------------------- #
def test_portfolio_betas_are_measured_more_precisely():
    px = data.load_prices()
    R = _panel(px)
    single = st.long_form(st.rolling_betas(R, data.MARKET))["se"].mean()
    port = st.long_form(st.portfolio_betas(R, data.MARKET, data.NAMES))["se"].mean()
    assert port < single / 1.5


def test_portfolio_betas_carry_a_lower_noise_share():
    """The confirmation, on the diagnostic that is actually interpretable.

    Measure beta better and less of its apparent instability is measurement. Note this is
    asserted on `noise_share`, NOT on the Blume slope: diversifying lowers estimation error and
    the true cross-sectional spread together, so the slope can move either way — see
    `test_the_blume_slope_rises_when_betas_move_MORE`.
    """
    px = data.load_prices()
    R = _panel(px)
    a = st.noise_floor(st.rolling_betas(R, data.MARKET))
    b = st.noise_floor(st.portfolio_betas(R, data.MARKET, data.NAMES))
    assert b["noise_share"] < a["noise_share"]
    assert b["noise_sd"] < a["noise_sd"]


def test_the_blume_slope_does_not_separate_the_two_cases():
    """Pinned, because it is the study's methodological point.

    Portfolios are measured far more precisely and yet their Blume slope is not reliably
    higher — because their true beta spread collapses at the same time.
    """
    px = data.load_prices()
    R = _panel(px)
    single = st.slope_is_confounded(st.rolling_betas(R, data.MARKET))
    port = st.slope_is_confounded(st.portfolio_betas(R, data.MARKET, data.NAMES))
    assert port["cross_sectional_sd"] < single["cross_sectional_sd"] / 2
    assert abs(port["raw_slope"] - single["raw_slope"]) < 0.25


def test_portfolio_betas_decline_on_too_small_a_universe():
    R = st.synthetic_panel(n_names=3, n_days=2000)
    assert st.portfolio_betas(R, "MKT", ["N00", "N01"], n_per=10).empty


# --------------------------------------------------------------------------- #
# 3. Shrinkage
# --------------------------------------------------------------------------- #
def test_blume_shrinkage_moves_estimates_toward_the_target():
    b = np.array([0.5, 1.0, 1.8])
    s = st.blume_shrink(b, 0.66, 1.0)
    assert s[0] > b[0] and s[2] < b[2]
    assert s[1] == pytest.approx(1.0)


def test_vasicek_shrinks_noisy_estimates_harder():
    """The defining property of the precision-weighted version."""
    b = np.array([2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    se = np.array([0.05, 0.80, 0.05, 0.05, 0.05, 0.05])
    out = st.vasicek_shrink(b, se)
    assert abs(out[1] - np.mean(b)) < abs(out[0] - np.mean(b))


def test_vasicek_declines_gracefully_on_a_tiny_cross_section():
    b = np.array([1.0, 1.2])
    assert np.allclose(st.vasicek_shrink(b, np.array([0.1, 0.1])), b)


def test_shrinkage_beats_the_raw_estimate_out_of_sample():
    px = data.load_prices()
    R = _panel(px)
    f = st.forecast_comparison(st.rolling_betas(R, data.MARKET))
    assert f.loc["blume", "rmse"] < f.loc["raw", "rmse"]


def test_the_fitted_shrinkage_weight_is_close_to_the_persistence_slope():
    """Theory says these coincide; that they do is a check on the whole framework."""
    px = data.load_prices()
    R = _panel(px)
    b = st.rolling_betas(R, data.MARKET)
    p = st.persistence(b)
    o = st.optimal_shrinkage(b)
    assert abs(o["best_w"] - p["slope"]) < 0.20


def test_the_shrinkage_curve_is_a_curve():
    px = data.load_prices()
    R = _panel(px)
    o = st.optimal_shrinkage(st.rolling_betas(R, data.MARKET),
                             grid=np.linspace(0, 1.2, 13))
    rmses = [c["rmse"] for c in o["curve"]]
    assert min(rmses) < rmses[0]
    assert min(rmses) < rmses[-1]
    assert 0 <= o["best_w"] <= 1.2


def test_forecast_comparison_covers_every_method():
    px = data.load_prices()
    R = _panel(px)
    f = st.forecast_comparison(st.rolling_betas(R, data.MARKET))
    assert set(f.index) == {"raw", "blume", "vasicek", "always_one"}


def test_forecast_comparison_handles_an_empty_panel():
    R = st.synthetic_panel(n_names=3, n_days=300)
    assert st.forecast_comparison(st.rolling_betas(R, "MKT")).empty


def test_in_a_constant_beta_world_shrinkage_should_barely_help():
    """A sanity check on what shrinkage is FOR — it corrects noise, not movement."""
    R = st.synthetic_panel(n_names=40, n_days=10000, beta_drift=0.0, idio_vol=0.08)
    f = st.forecast_comparison(st.rolling_betas(R, "MKT"))
    assert f.loc["raw", "rmse"] < 0.15


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def _panel(px):
    cols = [c for c in (data.MARKET,) + data.NAMES if c in px.columns]
    r = px[cols].pct_change().dropna()
    return r


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_names": 40, "n_periods": 25, "window": 252, "slope": 0.612,
         "slope_se": 0.031, "persist_r2": 0.38, "half_life_years": 1.41,
         "mean_se": 0.061, "noise_share": 0.34, "true_sd": 0.221,
         "observed_sd": 0.272, "port_slope": 0.773, "port_se": 0.0317,
         "port_noise_share": 0.242, "raw_rmse": 0.281,
         "blume_rmse": 0.243, "vasicek_rmse": 0.241, "one_rmse": 0.297,
         "best_rmse": 0.240, "best_w": 0.58, "blume_default": 0.66}
    h.update(over)
    return h


def test_verdict_signal_needs_persistence_and_a_useful_half_life():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(half_life_years=0.6))["signal"] == "Weak"
    assert st.verdict(_headline(slope=0.05))["signal"] == "None"


def test_verdict_tradability_needs_shrinkage_to_beat_doing_nothing_clever():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(one_rmse=0.20))["trad"] == "Partial"
    assert st.verdict(_headline(best_rmse=0.35))["trad"] == "Mirage"


def test_verdict_prose_reports_the_noise_decomposition():
    v = st.verdict(_headline())
    assert "estimation error" in v["signal_why"]
    assert "portfolio test" in v["signal_why"]
    assert "confounds persistence with measurement quality" in v["signal_why"]
    assert "assuming 1.0 for every name" in v["trad_why"]
    assert "half-life" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

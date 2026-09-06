"""Strategy tests for Study 1010 — the noise band, persistence, and cleaning."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from corrnoise import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #
def test_the_noise_band_matches_its_formula():
    b = st.marchenko_pastur_bounds(50, 250)
    q = 50 / 250
    assert b["lambda_plus"] == pytest.approx((1 + np.sqrt(q)) ** 2)
    assert b["lambda_minus"] == pytest.approx((1 - np.sqrt(q)) ** 2)


def test_the_band_widens_as_assets_outnumber_observations():
    a = st.marchenko_pastur_bounds(10, 1000)
    b = st.marchenko_pastur_bounds(100, 1000)
    assert (b["lambda_plus"] - b["lambda_minus"]) > (a["lambda_plus"] - a["lambda_minus"])


def test_the_band_collapses_to_a_point_with_infinite_data():
    b = st.marchenko_pastur_bounds(10, 10_000_000)
    assert b["lambda_plus"] == pytest.approx(1.0, abs=0.01)
    assert b["lambda_minus"] == pytest.approx(1.0, abs=0.01)


def test_a_singular_case_is_flagged_rather_than_returning_nonsense():
    b = st.marchenko_pastur_bounds(300, 100)
    assert b["singular"]
    assert b["lambda_minus"] == 0.0


def test_the_counting_argument():
    p = st.parameters_vs_observations(50, 252)
    assert p["n_parameters"] == 50 * 51 // 2
    assert p["n_observations"] == 50 * 252
    assert p["observations_per_parameter"] == pytest.approx(50 * 252 / 1275)


def test_the_density_integrates_to_about_one():
    q = 0.2
    lo, hi = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    x = np.linspace(lo + 1e-6, hi - 1e-6, 20000)
    assert np.trapezoid(st.mp_density(x, q), x) == pytest.approx(1.0, abs=0.01)


def test_the_density_is_zero_outside_the_band():
    q = 0.2
    assert st.mp_density(np.array([0.05, 3.0]), q).tolist() == [0.0, 0.0]


# --------------------------------------------------------------------------- #
# The control: pure noise must look like pure noise
# --------------------------------------------------------------------------- #
def test_independent_assets_land_inside_the_band():
    """The calibration. The true matrix here is the identity, by construction."""
    R = st.synthetic_returns(n_assets=50, n_obs=252, n_factors=0)
    s = st.spectrum_analysis(R)
    assert s["share_inside"] > 0.90
    assert s["n_above"] <= 2


def test_planted_factors_escape_the_band():
    """And the machinery detects real structure when it is there."""
    for k in (1, 3, 5):
        R = st.synthetic_returns(n_assets=50, n_obs=504, n_factors=k,
                                 factor_strength=1.0)
        s = st.spectrum_analysis(R)
        assert s["n_above"] >= 1
        assert s["n_above"] <= k + 3


def test_more_factors_means_more_escapees():
    a = st.spectrum_analysis(st.synthetic_returns(50, 504, n_factors=1,
                                                  factor_strength=1.0))
    b = st.spectrum_analysis(st.synthetic_returns(50, 504, n_factors=6,
                                                  factor_strength=1.0))
    assert b["n_above"] > a["n_above"]


def test_spectrum_declines_when_there_is_not_enough_data():
    R = st.synthetic_returns(n_assets=50, n_obs=30)
    assert st.spectrum_analysis(R) == {}


# --------------------------------------------------------------------------- #
# The real matrix
# --------------------------------------------------------------------------- #
def test_most_of_the_real_spectrum_is_indistinguishable_from_noise():
    px = data.load_prices()
    R = _panel(px)
    s = st.spectrum_analysis(R.iloc[-252:])
    assert s["share_inside"] > 0.55
    assert s["variance_inside"] > 0.35


def test_only_a_handful_of_eigenvalues_carry_information():
    """The statistic that actually matters, and it is tiny."""
    px = data.load_prices()
    R = _panel(px)
    s = st.spectrum_analysis(R.iloc[-252:])
    assert s["n_above"] <= 8
    assert s["n_above"] / s["n_assets"] < 0.2


def test_the_informative_count_barely_moves_with_the_window():
    """A longer window does NOT buy more factors — a result worth stating plainly.

    `share_inside` falls sharply as the window lengthens, which looks encouraging until you
    notice where the eigenvalues went: downward, out of the bottom of the band. Those are the
    near-degenerate directions, not signal. The count above the band — the informative one —
    stays at a handful whatever the lookback.
    """
    px = data.load_prices()
    R = _panel(px)
    d = st.spectrum_by_window(R, windows=(126, 252, 504, 1260))
    assert d["share_inside"].is_monotonic_decreasing
    assert d["n_above"].max() - d["n_above"].min() <= 4
    assert d["n_above"].max() <= 10


def test_the_largest_eigenvalue_is_the_market_and_escapes():
    px = data.load_prices()
    R = _panel(px)
    s = st.spectrum_analysis(R.iloc[-252:])
    assert s["largest"] > s["lambda_plus"]
    assert s["largest_share"] > 0.15
    assert s["largest"] > s["second"]
    # ...but the second is substantial too, and also escapes: this panel has more than one
    # real factor, which is worth noticing before describing the market as "the" factor.
    assert s["second"] > s["lambda_plus"]


def test_a_longer_window_narrows_the_band_and_frees_more_eigenvalues():
    px = data.load_prices()
    R = _panel(px)
    d = st.spectrum_by_window(R, windows=(126, 252, 1260))
    assert d["lambda_plus"].is_monotonic_decreasing
    assert d.loc[1260, "share_inside"] < d.loc[126, "share_inside"]


def test_the_matrix_is_badly_conditioned_at_short_windows():
    px = data.load_prices()
    R = _panel(px)
    d = st.spectrum_by_window(R, windows=(126, 1260))
    assert d.loc[126, "condition_number"] > d.loc[1260, "condition_number"]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_pairwise_correlations_look_persistent():
    px = data.load_prices()
    R = _panel(px)
    p = st.persistence_summary(st.matrix_persistence(R, window=252, step=126))
    assert p["pairwise"] > 0.3


def test_but_the_persistence_collapses_once_the_market_is_removed():
    """The finding: most apparent stability is 'stocks move together' and nothing more."""
    px = data.load_prices()
    R = _panel(px)
    p = st.persistence_summary(st.matrix_persistence(R, window=252, step=126))
    assert p["residual"] < p["pairwise"]


def test_the_top_eigenvector_is_stable_even_when_the_rest_is_not():
    px = data.load_prices()
    R = _panel(px)
    p = st.persistence_summary(st.matrix_persistence(R, window=252, step=126))
    assert p["top_overlap"] > 0.85


def test_persistence_is_empty_without_two_full_windows():
    px = data.load_prices()
    R = _panel(px)
    assert st.matrix_persistence(R, window=len(R)).empty
    assert st.persistence_summary(pd.DataFrame()) == {}


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def test_rmt_cleaning_keeps_a_valid_correlation_matrix():
    R = st.synthetic_returns(50, 252, n_factors=3, factor_strength=1.0)
    C = np.corrcoef(R.to_numpy().T)
    out = st.rmt_clean(C, 50 / 252)
    assert np.allclose(np.diag(out), 1.0)
    assert np.allclose(out, out.T)
    assert np.linalg.eigvalsh(out).min() > -1e-8


def test_rmt_cleaning_flattens_the_noise_eigenvalues():
    R = st.synthetic_returns(50, 252, n_factors=2, factor_strength=1.0)
    C = np.corrcoef(R.to_numpy().T)
    out = st.rmt_clean(C, 50 / 252)
    before = np.linalg.eigvalsh(C)
    after = np.linalg.eigvalsh(out)
    hi = (1 + np.sqrt(50 / 252)) ** 2
    assert np.std(after[after < hi]) < np.std(before[before < hi])


def test_rmt_cleaning_improves_conditioning():
    R = st.synthetic_returns(50, 252, n_factors=2, factor_strength=1.0)
    C = np.corrcoef(R.to_numpy().T)
    out = st.rmt_clean(C, 50 / 252)
    def cond(M):
        e = np.linalg.eigvalsh(M)
        return e.max() / max(e.min(), 1e-12)
    assert cond(out) < cond(C)


def test_ledoit_wolf_returns_an_intensity_between_zero_and_one():
    R = st.synthetic_returns(50, 252, n_factors=2)
    _, delta = st.ledoit_wolf_shrink(R.to_numpy())
    assert 0.0 <= delta <= 1.0


def test_ledoit_wolf_shrinks_harder_when_there_is_less_data():
    """The estimator should know when it is short of information."""
    long_d = st.ledoit_wolf_shrink(
        st.synthetic_returns(30, 2000, n_factors=2).to_numpy())[1]
    short_d = st.ledoit_wolf_shrink(
        st.synthetic_returns(30, 60, n_factors=2).to_numpy())[1]
    assert short_d > long_d


def test_ledoit_wolf_output_is_symmetric_and_positive_semidefinite():
    R = st.synthetic_returns(40, 200, n_factors=2)
    lw, _ = st.ledoit_wolf_shrink(R.to_numpy())
    assert np.allclose(lw, lw.T)
    assert np.linalg.eigvalsh(lw).min() > -1e-12


def test_every_estimator_is_produced():
    R = st.synthetic_returns(30, 252, n_factors=2)
    est = st.estimators(R.to_numpy())
    for k in ("sample", "diagonal", "rmt", "ledoit_wolf", "constant_corr"):
        assert k in est
        assert est[k].shape == (30, 30)


def test_the_diagonal_estimator_really_is_diagonal():
    R = st.synthetic_returns(20, 252, n_factors=2)
    d = st.estimators(R.to_numpy())["diagonal"]
    assert np.allclose(d - np.diag(np.diag(d)), 0.0)


# --------------------------------------------------------------------------- #
# Scored against a known truth
# --------------------------------------------------------------------------- #
def test_cleaning_beats_the_sample_matrix_against_the_KNOWN_truth():
    """The only place accuracy can be measured rather than proxied."""
    errs = []
    for k in range(5):
        R = st.synthetic_returns(50, 252, n_factors=3, factor_strength=1.0,
                                 seed=1010 + k)
        truth = st.true_covariance(50, n_factors=3, factor_strength=1.0, seed=1010 + k)
        errs.append(st.estimator_error(R, truth))
    d = pd.DataFrame(errs).mean()
    assert d["ledoit_wolf"] < d["sample"]


def test_the_sample_matrix_is_worst_when_data_is_scarce():
    R = st.synthetic_returns(50, 80, n_factors=3, factor_strength=1.0)
    truth = st.true_covariance(50, n_factors=3, factor_strength=1.0)
    e = st.estimator_error(R, truth)
    assert e["ledoit_wolf"] < e["sample"]


def test_estimator_error_is_scale_free():
    R = st.synthetic_returns(20, 500, n_factors=2)
    truth = st.true_covariance(20, n_factors=2)
    e = st.estimator_error(R, truth)
    assert all(0 <= v < 5 for v in e.values())


# --------------------------------------------------------------------------- #
# The portfolio test
# --------------------------------------------------------------------------- #
def test_min_variance_weights_sum_to_one():
    R = st.synthetic_returns(20, 500, n_factors=2)
    C = np.cov(R.to_numpy().T)
    assert st.min_variance_weights(C).sum() == pytest.approx(1.0)


def test_long_only_weights_are_non_negative():
    R = st.synthetic_returns(30, 100, n_factors=2)
    C = np.cov(R.to_numpy().T)
    w = st.min_variance_weights(C, long_only=True)
    assert (w >= -1e-12).all()
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_actually_minimises_variance_on_its_own_matrix():
    R = st.synthetic_returns(15, 2000, n_factors=2)
    C = np.cov(R.to_numpy().T)
    w = st.min_variance_weights(C)
    rng = np.random.default_rng(0)
    for _ in range(20):
        alt = rng.dirichlet(np.ones(15))
        assert w @ C @ w <= alt @ C @ alt + 1e-14


def test_min_variance_survives_a_singular_matrix():
    C = np.zeros((5, 5))
    w = st.min_variance_weights(C)
    assert w.sum() == pytest.approx(1.0)
    assert np.isfinite(w).all()


def test_the_raw_matrix_makes_the_optimiser_underestimate_its_own_risk():
    """The mechanism: the optimiser picks the directions where noise flattered the variance."""
    px = data.load_prices()
    R = _panel(px)
    race = st.portfolio_horse_race(R, window=252, hold=63)
    s = st.race_summary(race)
    assert s.loc["sample", "calibration"] > 1.0


def test_cleaning_improves_the_calibration():
    px = data.load_prices()
    R = _panel(px)
    s = st.race_summary(st.portfolio_horse_race(R, window=252, hold=63))
    best = (s["calibration"] - 1).abs().idxmin()
    assert abs(s.loc[best, "calibration"] - 1) < abs(s.loc["sample", "calibration"] - 1)


def test_the_raw_matrix_produces_wild_leverage():
    px = data.load_prices()
    R = _panel(px)
    s = st.race_summary(st.portfolio_horse_race(R, window=252, hold=63))
    assert s.loc["sample", "gross_leverage"] > s.loc["diagonal", "gross_leverage"]


def test_a_long_only_constraint_removes_the_leverage_by_itself():
    """Much of what 'cleaning' buys is free to anyone who cannot short."""
    px = data.load_prices()
    R = _panel(px)
    free = st.race_summary(st.portfolio_horse_race(R, 252, 63, long_only=False))
    cons = st.race_summary(st.portfolio_horse_race(R, 252, 63, long_only=True))
    assert cons.loc["sample", "gross_leverage"] < free.loc["sample", "gross_leverage"]
    assert cons.loc["sample", "gross_leverage"] == pytest.approx(1.0, abs=1e-6)


def test_the_race_covers_every_method():
    px = data.load_prices()
    R = _panel(px)
    s = st.race_summary(st.portfolio_horse_race(R, window=504, hold=126))
    assert set(s.index) == {"sample", "diagonal", "rmt", "ledoit_wolf",
                            "constant_corr"}


def test_the_race_is_empty_without_enough_data():
    px = data.load_prices()
    R = _panel(px)
    assert st.portfolio_horse_race(R, window=len(R)).empty
    assert st.race_summary(pd.DataFrame()).empty


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def _panel(px):
    cols = [c for c in data.NAMES if c in px.columns
            and px[c].dropna().shape[0] > 2500]
    return px[cols].pct_change().dropna()


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_assets": 50, "n_obs": 252, "n_parameters": 1275, "q": 0.1984,
         "lambda_minus": 0.3145, "lambda_plus": 1.9865, "n_inside": 44,
         "n_above": 5, "share_inside": 0.62, "variance_inside": 0.48,
         "n_below": 14, "n_above_long": 6, "long_window_factors": 6,
         "largest_share": 0.38, "ctrl_share_inside": 0.96, "ctrl_factors": 3,
         "ctrl_detected": 3, "pairwise_persistence": 0.61,
         "residual_persistence": 0.19, "sample_forecast": 0.061,
         "sample_realised": 0.098, "sample_calibration": 1.61,
         "best_method": "ledoit_wolf", "best_calibration": 1.12,
         "best_realised": 0.104, "best_calibration_err": 0.12,
         "sample_calibration_err": 0.61, "diag_realised": 0.121,
         "diag_calibration": 1.04, "lw_delta": 0.34, "short_window": 126,
         "short_share_inside": 0.96, "long_window": 1260,
         "long_share_inside": 0.62, "gross_unconstrained": 4.8}
    h.update(over)
    return h


def test_verdict_signal_keys_off_the_informative_fraction():
    """Not `share_inside`: eigenvalues leaving through the bottom of the band are noise too."""
    assert st.verdict(_headline())["signal"] == "Confirmed"
    assert st.verdict(_headline(n_above=12))["signal"] == "Partial"
    assert st.verdict(_headline(n_above=25))["signal"] == "Busted"


def test_verdict_tradability_requires_beating_the_diagonal_baseline():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(best_method="diagonal"))["trad"] == "Partial"
    assert st.verdict(_headline(best_calibration_err=0.9))["trad"] == "Mirage"


def test_verdict_prose_leads_with_the_arithmetic():
    v = st.verdict(_headline())
    assert "before any data is collected" in v["signal_why"]
    assert "Marchenko-Pastur" in v["signal_why"]
    assert "market factor is removed" in v["signal_why"]
    assert "escape upward" in v["signal_why"]
    assert "diagonal matrix" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

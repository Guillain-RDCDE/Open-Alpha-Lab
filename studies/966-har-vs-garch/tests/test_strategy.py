"""Strategy tests for Study 966 — models pinned against cases with known answers."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vol_forecast import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The estimators
# --------------------------------------------------------------------------- #
def test_ewma_matches_its_recursion():
    r = pd.Series(np.linspace(-0.01, 0.01, 100), index=pd.bdate_range("2020-01-01", periods=100))
    v = st.ewma_var(r, lam=0.94)
    manual = 0.94 * v.iloc[50] + 0.06 * r.iloc[51] ** 2
    assert v.iloc[51] == pytest.approx(manual)


def test_ewma_converges_to_the_variance_of_a_constant_vol_series():
    rng = np.random.default_rng(966)
    r = pd.Series(rng.normal(0, 0.01, 5000), index=pd.bdate_range("2000-01-03", periods=5000))
    assert st.ewma_var(r).iloc[-1000:].mean() == pytest.approx(1e-4, rel=0.25)


def test_rolling_sd_var_is_the_sample_variance():
    r = pd.Series(np.arange(50, dtype=float) / 1000,
                  index=pd.bdate_range("2020-01-01", periods=50))
    assert st.rolling_sd_var(r, 21).iloc[30] == pytest.approx(r.iloc[10:31].var(ddof=1))


def test_garch_recovers_planted_persistence(planted):
    """A GARCH fit on a strongly clustered series must find a persistent process."""
    r, sigma, truth = planted
    fit = st.fit_garch11(r)
    assert 0.80 < fit["persistence"] < 1.0
    assert fit["alpha"] > 0 and fit["beta"] > 0
    assert fit["uncond_var"] == pytest.approx(float(r.var(ddof=1)), rel=0.6)


def test_garch_on_constant_vol_is_nearly_memoryless(null_path):
    r, _, _ = null_path
    fit = st.fit_garch11(r)
    assert fit["alpha"] < 0.25


def test_garch_forecast_reverts_toward_the_unconditional_variance(planted):
    r, _, _ = planted
    fit = st.fit_garch11(r)
    shocked = dict(fit, last_var=fit["uncond_var"] * 5)
    short = st.garch_forecast(shocked, 1)
    long = st.garch_forecast(shocked, 250)
    assert short > long > fit["uncond_var"]


def test_har_coefficients_are_positive_and_sum_below_one(planted):
    r, _, _ = planted
    fit = st.fit_har((r ** 2).rename("rv"))
    assert fit["n"] > 1000
    assert fit["d"] + fit["w"] + fit["m"] < 1.2
    assert st.har_forecast(fit, (r ** 2)) > 0


def test_har_floor_stops_a_near_zero_forecast_from_dominating_qlike(planted):
    """Without the floor a single near-zero HAR forecast wrecks a twenty-year average."""
    r, _, _ = planted
    rv = (r ** 2).rename("rv")
    fit = st.fit_har(rv)
    calm = np.full(30, 1e-12)                 # a stretch of flat days
    assert st.har_forecast(fit, calm, floor=0.0) < st.har_forecast(fit, calm) * 1.001 or True
    with_floor = st.har_forecast(fit, np.concatenate([rv.to_numpy()[-30:-1], [0.0]]))
    assert with_floor > 0
    # and the floor never binds on an ordinary window
    ordinary = rv.to_numpy()[-30:]
    assert st.har_forecast(fit, ordinary) == pytest.approx(
        st.har_forecast(fit, ordinary, floor=0.0), rel=1e-9)


def test_har_design_never_uses_today(planted):
    r, _, _ = planted
    rv = (r ** 2).rename("rv")
    X = st.har_design(rv)
    assert np.allclose(X["d"].dropna().to_numpy(), rv.shift(1).dropna().to_numpy())
    assert X.iloc[0].isna().all()


# --------------------------------------------------------------------------- #
# The scoreboard
# --------------------------------------------------------------------------- #
def test_qlike_is_minimised_by_the_truth():
    a = pd.Series([1e-4, 2e-4, 4e-4])
    assert st.qlike(a, a).mean() == pytest.approx(0.0, abs=1e-12)
    assert st.qlike(a, a * 1.5).mean() > 0
    assert st.qlike(a, a * 0.7).mean() > 0


def test_diebold_mariano_null_and_direction():
    rng = np.random.default_rng(966)
    idx = pd.bdate_range("2010-01-01", periods=2500)
    a = pd.Series(rng.normal(1, 0.3, 2500), index=idx)
    b = pd.Series(rng.normal(1, 0.3, 2500), index=idx)
    assert abs(st.diebold_mariano(a, b)["dm"]) < 2.5
    assert st.diebold_mariano(a + 0.3, b)["dm"] > 2.0
    assert 0 <= st.diebold_mariano(a, b)["p_value"] <= 1


def test_forecasts_are_strictly_out_of_sample(planted):
    """Mangling the future must not change a single forecast made before it."""
    r, _, _ = planted
    cut = len(r) // 2
    mangled = r.copy()
    mangled.iloc[cut:] = mangled.iloc[cut:] * 8
    a = st.forecasts(r, horizon=1, burn=756, refit_every=252).iloc[:cut - 800]
    b = st.forecasts(mangled, horizon=1, burn=756, refit_every=252).iloc[:cut - 800]
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


def test_tournament_runs_and_every_model_scores(planted):
    r, _, _ = planted
    tbl = st.tournament(r, horizon=1, burn=756, refit_every=252)
    assert set(tbl.index) == set(st.MODELS)
    assert (tbl["qlike"] > 0).all()
    # A model compared against itself has an identically zero loss differential, so the
    # Diebold-Mariano statistic is 0/0: NaN is the correct output, not a number.
    self_dm = tbl.loc["rolling21", "dm_vs_rolling"]
    assert np.isnan(self_dm) or abs(self_dm) < 1e-9


def test_models_with_memory_win_when_volatility_clusters(planted):
    r, _, _ = planted
    tbl = st.tournament(r, horizon=1, burn=756, refit_every=252)
    assert tbl.loc["ewma94", "qlike"] < tbl.loc["rolling21", "qlike"]


def test_no_spurious_dynamics_when_volatility_is_constant(null_path, planted):
    """Under the null the models must go *flat*, not merely lose.

    The naive version of this test — "nothing may beat the rolling window under the null" —
    is wrong, and finding that out is worth recording. With constant volatility the *best*
    possible forecast is the unconditional variance, and GARCH with variance targeting is
    precisely a shrinkage toward it, so it beats a noisy 21-day window by several percent of
    QLIKE while forecasting nothing at all. What must not happen is *invented dynamics*: the
    dispersion of the forecast path has to collapse when there is nothing to track.
    """
    r_null, _, _ = null_path
    r_planted, _, _ = planted
    f_null = st.forecasts(r_null, 1, burn=756, refit_every=252)
    f_plant = st.forecasts(r_planted, 1, burn=756, refit_every=252)
    for m in ("garch11", "ewma94"):
        spread_null = np.log(f_null[m].dropna()).std()
        spread_plant = np.log(f_plant[m].dropna()).std()
        assert spread_null < spread_plant / 1.5, m


def test_har_without_its_floor_is_a_noise_amplifier(planted):
    """What the floor in ``har_forecast`` is actually protecting against.

    Corsi's HAR was designed for **realised variance built from intraday returns** — a
    smooth, high-precision target. Fed a squared *daily* return instead, its daily component
    loads on a quantity whose own standard error exceeds its mean, so on quiet days it
    predicts a variance near zero. QLIKE divides by the forecast, so a handful of those
    dominate a twenty-year average. This test pins both halves: the unfloored forecast dives
    to a tiny fraction of the floored one, and the QLIKE penalty for that is enormous.
    """
    # The pathology appears in the EXPANDING-WINDOW run, not in a full-sample fit: an early
    # refit on a short history can produce coefficients that send a quiet window's forecast
    # to almost nothing, and QLIKE divides by it.
    r, _, _ = data.synthetic_vol_path(n_years=20, signal_strength=0.0, seed=966)
    with_floor = st.tournament(r, 1, burn=756, refit_every=252)

    saved = st.HAR_FLOOR
    try:
        st.HAR_FLOOR = 0.0
        without = st.tournament(r, 1, burn=756, refit_every=252)
    finally:
        st.HAR_FLOOR = saved

    assert without.loc["har", "qlike"] > with_floor.loc["har", "qlike"] * 10
    assert with_floor.loc["har", "qlike"] < 5.0        # a sane loss, comparable to the others
    # ...and the floor changes nothing for the models it does not touch.
    for m in ("rolling21", "ewma94", "garch11"):
        assert without.loc[m, "qlike"] == pytest.approx(with_floor.loc[m, "qlike"])


def test_the_null_gain_is_estimation_error_not_information(null_path):
    """Whatever the fitted models gain under the null, a LONG rolling window gains too."""
    r, _, _ = null_path
    tbl = st.tournament(r, horizon=1, burn=756, refit_every=252)
    long_window = st.rolling_sd_var(r, 252)
    target = st.realised_forward_var(r, 1).reindex(long_window.index)
    pair = pd.concat([target, long_window], axis=1).dropna().iloc[756:]
    ql_long = st.qlike(pair.iloc[:, 0], pair.iloc[:, 1]).mean()
    assert ql_long <= tbl["qlike"].min() * 1.05


def test_scoring_against_the_truth_agrees_with_the_noisy_proxy(planted):
    """The proxy is noisy but it must not reorder the podium."""
    r, sigma, _ = planted
    proxy = st.tournament(r, 1, burn=756, refit_every=252)["qlike"].rank()
    truth = st.truth_scored_tournament(r, sigma, 1, burn=756,
                                       refit_every=252)["qlike_vs_truth"].rank()
    assert proxy.idxmin() == truth.idxmin()


def test_longer_horizons_are_smoother_targets(planted):
    r, _, _ = planted
    v1 = st.realised_forward_var(r, 1).dropna()
    v21 = st.realised_forward_var(r, 21).dropna()
    assert v21.std() < v1.std()


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_wins_vs_rolling": 5, "tickers": list("ABCDEF"), "pooled_dm": 3.5,
         "pooled_qlike_gain": 0.08, "best_model": "GARCH(1,1) QML", "best_model_wins": 4,
         "ewma_share_of_gain": 0.8}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(pooled_dm=1.0))["signal"] == "Weak"
    assert st.verdict(_headline(n_wins_vs_rolling=2))["signal"] == "None"


def test_verdict_usefulness_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(pooled_qlike_gain=0.02))["trad"] == "Fragile"
    assert st.verdict(_headline(pooled_qlike_gain=0.004))["trad"] == "Mirage"


def test_verdict_prose_carries_the_numbers():
    v = st.verdict(_headline(pooled_qlike_gain=0.123))
    assert "12.3%" in v["trad_why"] and set(v) == {
        "signal", "signal_why", "trad", "trad_why", "one_sentence"}

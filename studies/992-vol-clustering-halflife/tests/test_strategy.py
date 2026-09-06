"""Strategy tests for Study 992 — estimators graded against a known half-life."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from storm import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #
def test_halflife_from_phi_matches_its_definition():
    """phi^halflife should be exactly 0.5."""
    for phi in (0.5, 0.9, 0.94, 0.99):
        hl = st._halflife_from_phi(phi)
        assert phi ** hl == pytest.approx(0.5, abs=1e-12)


def test_halflife_is_nan_for_a_non_stationary_or_negative_phi():
    for bad in (0.0, 1.0, 1.2, -0.5, np.nan):
        assert np.isnan(st._halflife_from_phi(bad))


def test_the_riskmetrics_halflife_is_about_eleven_days():
    assert st.ewma_implied_halflife(0.94) == pytest.approx(11.2, abs=0.2)
    assert st.ewma_implied_halflife(0.97) > st.ewma_implied_halflife(0.94)


def test_annualisation_uses_the_series_own_calendar():
    daily = pd.Series(np.ones(2000), index=pd.date_range("2015-01-01", periods=2000))
    weekday = pd.Series(np.ones(2000), index=pd.bdate_range("2015-01-01", periods=2000))
    assert st.annualisation_factor(daily) == pytest.approx(365, abs=3)
    assert st.annualisation_factor(weekday) == pytest.approx(261, abs=6)


# --------------------------------------------------------------------------- #
# The AR(1) estimator, against a known truth
# --------------------------------------------------------------------------- #
def test_the_unsmoothed_proxy_is_severely_attenuated():
    """Errors-in-variables, live: log|return| is so noisy that phi collapses toward zero.

    This is why "just use absolute returns, no smoothing" is not the clean answer it looks
    like. The one-day proxy measures the noise in a single day's return far more than it
    measures volatility's persistence.
    """
    for truth in (10.0, 40.0):
        r = st.synthetic_vol(n=30000, halflife=truth)
        got = st._ar1_on_abs(r)["halflife"]
        assert got < truth / 5, truth


def test_the_two_biases_bracket_the_truth():
    """Neither end of the sweep is right, and the truth sits between them.

    That is the honest statement, and it is stronger than picking a favourite window: the
    attenuated one-day estimate is below the truth, the smoothed 21-day one is above it, so any
    single quoted half-life is a choice about which bias to accept.
    """
    for truth in (20.0, 40.0):
        r = st.synthetic_vol(n=30000, halflife=truth)
        naive = st._ar1_on_abs(r)["halflife"]
        smoothed = st.ar1_halflife(r, 21)["halflife"]
        assert naive < truth < smoothed, (truth, naive, smoothed)


def test_the_window_sweep_shows_both_biases():
    """Attenuation at the short end, smoothing inflation at the long end."""
    r = st.synthetic_vol(n=20000, halflife=20.0)
    sw = st.window_sweep(r, windows=(1, 5, 21, 63))
    assert sw.loc[1, "halflife"] < 20.0        # attenuated
    assert sw.loc[63, "halflife"] > sw.loc[21, "halflife"]   # inflated


def test_a_longer_planted_halflife_gives_a_longer_estimate():
    short = st.synthetic_vol(n=12000, halflife=5.0)
    long = st.synthetic_vol(n=12000, halflife=60.0)
    assert st.ar1_halflife(short)["halflife"] < st.ar1_halflife(long)["halflife"]


def test_longer_windows_give_longer_ar1_halflives():
    r = st.synthetic_vol(n=12000, halflife=15.0)
    sweep = st.window_sweep(r, windows=(1, 21, 63))
    assert sweep.loc[1, "halflife"] < sweep.loc[21, "halflife"] < sweep.loc[63, "halflife"]


def test_ar1_reports_a_confidence_interval():
    r = st.synthetic_vol(n=8000, halflife=20.0)
    a = st.ar1_halflife(r)
    assert a["halflife_lo"] < a["halflife"] < a["halflife_hi"]


def test_ar1_declines_on_too_little_data():
    assert "halflife" not in st.ar1_halflife(st.synthetic_vol(n=150))


# --------------------------------------------------------------------------- #
# The ACF estimator
# --------------------------------------------------------------------------- #
def test_acf_halflife_grows_with_the_planted_persistence():
    a = st.acf_halflife(st.synthetic_vol(n=15000, halflife=5.0))
    b = st.acf_halflife(st.synthetic_vol(n=15000, halflife=60.0))
    assert a["halflife"] < b["halflife"]


def test_acf_is_near_zero_for_iid_returns():
    rng = np.random.default_rng(992)
    r = pd.Series(rng.normal(0, 0.01, 8000),
                  index=pd.bdate_range("1993-02-01", periods=8000))
    a = st.acf_halflife(r)
    assert abs(a["acf_1"]) < 0.06


def test_both_acf_proxies_are_available_and_agree_in_direction():
    r = st.synthetic_vol(n=15000, halflife=30.0)
    a = st.acf_halflife(r, proxy="abs")
    s = st.acf_halflife(r, proxy="sq")
    assert a["proxy"] == "abs" and s["proxy"] == "sq"
    assert a["acf_1"] > 0 and s["acf_1"] > 0


def test_acf_declines_on_a_short_series():
    assert "halflife" not in st.acf_halflife(st.synthetic_vol(n=200))


# --------------------------------------------------------------------------- #
# GARCH
# --------------------------------------------------------------------------- #
def test_garch_recovers_a_stationary_persistence():
    g = st.fit_garch11(st.synthetic_vol(n=6000, halflife=25.0))
    assert 0 < g["alpha"] < 1
    assert 0 < g["beta"] < 1
    assert g["persistence"] < 1.0


def test_garch_persistence_rises_with_the_planted_halflife():
    a = st.garch_halflife(st.synthetic_vol(n=6000, halflife=5.0))
    b = st.garch_halflife(st.synthetic_vol(n=6000, halflife=80.0))
    assert a["persistence"] < b["persistence"]


def test_garch_finds_almost_no_persistence_in_iid_returns():
    rng = np.random.default_rng(992)
    r = pd.Series(rng.normal(0, 0.01, 4000),
                  index=pd.bdate_range("1993-02-01", periods=4000))
    assert st.fit_garch11(r)["persistence"] < 0.9


def test_garch_recovers_the_unconditional_volatility():
    r = st.synthetic_vol(n=6000, halflife=20.0, base_vol=0.02)
    g = st.fit_garch11(r)
    assert g["uncond_vol"] == pytest.approx(float(r.std()), rel=0.05)


def test_garch_declines_on_too_little_data():
    assert "persistence" not in st.fit_garch11(st.synthetic_vol(n=200))


# --------------------------------------------------------------------------- #
# The impulse response
# --------------------------------------------------------------------------- #
def test_impulse_response_decays_from_a_positive_shock():
    i = st.impulse_response_halflife(st.synthetic_vol(n=15000, halflife=25.0))
    assert i["initial_excess"] > 0
    assert i["excess_at_63d"] < i["excess_at_5d"] < i["initial_excess"]


def test_impulse_halflife_grows_with_the_planted_persistence():
    a = st.impulse_response_halflife(st.synthetic_vol(n=20000, halflife=8.0))
    b = st.impulse_response_halflife(st.synthetic_vol(n=20000, halflife=80.0))
    assert a["halflife"] < b["halflife"]


def test_impulse_response_declines_without_enough_shocks():
    out = st.impulse_response_halflife(st.synthetic_vol(n=1200), shock_q=0.999)
    assert "halflife" not in out or np.isnan(out.get("halflife", np.nan))


# --------------------------------------------------------------------------- #
# Why the estimators disagree
# --------------------------------------------------------------------------- #
def test_all_estimators_roughly_agree_when_the_truth_IS_one_exponential():
    """The control: with a single AR(1) planted, the spread should be modest."""
    r = st.synthetic_vol(n=20000, halflife=20.0, second_halflife=0.0)
    t = st.halflife_table(r)
    hls = t.drop("ewma")["halflife"].dropna()
    assert hls.max() / hls.min() < 6.0


def test_the_estimators_diverge_when_the_truth_is_two_processes():
    """The study's central claim, planted and measured."""
    one = st.halflife_table(st.synthetic_vol(n=20000, halflife=20.0))
    two = st.halflife_table(st.synthetic_vol(n=20000, halflife=3.0,
                                             second_halflife=400.0, weight_fast=0.6))
    def spread(t):
        v = t.drop("ewma")["halflife"].dropna()
        return v.max() / v.min()
    assert spread(two) > spread(one)


def test_two_component_fit_recovers_planted_timescales():
    r = st.synthetic_vol(n=25000, halflife=4.0, second_halflife=150.0, weight_fast=0.5)
    f = st.two_component_fit(r)
    assert f["halflife_fast"] < f["halflife_slow"]
    assert f["halflife_slow"] > 20


def test_two_components_beat_one_when_two_are_planted():
    two = st.two_component_fit(st.synthetic_vol(n=20000, halflife=4.0,
                                                second_halflife=200.0, weight_fast=0.5))
    assert two["improvement"] > 0.1


def test_two_component_fit_declines_on_a_short_series():
    assert "weight_fast" not in st.two_component_fit(st.synthetic_vol(n=300))


def test_halflife_table_returns_all_five_methods():
    t = st.halflife_table(st.synthetic_vol(n=8000, halflife=20.0))
    assert list(t.index) == list(st.METHODS)
    assert t.loc["ewma", "halflife"] == pytest.approx(11.2, abs=0.2)


# --------------------------------------------------------------------------- #
# The practical version
# --------------------------------------------------------------------------- #
def test_volatility_after_a_shock_exceeds_volatility_after_calm():
    p = st.practical_decay(st.synthetic_vol(n=15000, halflife=40.0))
    assert (p["ratio"] > 1).all()


def test_the_advantage_decays_with_horizon():
    p = st.practical_decay(st.synthetic_vol(n=20000, halflife=20.0))
    assert p.loc[2, "ratio"] > p.loc[126, "ratio"]


def test_practical_decay_is_flat_for_iid_returns():
    rng = np.random.default_rng(992)
    r = pd.Series(rng.normal(0, 0.01, 10000),
                  index=pd.bdate_range("1993-02-01", periods=10000))
    p = st.practical_decay(r)
    assert abs(p.loc[63, "ratio"] - 1.0) < 0.15


def test_practical_decay_is_empty_on_a_short_series():
    assert st.practical_decay(st.synthetic_vol(n=400)).empty


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"asset": "SPY", "n_days": 8300, "n_assets": 7, "hl_min": 9.0, "hl_max": 88.0,
         "hl_garch": 88.0, "hl_acf": 9.0, "impulse_halflife": 24.0,
         "halflife_fast": 3.4, "halflife_slow": 121.0, "weight_fast": 0.52,
         "two_component_improvement": 0.44, "ratio_5d": 1.71, "ratio_21d": 1.44,
         "ratio_63d": 1.22, "cross_min": 12.0, "cross_max": 140.0,
         "sweep_min": 6.0, "sweep_max": 95.0}
    h.update(over)
    return h


def test_verdict_signal_needs_persistence_and_multiple_timescales():
    assert st.verdict(_headline())["signal"] == "Confirmed"
    assert st.verdict(_headline(two_component_improvement=0.05))["signal"] == "Partial"
    assert st.verdict(_headline(impulse_halflife=2.0))["signal"] == "Busted"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(ratio_21d=1.05))["trad"] == "Partial"
    assert st.verdict(_headline(ratio_21d=1.05, ratio_5d=1.05))["trad"] == "Mirage"


def test_verdict_prose_explains_the_disagreement():
    v = st.verdict(_headline())
    assert "two-component" in v["signal_why"] or "two processes" in v["one_sentence"]
    assert "window" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

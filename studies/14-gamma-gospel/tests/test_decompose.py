"""The decomposition recovers the baked-in truth: a raw regime gap that is real but VIX-driven,
a genuine gamma effect that survives the VIX control when beta > 0, and a gap that collapses to
nothing under the control when beta = 0 (the trenchcoat)."""

import numpy as np

from gamma_gospel import decompose


# --------------------------------------------------------------------------- #
# the HAC-OLS workhorse recovers a known line
# --------------------------------------------------------------------------- #

def test_hac_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    y = 2.0 + 3.0 * x + rng.normal(0, 0.1, 2000)
    X = np.column_stack([np.ones_like(x), x])
    fit = decompose.hac_ols(y, X)
    assert np.isclose(fit["beta"][0], 2.0, atol=0.02)
    assert np.isclose(fit["beta"][1], 3.0, atol=0.02)
    assert abs(fit["t"][1]) > 10          # a real slope is overwhelmingly significant
    assert fit["r2"] > 0.98


# --------------------------------------------------------------------------- #
# the raw gap is there (the pitch's headline)
# --------------------------------------------------------------------------- #

def test_raw_gap_present_for_both_outcomes(panel):
    for y in ("rv", "de"):
        g = decompose.regime_gap(panel, y)
        assert g["gap"] > 0
        assert g["mean_neg"] > g["mean_pos"]


# --------------------------------------------------------------------------- #
# beta > 0: the genuine effect survives the VIX control, and is recovered
# --------------------------------------------------------------------------- #

def test_genuine_effect_survives_vix_control(panel, truth):
    p = decompose.partial_over_vix(panel, "de")
    assert truth.beta_de > 0
    # the surviving coefficient recovers the baked-in beta_de, and stays significant
    assert np.isclose(p["surviving_coef"], truth.beta_de, atol=0.025)
    assert p["surviving_t"] > 2.0
    assert p["survival_share"] > 0.4               # a real chunk of the raw gap is not just VIX
    # controlling for VIX shrinks the gap (VIX explains part of it), but not to zero
    assert p["surviving_coef"] < p["raw_gap"]


def test_genuine_vol_effect_recovered(panel, truth):
    p = decompose.partial_over_vix(panel, "rv")
    assert np.isclose(p["surviving_coef"], truth.beta_vol, atol=0.0010)
    assert p["surviving_t"] > 2.0


# --------------------------------------------------------------------------- #
# beta == 0: the trenchcoat — raw gap present, but nothing survives VIX
# --------------------------------------------------------------------------- #

def test_mirage_gap_collapses_under_vix(mirage):
    panel, truth = mirage
    assert truth.beta_de == 0.0
    raw = decompose.regime_gap(panel, "de")
    p = decompose.partial_over_vix(panel, "de")
    assert raw["gap"] > 0.02                        # the raw effect is there (pure confound)
    assert abs(p["surviving_coef"]) < 0.025         # ...and it evaporates once VIX is controlled
    assert abs(p["surviving_t"]) < 2.5
    assert p["survival_share"] < 0.4


def test_summary_packs_both_outcomes(panel):
    s = decompose.summary(panel)
    assert set(s) >= {"n", "rv", "de"}
    assert "raw" in s["de"] and "partial" in s["de"]
    assert s["n"] > 700

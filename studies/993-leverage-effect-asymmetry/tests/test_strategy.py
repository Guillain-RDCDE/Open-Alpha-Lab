"""Strategy tests for Study 993 — asymmetry, graded against a planted gamma."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from downhurts import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The generator itself
# --------------------------------------------------------------------------- #
def test_the_generator_is_symmetric_at_gamma_zero():
    r = st.synthetic_returns(n=30000, gamma=0.0)
    assert abs(r.skew()) < 0.35


def test_a_negative_gamma_plants_a_negative_return_volatility_link():
    """Gamma acts on the volatility RESPONSE, not on the return's own skewness.

    Worth stating explicitly because it is a common confusion: an EGARCH with gamma < 0 draws
    symmetric innovations, so its unconditional skew stays near zero. What it plants is the
    *correlation between returns and subsequent volatility changes* — which is the leverage
    effect, and is what every measurement in this module targets.
    """
    sym = st.synthetic_returns(n=20000, gamma=0.0)
    asym = st.synthetic_returns(n=20000, gamma=-0.15)
    assert (st.correlation_asymmetry(asym)["corr"]
            < st.correlation_asymmetry(sym)["corr"] - 0.05)
    assert abs(asym.skew()) < 0.5          # and the skew barely moves


def test_the_generator_is_deterministic_and_seed_sensitive():
    a = st.synthetic_returns(n=2000, seed=993)
    b = st.synthetic_returns(n=2000, seed=993)
    c = st.synthetic_returns(n=2000, seed=994)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert not np.allclose(a.to_numpy(), c.to_numpy())


# --------------------------------------------------------------------------- #
# Sign split
# --------------------------------------------------------------------------- #
def test_sign_split_finds_a_planted_asymmetry():
    r = st.synthetic_returns(n=20000, gamma=-0.15)
    s = st.sign_split(r)
    assert s["ratio"] > 1.05
    assert s["vol_after_down"] > s["vol_after_up"]


def test_sign_split_finds_nothing_on_average_under_the_null():
    ratios = []
    for k in range(10):
        r = st.synthetic_returns(n=8000, gamma=0.0, seed=993 + k)
        ratios.append(st.sign_split(r)["ratio"])
    assert abs(np.mean(ratios) - 1.0) < 0.04


def test_but_a_single_null_run_can_look_asymmetric():
    """The failure mode the bootstrap exists to catch."""
    hits = 0
    for k in range(20):
        r = st.synthetic_returns(n=4000, gamma=0.0, seed=993 + k)
        s = st.sign_split(r)
        hits += abs(s["naive_t"]) >= 2
    assert hits >= 1


def test_sign_split_measures_forward_not_contemporaneous_volatility():
    """A contemporaneous comparison would find asymmetry even in i.i.d. data."""
    rng = np.random.default_rng(993)
    r = pd.Series(rng.normal(0, 0.01, 8000),
                  index=pd.bdate_range("1993-02-01", periods=8000))
    s = st.sign_split(r)
    assert abs(s["ratio"] - 1.0) < 0.05


def test_sign_split_rejects_a_one_day_horizon():
    """A one-observation sample standard deviation is undefined; say so loudly."""
    with pytest.raises(ValueError):
        st.sign_split(st.synthetic_returns(n=2000), horizon=1)


def test_sign_split_declines_on_a_short_series():
    assert "ratio" not in st.sign_split(st.synthetic_returns(n=200))


# --------------------------------------------------------------------------- #
# Magnitude matching — the control that matters
# --------------------------------------------------------------------------- #
def test_magnitude_matching_still_finds_a_planted_asymmetry():
    r = st.synthetic_returns(n=25000, gamma=-0.15)
    m = st.magnitude_matched_split(r)
    assert (m["ratio"] > 1.0).mean() >= 0.6


def test_magnitude_matching_finds_nothing_under_the_null():
    r = st.synthetic_returns(n=25000, gamma=0.0)
    m = st.magnitude_matched_split(r)
    assert abs(m["ratio"].mean() - 1.0) < 0.06


def test_magnitude_buckets_are_ordered_by_move_size():
    r = st.synthetic_returns(n=15000, gamma=-0.1)
    m = st.magnitude_matched_split(r)
    assert m["mean_abs_move"].is_monotonic_increasing


def test_magnitude_matching_is_empty_on_a_short_series():
    assert st.magnitude_matched_split(st.synthetic_returns(n=400)).empty


# --------------------------------------------------------------------------- #
# The news-impact curve
# --------------------------------------------------------------------------- #
def test_the_news_impact_curve_is_a_symmetric_parabola_under_the_null():
    r = st.synthetic_returns(n=30000, gamma=0.0)
    cm = st.curve_minimum(st.news_impact_curve(r))
    assert abs(cm["vertex_z"]) < 0.4
    assert cm["curvature"] > 0


def test_the_curve_shifts_right_when_asymmetry_is_planted():
    sym = st.curve_minimum(st.news_impact_curve(st.synthetic_returns(n=30000, gamma=0.0)))
    asym = st.curve_minimum(st.news_impact_curve(st.synthetic_returns(n=30000, gamma=-0.15)))
    assert asym["vertex_z"] > sym["vertex_z"]


def test_the_news_impact_curve_has_the_requested_resolution():
    nic = st.news_impact_curve(st.synthetic_returns(n=15000), n_bins=15)
    assert 10 <= len(nic) <= 15
    assert "mean_fwd_vol" in nic.columns


def test_curve_minimum_declines_on_too_few_bins():
    assert "vertex_z" not in st.curve_minimum(pd.DataFrame({"z": [0, 1], "mean_fwd_vol": [1, 2]}))


# --------------------------------------------------------------------------- #
# EGARCH
# --------------------------------------------------------------------------- #
def test_egarch_recovers_a_planted_gamma_sign():
    for planted in (-0.20, -0.10):
        g = st.fit_egarch(st.synthetic_returns(n=2000, gamma=planted))
        assert g["gamma"] < 0, planted


def test_egarch_gamma_is_near_zero_under_the_null():
    gammas = [st.fit_egarch(st.synthetic_returns(n=2000, gamma=0.0, seed=993 + k))["gamma"]
              for k in range(3)]
    assert abs(np.mean(gammas)) < 0.06


def test_egarch_gamma_is_ordered_by_the_planted_value():
    strong = st.fit_egarch(st.synthetic_returns(n=2500, gamma=-0.30))["gamma"]
    weak = st.fit_egarch(st.synthetic_returns(n=2500, gamma=-0.02))["gamma"]
    assert strong < weak


def test_egarch_is_stationary():
    g = st.fit_egarch(st.synthetic_returns(n=2000, gamma=-0.1))
    assert abs(g["beta"]) < 1.0


def test_egarch_asymmetry_reports_the_response_ratio():
    a = st.egarch_asymmetry(st.synthetic_returns(n=2500, gamma=-0.25))
    assert a["response_ratio"] > 1.0
    assert a["asymmetric"]


def test_egarch_declines_on_too_little_data():
    assert "gamma" not in st.fit_egarch(st.synthetic_returns(n=200))


# --------------------------------------------------------------------------- #
# Correlation and bootstrap
# --------------------------------------------------------------------------- #
def test_returns_correlate_negatively_with_volatility_changes_when_planted():
    c = st.correlation_asymmetry(st.synthetic_returns(n=20000, gamma=-0.15))
    assert c["corr"] < -0.05


def test_that_correlation_is_near_zero_under_the_null():
    cs = [st.correlation_asymmetry(st.synthetic_returns(n=10000, gamma=0.0,
                                                        seed=993 + k))["corr"]
          for k in range(6)]
    assert abs(np.mean(cs)) < 0.08


def test_correlation_asymmetry_declines_on_a_short_series():
    assert "corr" not in st.correlation_asymmetry(st.synthetic_returns(n=200))


def test_the_bootstrap_and_the_naive_test_broadly_agree_here():
    """Unlike study 989, the naive standard error is roughly right for THIS statistic.

    Worth pinning down rather than assuming, because the usual reflex is "the bootstrap is
    always wider". Two effects offset. The up-day and down-day subsamples are **interleaved in
    time** and share the prevailing volatility regime, so their difference cancels most of the
    common variation — which the block bootstrap sees and a two-sample formula does not. But
    the naive formula also charges each group its *total* variance, including that same common
    component, which makes it conservative. The two roughly cancel, and the bootstrap ends up
    within about ten percent either way.
    """
    for seed in (993, 994, 995):
        r = st.synthetic_returns(n=12000, gamma=-0.10, seed=seed)
        s = st.sign_split(r)
        b = st.bootstrap_asymmetry(r, n_boot=250)
        ratio = abs(b["t"]) / abs(s["naive_t"])
        assert 0.6 < ratio < 1.6, (seed, ratio)


def test_the_bootstrap_still_finds_a_large_planted_asymmetry():
    b = st.bootstrap_asymmetry(st.synthetic_returns(n=20000, gamma=-0.25), n_boot=300)
    assert b["t"] > 2


def test_the_bootstrap_does_not_cry_wolf_under_the_null():
    ts = [st.bootstrap_asymmetry(st.synthetic_returns(n=8000, gamma=0.0, seed=993 + k),
                                 n_boot=200)["t"] for k in range(8)]
    assert np.nanmean(np.abs(ts)) < 2.0


def test_bootstrap_declines_on_a_short_series():
    assert "t" not in st.bootstrap_asymmetry(st.synthetic_returns(n=200))


# --------------------------------------------------------------------------- #
# Which story?
# --------------------------------------------------------------------------- #
def test_lead_lag_is_symmetric_in_shape_and_covers_both_sides():
    ll = st.lead_lag_asymmetry(st.synthetic_returns(n=15000, gamma=-0.15), max_lag=8)
    assert len(ll) == 17
    assert 0 in ll.index and 8 in ll.index and -8 in ll.index


def test_a_planted_leverage_mechanism_shows_up_on_the_leverage_side():
    """The generator makes the RETURN move first by construction, so lag > 0 must dominate."""
    ll = st.lead_lag_asymmetry(st.synthetic_returns(n=25000, gamma=-0.20))
    w = st.which_story(ll)
    assert w["leverage_side"] < 0
    assert abs(w["leverage_side"]) > abs(w["feedback_side"])
    assert w["leans"] == "leverage"


def test_lead_lag_is_flat_under_the_null():
    ll = st.lead_lag_asymmetry(st.synthetic_returns(n=15000, gamma=0.0))
    assert ll["correlation"].abs().max() < 0.25


def test_which_story_handles_an_empty_table():
    assert st.which_story(pd.DataFrame()) == {}


def test_panel_runs_every_measurement_on_every_asset():
    assets = {f"A{k}": st.synthetic_returns(n=3000, gamma=-0.05 * k, seed=993 + k)
              for k in range(3)}
    p = st.panel(assets)
    assert len(p) == 3
    for c in ("ratio", "corr_r_dvol", "egarch_gamma", "vertex_z"):
        assert c in p.columns


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"asset": "SPY", "n_days": 8300, "vol_after_up": 0.146, "vol_after_down": 0.192,
         "ratio": 1.32, "matched_ratio": 1.19, "naive_t": 14.2, "boot_t": 4.1,
         "egarch_gamma": -0.086, "egarch_ratio": 1.19, "vertex_z": 0.34,
         "gold_ratio": 1.09, "crypto_ratio": 1.04, "leverage_side": -0.11,
         "feedback_side": -0.04, "leans": "leverage"}
    h.update(over)
    return h


def test_verdict_signal_needs_three_agreeing_measurements():
    assert st.verdict(_headline())["signal"] == "Confirmed"
    assert st.verdict(_headline(egarch_gamma=0.01))["signal"] == "Partial"
    assert st.verdict(_headline(egarch_gamma=0.01, boot_t=0.4))["signal"] == "Busted"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(ratio=1.10))["trad"] == "Partial"
    assert st.verdict(_headline(ratio=1.00))["trad"] == "Mirage"


def test_verdict_prose_names_the_no_balance_sheet_control():
    v = st.verdict(_headline())
    assert "balance sheet" in v["trad_why"]
    assert "gold" in v["one_sentence"] and "leverage" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}


def test_verdict_prose_switches_on_the_no_leverage_evidence():
    with_effect = st.verdict(_headline(gold_ratio=1.20))["trad_why"]
    without = st.verdict(_headline(gold_ratio=1.00, crypto_ratio=0.99))["trad_why"]
    assert "without any leverage" in with_effect
    assert "materially weaker" in without

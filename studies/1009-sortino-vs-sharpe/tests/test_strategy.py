"""Strategy tests for Study 1009 — Sortino against Sharpe, and what it costs."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sortino import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Definitions
# --------------------------------------------------------------------------- #
def test_downside_deviation_ignores_upside():
    a = np.array([-0.01, 0.02, -0.02, 0.05])
    b = np.array([-0.01, 0.20, -0.02, 0.50])
    assert st.downside_deviation(a) == pytest.approx(st.downside_deviation(b))


def test_downside_deviation_is_zero_when_nothing_falls_below_the_threshold():
    assert st.downside_deviation(np.array([0.01, 0.02, 0.03])) == pytest.approx(0.0)


def test_downside_deviation_divides_by_the_FULL_sample():
    """The convention that keeps it on the same scale as sigma. Easy to get wrong."""
    r = np.array([-0.02, 0.01, 0.01, 0.01])
    expected = np.sqrt((0.02 ** 2) / 4)
    assert st.downside_deviation(r) == pytest.approx(expected)
    conditional = np.sqrt((0.02 ** 2) / 1)
    assert st.downside_deviation(r) != pytest.approx(conditional)


def test_the_threshold_moves_the_denominator():
    r = np.array([-0.01, 0.005, 0.02, -0.03])
    assert st.downside_deviation(r, mar=0.01) > st.downside_deviation(r, mar=0.0)


def test_sharpe_matches_its_definition():
    rng = np.random.default_rng(1009)
    r = rng.normal(0.0005, 0.01, 5000)
    expected = (r.mean() / r.std(ddof=1)) * np.sqrt(252)
    assert st.sharpe(r) == pytest.approx(expected)


def test_both_ratios_decline_on_a_constant_series():
    assert np.isnan(st.sharpe(np.full(500, 0.001)))
    assert np.isnan(st.sortino(np.full(500, 0.001)))


def test_both_ratios_decline_on_a_tiny_sample():
    assert np.isnan(st.sharpe(np.array([0.01])))
    assert np.isnan(st.sortino(np.array([0.01])))


# --------------------------------------------------------------------------- #
# The identity that bounds the whole disagreement
# --------------------------------------------------------------------------- #
def test_a_symmetric_distribution_gives_exactly_sigma_over_root_two():
    """The arithmetic ceiling on how much the two ratios can ever differ."""
    x = st.synthetic_skewed(n=200000, target_skew=0.0, mean=0.0)
    assert st.symmetric_identity(x, mar=0.0)["sd_over_dd"] == pytest.approx(
        np.sqrt(2), rel=0.005)


def test_the_identity_needs_symmetry_ABOUT_THE_THRESHOLD():
    """A positive drift shifts the ratio without any skewness being involved.

    Easy to mistake for an asymmetry effect: with a zero threshold and a positive mean, more
    than half the observations sit above the line, downside deviation shrinks, and the ratio
    rises above 1.414. Setting the threshold at the mean restores the identity exactly.
    """
    x = st.synthetic_skewed(n=200000, target_skew=0.0, mean=0.0004, vol=0.011)
    assert st.symmetric_identity(x, mar=0.0)["sd_over_dd"] > 1.44
    assert st.symmetric_identity(x, mar=0.0004)["sd_over_dd"] == pytest.approx(
        np.sqrt(2), rel=0.005)


def test_sortino_is_sharpe_times_root_two_under_symmetry():
    x = st.synthetic_skewed(n=200000, target_skew=0.0, mean=0.0)
    assert st.sortino(x) / st.sharpe(x) == pytest.approx(np.sqrt(2), rel=0.01)


def test_positive_skew_makes_sortino_relatively_kinder():
    pos = st.synthetic_skewed(n=100000, target_skew=1.5)
    neg = st.synthetic_skewed(n=100000, target_skew=-1.5)
    assert st.sortino(pos) / st.sharpe(pos) > st.sortino(neg) / st.sharpe(neg)


def test_the_ratio_moves_monotonically_with_skew():
    d = st.skew_sweep(skews=(-1.5, -0.5, 0.0, 0.5, 1.5), n=40000, n_reps=4)
    assert d["sortino_over_sharpe"].is_monotonic_increasing


def test_the_control_holds_mean_and_volatility_fixed():
    """Otherwise any difference could be attributed to the first two moments."""
    means, vols = [], []
    for sk in (-1.5, 0.0, 1.5):
        x = st.synthetic_skewed(n=100000, target_skew=sk)
        means.append(x.mean())
        vols.append(x.std(ddof=1))
    assert np.std(means) / abs(np.mean(means)) < 0.02
    assert np.std(vols) / np.mean(vols) < 0.02


def test_the_control_actually_produces_the_skew_it_targets():
    for sk in (-1.0, 0.0, 1.0):
        x = st.synthetic_skewed(n=200000, target_skew=sk)
        realised = float(pd.Series(x).skew())
        assert np.sign(realised) == np.sign(sk) or abs(sk) < 1e-9
        if abs(sk) < 1e-9:
            assert abs(realised) < 0.05


def test_symmetric_identity_declines_on_a_tiny_sample():
    assert st.symmetric_identity(np.array([0.01, -0.01])) == {}


# --------------------------------------------------------------------------- #
# Real data
# --------------------------------------------------------------------------- #
def test_the_ratio_table_covers_the_panel():
    px = data.load_prices()
    R = _panel(px)
    t = st.ratio_table(R)
    assert len(t) >= 8
    assert t["sharpe"].notna().all()
    assert t["sortino"].notna().all()


def test_the_two_rankings_mostly_agree():
    px = data.load_prices()
    R = _panel(px)
    a = st.rank_agreement(st.ratio_table(R))
    assert a["spearman"] > 0.85


def test_the_rankings_are_in_fact_IDENTICAL_on_this_panel():
    """Pre-registered expecting a difference to measure. There is none at all.

    Across every asset here — including one whose skewness is about -1.8 — the two ratios
    produce the same order, position for position. The Sortino/Sharpe ratio spans a band so
    narrow that no pair ever crosses. That is a stronger result than "they mostly agree", and
    it is exactly what the arithmetic in section 1 predicts.
    """
    px = data.load_prices()
    R = _panel(px)
    a = st.rank_agreement(st.ratio_table(R))
    assert a["spearman"] == pytest.approx(1.0)
    assert a["max_rank_change"] == 0.0
    assert a["n_unchanged"] == a["n"]


def test_the_ratio_band_is_too_narrow_for_ranks_to_cross():
    """The mechanism behind the identical ranking, measured rather than asserted."""
    px = data.load_prices()
    R = _panel(px)
    t = st.ratio_table(R)
    assert t["ratio"].max() / t["ratio"].min() < 1.15
    assert t["ratio"].between(1.35, 1.55).all()


def test_rank_agreement_declines_on_a_tiny_panel():
    t = pd.DataFrame({"sharpe": [1.0, 2.0], "sortino": [1.0, 2.0],
                      "skew": [0.0, 0.0]}, index=["A", "B"])
    assert st.rank_agreement(t) == {}


def test_the_disagreement_tracks_skewness_on_real_assets():
    """The mechanism, confirmed rather than assumed."""
    px = data.load_prices()
    R = _panel(px)
    d = st.disagreement_vs_skew(R)
    assert d.attrs["corr_skew_excess"] < -0.3 or d.attrs["corr_skew_excess"] > 0.3


def test_real_assets_sit_near_the_symmetric_value():
    """Bounding how much room there is for the two ratios to differ at all."""
    px = data.load_prices()
    R = _panel(px)
    d = st.disagreement_vs_skew(R)
    assert d["sd_over_dd"].between(1.2, 1.7).mean() > 0.7


# --------------------------------------------------------------------------- #
# The cost: precision
# --------------------------------------------------------------------------- #
def test_downside_deviation_uses_only_part_of_the_sample():
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna().to_numpy()
    p = st.estimation_precision(r, n_boot=150)
    assert 0.35 < p["below_share"] < 0.55


def test_sortino_is_noisier_than_sharpe_on_identical_resamples():
    """The trade-off that never appears beside the ratio."""
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna().to_numpy()
    p = st.estimation_precision(r, n_boot=400)
    assert p["noise_ratio"] > 1.0


def test_the_noise_penalty_holds_across_the_panel():
    px = data.load_prices()
    R = _panel(px)
    ratios = []
    for c in R.columns:
        p = st.estimation_precision(R[c].dropna().to_numpy(), n_boot=150)
        if p:
            ratios.append(p["noise_ratio"])
    assert np.median(ratios) > 1.0


def test_estimation_precision_declines_on_a_short_series():
    assert st.estimation_precision(np.random.default_rng(0).normal(0, 0.01, 100)) == {}


def test_skewness_is_badly_estimated():
    """The quantity Sortino's whole case rests on."""
    px = data.load_prices()
    r = px[data.EQUITY].dropna().pct_change().dropna().to_numpy()
    s = st.skew_reliability(r, n_boot=400)
    assert s["se"] > 0.1


def test_skewness_intervals_span_zero_for_many_assets():
    px = data.load_prices()
    R = _panel(px)
    spans = []
    for c in R.columns:
        s = st.skew_reliability(R[c].dropna().to_numpy(), n_boot=200)
        if s:
            spans.append(s["spans_zero"])
    assert np.mean(spans) > 0.0


def test_skew_reliability_declines_on_a_short_series():
    assert st.skew_reliability(np.random.default_rng(0).normal(0, 0.01, 100)) == {}


def test_a_strongly_skewed_series_is_detected_as_skewed():
    """Otherwise the previous test would just mean the measurement is broken."""
    x = st.synthetic_skewed(n=20000, target_skew=1.5)
    s = st.skew_reliability(x, n_boot=200)
    assert not s["spans_zero"]
    assert s["skew"] > 0


# --------------------------------------------------------------------------- #
# The horse race
# --------------------------------------------------------------------------- #
def test_the_horse_race_produces_splits():
    px = data.load_prices()
    R = _panel(px)
    oos = st.out_of_sample_ranking(R, n_splits=6)
    assert len(oos) >= 3
    assert set(oos.columns) >= {"sharpe_predicts_sharpe", "sortino_predicts_sortino",
                                "sharpe_predicts_sortino", "sortino_predicts_sharpe"}


def test_each_metric_is_graded_on_both_scoreboards():
    """Neither ratio gets to mark its own examination paper alone."""
    px = data.load_prices()
    R = _panel(px)
    s = st.horse_race_summary(st.out_of_sample_ranking(R, n_splits=6))
    for k in ("sharpe_predicts_sharpe", "sortino_predicts_sortino",
              "sharpe_predicts_sortino", "sortino_predicts_sharpe"):
        assert -1.0 <= s[k] <= 1.0


def test_past_rankings_carry_some_information():
    """A sanity check: if nothing predicted anything the race would be meaningless."""
    px = data.load_prices()
    R = _panel(px)
    s = st.horse_race_summary(st.out_of_sample_ranking(R, n_splits=6))
    assert max(s["sharpe_predicts_sharpe"], s["sortino_predicts_sortino"]) > 0.0


def test_the_horse_race_declines_without_enough_data():
    px = data.load_prices()
    R = _panel(px)
    assert st.out_of_sample_ranking(R, n_splits=200).empty
    assert st.horse_race_summary(pd.DataFrame()) == {}


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def _panel(px):
    cols = [c for c in data.TICKERS if c not in (data.CASH,)
            and c in px.columns and px[c].dropna().shape[0] > 1500]
    return px[cols].pct_change().dropna(how="all")


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_assets": 12, "spearman": 0.951, "mean_rank_change": 0.83,
         "max_rank_change": 3.0, "biggest_mover": "GLD", "n_unchanged": 6,
         "ratio_band_lo": 1.388, "ratio_band_hi": 1.534,
         "mean_sd_over_dd": 1.4106, "corr_skew_excess": -0.71,
         "skew_spans_zero": 0.58, "below_share": 0.47,
         "sharpe_cv": 0.312, "sortino_cv": 0.361, "noise_ratio": 1.16,
         "n_splits": 5, "sortino_predicts_sortino": 0.281,
         "sharpe_predicts_sortino": 0.264, "sortino_edge": 0.017}
    h.update(over)
    return h


def test_verdict_signal_keys_off_how_much_the_rankings_move():
    assert st.verdict(_headline(mean_rank_change=1.2))["signal"] == "Real"
    assert st.verdict(_headline())["signal"] == "Weak"
    assert st.verdict(_headline(mean_rank_change=0.1))["signal"] == "None"


def test_verdict_tradability_is_decided_on_sortinos_own_scoreboard():
    assert st.verdict(_headline())["trad"] == "Partial"
    assert st.verdict(_headline(sortino_edge=0.12))["trad"] == "Useful"
    assert st.verdict(_headline(sortino_edge=-0.10))["trad"] == "Mirage"


def test_verdict_prose_states_the_arithmetic_ceiling_and_the_cost():
    v = st.verdict(_headline())
    assert "σ/√2" in v["signal_why"]
    assert "arithmetic rather than coincidence" in v["signal_why"]
    assert "least reliably estimated" in v["signal_why"]
    assert "own scoreboard" in v["trad_why"]
    assert "report both" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

"""Strategy tests for Study 991 — graded against a closed form."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from slowbell import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def test_aggregation_sums_log_returns_exactly():
    idx = pd.bdate_range("2000-01-03", periods=100)
    r = pd.Series(np.full(100, 0.01), index=idx)
    agg = st.aggregate(r, 10)
    assert len(agg) == 10
    assert agg.iloc[0] == pytest.approx(10 * np.log(1.01))


def test_a_one_day_horizon_is_just_log_returns():
    r = st.synthetic_returns(n=500)
    assert np.allclose(st.aggregate(r, 1).to_numpy(), np.log1p(r).to_numpy())


def test_non_overlapping_windows_do_not_share_days():
    r = st.synthetic_returns(n=1000)
    a = st.aggregate(r, 21, overlapping=False)
    assert len(a) == 1000 // 21


def test_overlapping_windows_give_far_more_rows():
    r = st.synthetic_returns(n=1000)
    ov = st.aggregate(r, 21, overlapping=True)
    nov = st.aggregate(r, 21, overlapping=False)
    assert len(ov) > len(nov) * 15


def test_aggregation_is_empty_when_the_horizon_exceeds_the_data():
    r = st.synthetic_returns(n=100)
    assert len(st.aggregate(r, 252)) == 0


def test_standardise_gives_zero_mean_unit_variance():
    z = st.standardise(st.synthetic_returns(n=2000))
    assert abs(z.mean()) < 1e-10
    assert z.std(ddof=1) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# The closed form — the study's spine
# --------------------------------------------------------------------------- #
def test_iid_kurtosis_prediction_is_k_over_n():
    assert st.iid_kurtosis_prediction(6.0, 1) == 6.0
    assert st.iid_kurtosis_prediction(6.0, 3) == 2.0
    assert st.iid_kurtosis_prediction(6.0, 252) == pytest.approx(6.0 / 252)


def test_iid_draws_actually_follow_the_closed_form():
    """The measurement apparatus, graded against an exact answer."""
    r = st.synthetic_returns(n=200000, df_t=6.0, clustering=0.0)
    k1 = st.excess_kurtosis(st.aggregate(r, 1))
    for hz in (5, 10, 21):
        measured = st.excess_kurtosis(st.aggregate(r, hz))
        predicted = st.iid_kurtosis_prediction(k1, hz)
        assert measured == pytest.approx(predicted, rel=0.45), hz


def test_a_normal_world_has_no_excess_kurtosis_at_any_horizon():
    r = st.synthetic_returns(n=60000, df_t=1000, clustering=0.0)
    for hz in (1, 21, 63):
        assert abs(st.excess_kurtosis(st.aggregate(r, hz))) < 0.2


def test_clustering_slows_the_decay_below_the_iid_rate():
    """The whole point of the study, as a test."""
    iid = st.synthetic_returns(n=40000, df_t=1000, clustering=0.0)
    clus = st.synthetic_returns(n=40000, df_t=1000, clustering=0.99)
    k_iid = st.excess_kurtosis(st.aggregate(iid, 21))
    k_clus = st.excess_kurtosis(st.aggregate(clus, 21))
    assert k_clus > k_iid + 0.3


# --------------------------------------------------------------------------- #
# The four measurements
# --------------------------------------------------------------------------- #
def test_excess_kurtosis_is_zero_for_a_normal_sample():
    rng = np.random.default_rng(991)
    assert abs(st.excess_kurtosis(pd.Series(rng.normal(0, 1, 200000)))) < 0.05


def test_excess_kurtosis_matches_the_student_t_formula():
    """A t(nu) has excess kurtosis 6/(nu-4) for nu > 4."""
    rng = np.random.default_rng(991)
    for nu in (8.0, 12.0):
        x = pd.Series(rng.standard_t(nu, 400000))
        assert st.excess_kurtosis(x) == pytest.approx(6 / (nu - 4), rel=0.35)


def test_jarque_bera_rejects_fat_tails_and_accepts_normals():
    rng = np.random.default_rng(991)
    assert st.jarque_bera(pd.Series(rng.standard_t(3, 5000)))["reject_5pct"]
    assert not st.jarque_bera(pd.Series(rng.normal(0, 1, 5000)))["reject_5pct"]


def test_anderson_darling_agrees_with_jarque_bera_on_clear_cases():
    rng = np.random.default_rng(991)
    fat = pd.Series(rng.standard_t(3, 5000))
    assert st.anderson_darling(fat)["reject_5pct"]
    assert not st.anderson_darling(pd.Series(rng.normal(0, 1, 5000)))["reject_5pct"]


def test_both_tests_decline_on_tiny_samples():
    x = pd.Series([1.0, 2.0, 3.0])
    assert "p_value" not in st.jarque_bera(x)
    assert "statistic" not in st.anderson_darling(x)


def test_tail_ratio_is_one_for_a_normal():
    rng = np.random.default_rng(991)
    tr = st.tail_ratio(pd.Series(rng.normal(0, 1, 400000)))
    assert tr["ratio_2sig"] == pytest.approx(1.0, abs=0.05)
    assert tr["ratio_3sig"] == pytest.approx(1.0, abs=0.2)


def test_tail_ratio_explodes_for_fat_tails():
    rng = np.random.default_rng(991)
    tr = st.tail_ratio(pd.Series(rng.standard_t(3, 100000)))
    assert tr["ratio_4sig"] > 5
    assert tr["ratio_5sig"] > tr["ratio_3sig"]


def test_tail_ratio_declines_on_a_short_sample():
    assert "ratio_3sig" not in st.tail_ratio(pd.Series(np.arange(10.0)))


# --------------------------------------------------------------------------- #
# The tail index
# --------------------------------------------------------------------------- #
def test_hill_recovers_a_planted_tail_index():
    rng = np.random.default_rng(991)
    for nu in (3.0, 5.0):
        x = pd.Series(rng.standard_t(nu, 100000))
        assert st.hill_estimator(x, 0.02)["alpha"] == pytest.approx(nu, rel=0.35)


def test_hill_says_a_normal_has_a_very_heavy_index():
    rng = np.random.default_rng(991)
    a = st.hill_estimator(pd.Series(rng.normal(0, 1, 100000)), 0.02)["alpha"]
    assert a > 5


def test_hill_flags_whether_the_moments_exist():
    rng = np.random.default_rng(991)
    heavy = st.hill_estimator(pd.Series(rng.standard_t(1.5, 100000)), 0.02)
    light = st.hill_estimator(pd.Series(rng.standard_t(8.0, 100000)), 0.02)
    assert not heavy["variance_exists"]
    assert light["variance_exists"] and light["kurtosis_exists"]


def test_hill_declines_when_the_tail_sample_is_too_small():
    assert "alpha" not in st.hill_estimator(pd.Series(np.random.normal(0, 1, 100)), 0.05)


# --------------------------------------------------------------------------- #
# The profile
# --------------------------------------------------------------------------- #
def test_the_profile_covers_every_horizon_it_can():
    r = st.synthetic_returns(n=20000, df_t=4.0)
    p = st.convergence_profile(r)
    assert list(p.index) == list(st.HORIZONS)
    assert (p["n"] > 20).all()


def test_kurtosis_falls_with_horizon_on_iid_data():
    r = st.synthetic_returns(n=100000, df_t=5.0, clustering=0.0)
    p = st.convergence_profile(r, horizons=(1, 5, 21, 63))
    assert p["excess_kurtosis"].is_monotonic_decreasing


def test_the_profile_reports_the_ratio_to_the_iid_prediction():
    r = st.synthetic_returns(n=40000, df_t=5.0, clustering=0.98)
    p = st.convergence_profile(r, horizons=(1, 21, 63))
    assert p.loc[63, "kurtosis_vs_iid"] > 1.5


def test_fit_decay_rate_recovers_the_iid_exponent_of_one():
    r = st.synthetic_returns(n=200000, df_t=5.0, clustering=0.0)
    p = st.convergence_profile(r, horizons=(1, 5, 10, 21, 63))
    assert st.fit_decay_rate(p)["exponent"] == pytest.approx(1.0, abs=0.3)


def test_fit_decay_rate_finds_a_slower_exponent_under_clustering():
    iid = st.convergence_profile(st.synthetic_returns(n=100000, df_t=5.0, clustering=0.0),
                                 horizons=(1, 5, 10, 21, 63))
    clus = st.convergence_profile(st.synthetic_returns(n=100000, df_t=5.0, clustering=0.99),
                                  horizons=(1, 5, 10, 21, 63))
    assert st.fit_decay_rate(clus)["exponent"] < st.fit_decay_rate(iid)["exponent"]


def test_fit_decay_rate_declines_on_too_few_points():
    r = st.synthetic_returns(n=5000)
    assert "exponent" not in st.fit_decay_rate(st.convergence_profile(r, horizons=(1, 21)))


def test_convergence_horizon_reports_actual_against_iid():
    r = st.synthetic_returns(n=60000, df_t=4.0, clustering=0.98)
    p = st.convergence_profile(r)
    c = st.convergence_horizon(p, threshold=0.5)
    assert c["actual_horizon"] is None or c["actual_horizon"] >= c["iid_horizon"]


def test_convergence_horizon_handles_an_empty_profile():
    assert st.convergence_horizon(pd.DataFrame()) == {}


# --------------------------------------------------------------------------- #
# Power and overlap
# --------------------------------------------------------------------------- #
def test_normality_tests_lose_power_as_the_sample_shrinks():
    p = st.power_of_normality_tests({1: 8000, 21: 380, 252: 32}, true_df=4.0, n_sims=200)
    assert p.loc[1, "power_vs_t4"] > p.loc[252, "power_vs_t4"]
    assert p.loc[252, "power_vs_t4"] < 0.7


def test_power_table_skips_hopeless_sample_sizes():
    p = st.power_of_normality_tests({252: 10}, n_sims=50)
    assert p.empty


def test_overlapping_windows_add_rows_but_not_information():
    """The tempting shortcut at long horizons, measured."""
    r = st.synthetic_returns(n=20000, df_t=4.0)
    o = st.overlap_inflation(r, 252, n_boot=200)
    assert o["apparent_gain"] > 100
    assert o["effective_gain"] < o["apparent_gain"] / 3


def test_overlap_inflation_declines_on_a_short_series():
    r = st.synthetic_returns(n=500)
    o = st.overlap_inflation(r, 252)
    assert "effective_gain" not in o


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"asset": "SPY", "n_days": 8300, "kurtosis_1d": 12.4, "kurtosis_longest": 0.61,
         "longest_horizon": 252, "iid_at_longest": 0.049, "decay_exponent": 0.62,
         "decay_se": 0.06, "decay_t_vs_one": -6.3, "hill_alpha": 3.1,
         "actual_horizon": 252, "iid_horizon": 21, "slowdown": 12.0,
         "ratio_3sig_longest": 1.4, "n_at_longest": 32, "power_at_longest": 0.22}
    h.update(over)
    return h


def test_verdict_signal_needs_decay_and_a_slower_than_iid_rate():
    assert st.verdict(_headline())["signal"] == "Confirmed"
    assert st.verdict(_headline(decay_t_vs_one=-0.5))["signal"] == "Partial"
    assert st.verdict(_headline(kurtosis_longest=15.0))["signal"] == "Busted"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(ratio_3sig_longest=4.0))["trad"] == "Partial"
    assert st.verdict(_headline(actual_horizon=None,
                                ratio_3sig_longest=4.0))["trad"] == "Mirage"


def test_verdict_prose_names_the_closed_form_and_the_power_problem():
    v = st.verdict(_headline())
    assert "i.i.d." in v["signal_why"]
    assert "power" in v["trad_why"]
    assert "exponent" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}


def test_verdict_switches_its_kurtosis_caveat_on_the_tail_index():
    below = st.verdict(_headline(hill_alpha=3.1))["signal_why"]
    above = st.verdict(_headline(hill_alpha=5.5))["signal_why"]
    assert "may not exist" in below
    assert "well-defined" in above

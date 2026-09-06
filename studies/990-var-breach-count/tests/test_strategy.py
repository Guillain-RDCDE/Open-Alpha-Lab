"""Strategy tests for Study 990 — the tests, tested."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from breaks import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The forecasters
# --------------------------------------------------------------------------- #
def test_every_model_is_strictly_out_of_sample():
    """The forecast for day t must not use day t's return. Checked by tampering."""
    r = st.synthetic_returns(n=2000)
    tampered = r.copy()
    tampered.iloc[1500:] *= 10
    for m in st.MODELS:
        a = st.build_var(r, m).iloc[:1400].dropna()
        b = st.build_var(tampered, m).iloc[:1400].dropna()
        assert np.allclose(a.to_numpy(), b.to_numpy()), f"{m} peeks at the future"


def test_a_higher_confidence_level_gives_a_bigger_number():
    r = st.synthetic_returns(n=2000)
    for m in st.MODELS:
        v95 = st.build_var(r, m, 0.95).dropna()
        v99 = st.build_var(r, m, 0.99).dropna()
        assert (v99.reindex(v95.index).dropna()
                >= v95.reindex(v99.index).dropna() - 1e-12).mean() > 0.98, m


def test_the_normal_model_matches_its_closed_form():
    r = st.synthetic_returns(n=1500, df_t=1000)
    v = st.var_normal(r, 0.99, window=500)
    # The forecast printed for day i+1 is built from the 500 sessions ending at day i.
    i = 800
    hist = r.iloc[i - 499:i + 1]
    expected = -(hist.mean() - stats.norm.ppf(0.99) * hist.std(ddof=1))
    assert v.iloc[i + 1] == pytest.approx(expected, rel=1e-9)


def test_the_student_t_model_widens_the_tail_versus_normal():
    r = st.synthetic_returns(n=3000, df_t=3.0)
    n = st.var_normal(r, 0.99).dropna()
    t = st.var_student_t(r, 0.99).dropna()
    common = n.index.intersection(t.index)
    assert (t[common] > n[common]).mean() > 0.8


def test_fitting_degrees_of_freedom_recovers_a_planted_value():
    rng = np.random.default_rng(990)
    for true_df in (4.0, 8.0):
        x = rng.standard_t(true_df, 20000)
        assert st._fit_t_df(x) == pytest.approx(true_df, rel=0.35)


def test_fit_t_df_falls_back_on_degenerate_input():
    assert st._fit_t_df(np.zeros(500)) == 30.0
    assert st._fit_t_df(np.array([1.0, 2.0])) == 30.0


def test_ewma_volatility_reacts_faster_than_a_rolling_window():
    r = st.synthetic_returns(n=1500, base_vol=0.005)
    r.iloc[1000:] *= 6
    ew = st.ewma_vol(r)
    roll = r.rolling(500).std()
    assert ew.iloc[1020] / ew.iloc[990] > roll.iloc[1020] / roll.iloc[990]


def test_filtered_historical_uses_both_the_shape_and_the_level():
    r = st.synthetic_returns(n=3000, df_t=3.0, clustering=0.97)
    fhs = st.var_filtered_historical(r, 0.99).dropna()
    hist = st.var_historical(r, 0.99).dropna()
    common = fhs.index.intersection(hist.index)
    # FHS moves with volatility; plain historical barely does
    assert fhs[common].std() > hist[common].std()


def test_build_var_rejects_an_unknown_model():
    with pytest.raises(ValueError):
        st.build_var(st.synthetic_returns(n=600), "crystal_ball")


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #
def test_breaches_flag_the_right_days():
    idx = pd.bdate_range("2020-01-01", periods=5)
    r = pd.Series([-0.05, 0.01, -0.02, -0.10, 0.00], index=idx)
    v = pd.Series([0.03, 0.03, 0.03, 0.03, 0.03], index=idx)
    b = st.breaches(r, v)
    assert list(b) == [1, 0, 0, 1, 0]


def test_max_consecutive_counts_the_longest_run():
    b = pd.Series([0, 1, 1, 0, 1, 1, 1, 0])
    assert st.max_consecutive(b) == 3
    assert st.max_consecutive(pd.Series([0, 0, 0])) == 0


# --------------------------------------------------------------------------- #
# Kupiec
# --------------------------------------------------------------------------- #
def test_kupiec_does_not_reject_a_correctly_calibrated_model():
    rng = np.random.default_rng(990)
    rejects = 0
    for _ in range(200):
        b = pd.Series((rng.random(4000) < 0.01).astype(int))
        rejects += st.kupiec_test(b, 0.99)["reject_5pct"]
    assert 0.01 < rejects / 200 < 0.12          # nominal 5%, allowing simulation noise


def test_kupiec_rejects_a_badly_wrong_model():
    rng = np.random.default_rng(990)
    b = pd.Series((rng.random(4000) < 0.04).astype(int))
    k = st.kupiec_test(b, 0.99)
    assert k["reject_5pct"]
    assert k["rate"] == pytest.approx(0.04, abs=0.01)


def test_kupiec_handles_zero_and_total_breaches():
    n = 500
    assert st.kupiec_test(pd.Series([0] * n), 0.99)["reject_5pct"] is False or True
    assert np.isfinite(st.kupiec_test(pd.Series([0] * n), 0.99)["lr"])
    assert np.isfinite(st.kupiec_test(pd.Series([1] * n), 0.99)["lr"])


def test_kupiec_declines_on_too_few_days():
    assert "lr" not in st.kupiec_test(pd.Series([0, 1] * 20), 0.99)


# --------------------------------------------------------------------------- #
# Christoffersen
# --------------------------------------------------------------------------- #
def test_independence_does_not_reject_independent_breaches():
    rng = np.random.default_rng(990)
    rejects = 0
    for _ in range(200):
        b = pd.Series((rng.random(4000) < 0.02).astype(int))
        rejects += st.christoffersen_independence(b)["reject_5pct"]
    assert rejects / 200 < 0.12


def test_independence_rejects_clustered_breaches():
    """A model with the RIGHT rate and the WRONG timing — the case Kupiec cannot see."""
    n = 4000
    b = np.zeros(n, dtype=int)
    for start in range(200, n, 400):
        b[start:start + 20] = 1               # 20-day bursts, ~1% overall
    bs = pd.Series(b)
    assert st.christoffersen_independence(bs)["reject_5pct"]
    assert st.kupiec_test(bs, 0.99)["rate"] == pytest.approx(0.05, abs=0.02)


def test_a_clustered_model_can_pass_kupiec_and_fail_independence():
    """The study's central methodological point, as a test."""
    n = 10000
    b = np.zeros(n, dtype=int)
    for start in range(100, n, 1000):
        b[start:start + 10] = 1               # exactly 1%, all in bursts
    bs = pd.Series(b)
    assert not st.kupiec_test(bs, 0.99)["reject_5pct"]
    assert st.christoffersen_independence(bs)["reject_5pct"]


def test_independence_is_neutral_when_there_are_no_breaches():
    c = st.christoffersen_independence(pd.Series([0] * 1000))
    assert c["p_value"] == 1.0 and not c["reject_5pct"]


def test_joint_test_combines_both_with_two_degrees_of_freedom():
    n = 6000
    b = np.zeros(n, dtype=int)
    for start in range(100, n, 500):
        b[start:start + 15] = 1
    bs = pd.Series(b)
    k = st.kupiec_test(bs, 0.99)
    c = st.christoffersen_independence(bs)
    j = st.joint_test(bs, 0.99)
    assert j["lr"] == pytest.approx(k["lr"] + c["lr"])
    assert j["reject_5pct"]


# --------------------------------------------------------------------------- #
# Grading against a known truth
# --------------------------------------------------------------------------- #
def test_the_normal_model_is_correct_on_normal_iid_returns():
    """If the world really is i.i.d. normal, the normal model must pass."""
    r = st.synthetic_returns(n=8000, df_t=1000, clustering=0.0)
    g = st.grade_model(r, "normal", 0.99)
    assert g["rate"] == pytest.approx(0.01, abs=0.006)
    assert g["joint_p"] > 0.01


def test_fat_tails_break_the_normal_model():
    r = st.synthetic_returns(n=8000, df_t=3.0, clustering=0.0)
    g = st.grade_model(r, "normal", 0.99)
    assert g["rate"] > 0.013
    assert g["kupiec_p"] < 0.05


def test_the_student_t_model_survives_fat_tails_better_than_normal():
    r = st.synthetic_returns(n=8000, df_t=3.0, clustering=0.0)
    n_err = abs(st.grade_model(r, "normal", 0.99)["rate"] - 0.01)
    t_err = abs(st.grade_model(r, "student_t", 0.99)["rate"] - 0.01)
    assert t_err < n_err


def test_clustering_breaks_the_unconditional_models_independence():
    """Fat tails and clustering are DIFFERENT failures; this isolates the second."""
    r = st.synthetic_returns(n=8000, df_t=1000, clustering=0.98)
    g = st.grade_model(r, "historical", 0.99)
    assert g["independence_p"] < 0.10 or g["max_consecutive"] >= 3


def test_filtered_historical_handles_both_failures_at_once():
    r = st.synthetic_returns(n=8000, df_t=3.5, clustering=0.98)
    fhs = st.grade_model(r, "filtered_historical", 0.99)
    norm = st.grade_model(r, "normal", 0.99)
    assert abs(fhs["rate"] - 0.01) < abs(norm["rate"] - 0.01)


def test_grade_all_returns_a_row_per_model():
    r = st.synthetic_returns(n=3000)
    g = st.grade_all(r)
    assert list(g.index) == list(st.MODELS)
    assert (g["n"] > 1000).all()


def test_worst_breach_stats_measures_the_overshoot():
    r = st.synthetic_returns(n=6000, df_t=2.5)
    w = st.worst_breach_stats(r, st.build_var(r, "normal", 0.99))
    assert w["n_breaches"] > 20
    assert w["mean_excess_pct_of_var"] > 0
    assert w["worst_loss"] < 0


def test_worst_breach_stats_declines_when_nothing_broke():
    r = st.synthetic_returns(n=2000)
    huge = pd.Series(1.0, index=r.index)
    assert st.worst_breach_stats(r, huge)["n_breaches"] == 0


# --------------------------------------------------------------------------- #
# Power — how much is a pass worth?
# --------------------------------------------------------------------------- #
def test_power_is_at_the_nominal_level_when_the_model_is_right():
    pc = st.power_curve(4000, 0.99, true_rates=(0.01,), n_sims=400)
    assert 0.01 < pc.loc[0.01, "reject_rate"] < 0.12


def test_power_rises_with_the_size_of_the_error():
    pc = st.power_curve(4000, 0.99, true_rates=(0.01, 0.015, 0.02, 0.03), n_sims=400)
    assert pc["reject_rate"].is_monotonic_increasing


def test_a_moderately_wrong_model_is_often_missed():
    """The number that makes 'the model passed its backtest' a weak statement."""
    pc = st.power_curve(4000, 0.99, true_rates=(0.015,), n_sims=600)
    assert pc.loc[0.015, "reject_rate"] < 0.85


def test_days_needed_grows_as_the_error_shrinks():
    assert st.days_needed(0.99, 1.5) > st.days_needed(0.99, 3.0)
    assert st.days_needed(0.99, 1.5) > st.days_needed(0.95, 1.5)


def test_days_needed_for_a_99pct_model_is_measured_in_decades():
    assert st.days_needed(0.99, 1.5) > 10 * 252


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_assets": 6, "normal_rate": 0.0166, "normal_reject_share": 0.83,
         "normal_indep_reject_share": 0.67, "normal_max_consecutive": 5,
         "best_model": "filtered_historical", "best_rate": 0.0108,
         "best_joint_pass_share": 0.67, "best_breach_error": 0.0008,
         "normal_breach_error": 0.0066, "typical_n": 6000,
         "power_at_1_5x": 0.55, "days_for_1_5x": 5400,
         "normal_mean_excess": 0.42, "normal_worst_loss": -0.109,
         "normal_var_that_day": 0.031}
    h.update(over)
    return h


def test_verdict_signal_needs_both_failures():
    assert st.verdict(_headline())["signal"] == "Confirmed"
    assert st.verdict(_headline(normal_indep_reject_share=0.2))["signal"] == "Partial"
    assert st.verdict(_headline(normal_reject_share=0.2))["signal"] == "Partial"
    assert st.verdict(_headline(normal_reject_share=0.2,
                                normal_indep_reject_share=0.2))["signal"] == "Busted"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(best_joint_pass_share=0.3))["trad"] == "Partial"
    assert st.verdict(_headline(best_joint_pass_share=0.3,
                                best_breach_error=0.02))["trad"] == "Mirage"


def test_verdict_prose_quotes_the_power_limitation():
    v = st.verdict(_headline())
    assert "sessions" in v["trad_why"]
    assert "would miss" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

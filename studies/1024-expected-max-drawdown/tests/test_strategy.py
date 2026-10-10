"""Strategy tests for Study 1024 — the drawdown promise, its challengers, and the verdict."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from promised import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="bundled tapes not available — synthetic tests cover the logic")


# --------------------------------------------------------------------------- #
# Drawdown arithmetic
# --------------------------------------------------------------------------- #
def test_max_drawdown_on_a_hand_path():
    # log path: 0, .1, -.1, -.05, -.15 -> peak .1, trough -.15 -> D = .25
    r = np.array([0.1, -0.2, 0.05, -0.1])
    assert st.max_drawdown_log(r) == pytest.approx(0.25)
    assert st.max_drawdown(r) == pytest.approx(1 - np.exp(-0.25))


def test_a_rising_path_has_no_drawdown_and_a_first_day_loss_counts():
    assert st.max_drawdown(np.full(10, 0.01)) == 0.0
    assert st.max_drawdown_log(np.array([-0.05, 0.2])) == pytest.approx(0.05)


def test_max_drawdown_is_vectorised_over_rows():
    r = np.array([[0.1, -0.2], [-0.1, 0.1]])
    assert np.allclose(st.max_drawdown_log(r), [0.2, 0.1])


def test_drawdown_curve_matches_max_drawdown():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500))
    assert st.drawdown_curve(r).max() == pytest.approx(st.max_drawdown(r.to_numpy()))


# --------------------------------------------------------------------------- #
# The promise
# --------------------------------------------------------------------------- #
def test_fine_sampled_gbm_converges_on_the_magdon_ismail_zero_drift_mean():
    sig = 0.2
    m = st.gbm_mdd(0.0, sig / np.sqrt(252 * 8), 252 * 8, 2000, seed=11)
    mean_log = float(np.mean(-np.log1p(-m)))
    assert mean_log == pytest.approx(st.magdon_ismail_zero_drift_mean(sig, 1.0), rel=0.03)


def test_coarser_sampling_hides_drawdown():
    d = st.discretisation_table(0.0, 0.16, 1.0, steps_per_year=(12, 52, 252), n_sims=2000)
    assert d.loc[12, "mean"] < d.loc[52, "mean"] < d.loc[252, "mean"]
    assert (d["mean_log"] < d["continuous_formula_log"]).all()


def test_drift_and_volatility_move_the_promise_the_right_way():
    base = st.summary(st.gbm_mdd(0.0, 0.01, 252))
    up = st.summary(st.gbm_mdd(0.001, 0.01, 252))
    wild = st.summary(st.gbm_mdd(0.0, 0.02, 252))
    assert up["mean"] < base["mean"] < wild["mean"]
    assert base["median"] < base["q95"] < base["q99"]


def test_simulated_quantile_is_calibrated_against_fresh_paths():
    mu, s = 0.06 / 252, 0.16 / np.sqrt(252)
    q = np.quantile(st.gbm_mdd(mu, s, 252, 4000, seed=1024), 0.95)
    fresh = st.max_drawdown(mu + s * np.random.default_rng(99).standard_normal((8000, 252)))
    assert np.mean(fresh > q) == pytest.approx(0.05, abs=0.01)


# --------------------------------------------------------------------------- #
# Challengers
# --------------------------------------------------------------------------- #
def test_stationary_bootstrap_indices_are_valid_with_the_right_block_length():
    idx = st.stationary_bootstrap_index(500, 400, 300, 20.0, np.random.default_rng(0))
    assert idx.shape == (300, 400) and idx.min() >= 0 and idx.max() < 500
    cont = np.mean(np.diff(idx, axis=1) % 500 == 1)
    assert cont == pytest.approx(1 - 1 / 20.0, abs=0.01)


def test_bootstrap_of_iid_gaussian_matches_the_gaussian_promise():
    rng = np.random.default_rng(5)
    r = rng.normal(0.0002, 0.01, 2500)
    b = st.summary(st.bootstrap_mdd(r, 252, 2000, mean_block=5.0, seed=1))
    g = st.summary(st.gbm_mdd(r.mean(), r.std(ddof=1), 252, 2000))
    assert b["q95"] == pytest.approx(g["q95"], rel=0.1)


def test_garch_fit_finds_the_planted_clustering():
    r, _ = data.synthetic_returns(n_years=10, signal_strength=1.0, seed=3)
    p = st.fit_garch_t(r.to_numpy())
    assert p["ok"] and p["alpha"] + p["beta"] > 0.9 and p["nu"] < 10


def test_garch_fit_falls_back_gracefully():
    p = st.fit_garch_t(np.zeros(300))
    assert not p["ok"] or np.isfinite(p["s2_next"])


def test_garch_without_dynamics_is_the_gaussian():
    p = {"mu": 0.0, "omega": 1e-4, "alpha": 0.0, "beta": 0.0, "nu": np.inf, "s2_next": 1e-4}
    a = st.summary(st.garch_t_mdd(p, 252, 2000, seed=3))
    g = st.summary(st.gbm_mdd(0.0, 0.01, 252, 2000))
    assert a["mean"] == pytest.approx(g["mean"], rel=0.05)


def test_starting_from_a_storm_promises_a_deeper_drawdown():
    calm = {"mu": 0.0, "omega": 2e-6, "alpha": 0.08, "beta": 0.9, "nu": 6.0, "s2_next": 2e-5}
    storm = dict(calm, s2_next=1e-3)
    assert st.summary(st.garch_t_mdd(storm, 252, 1000))["median"] > \
        st.summary(st.garch_t_mdd(calm, 252, 1000))["median"]


def test_promise_rejects_unknown_models_and_oracle_needs_moments():
    with pytest.raises(KeyError):
        st.promise(np.zeros(10) + 0.01, 10, "nope", 252)
    with pytest.raises(ValueError):
        st.promise(np.random.default_rng(0).normal(0, .01, 100), 10, "gauss_oracle", 252)


# --------------------------------------------------------------------------- #
# Windows: no look-ahead
# --------------------------------------------------------------------------- #
def test_calendar_windows_are_complete_disjoint_and_strictly_after_estimation():
    r, _ = data.synthetic_returns(n_years=20, seed=1)
    r = r[r.index < "1969-07-01"]          # last year partial -> dropped
    ws = st.calendar_windows(r.index, 3, 5)
    years = r.index.year
    prev_end = -1
    for start, est, win in ws:
        assert years[est].max() == start - 1 and years[est].min() == start - 5
        assert years[win].min() == start and years[win].max() == start + 2
        assert start > prev_end
        prev_end = start + 2
    assert prev_end <= 1968


def test_overlapping_and_expanding_windows():
    r, _ = data.synthetic_returns(n_years=20, seed=1)
    ov = st.calendar_windows(r.index, 3, 5, step_years=1)
    assert [w[0] for w in ov][:3] == [1955, 1956, 1957]
    ex = st.calendar_windows(r.index, 1, "expanding")
    assert ex[0][1].sum() < ex[-1][1].sum()


def test_the_promise_never_sees_its_own_window():
    r, _ = data.synthetic_returns(n_years=14, signal_strength=0.0, seed=2)
    a = st.coverage(r, "x", 252, (1,), 5, ("gauss",), n_sims=500)
    r2 = r.copy()
    r2[r2.index.year == 1960] = -0.05     # a crash inside one window only
    b = st.coverage(r2, "x", 252, (1,), 5, ("gauss",), n_sims=500)
    row_a = a[a["start"] == 1960].iloc[0]
    row_b = b[b["start"] == 1960].iloc[0]
    assert row_a["p_q95"] == pytest.approx(row_b["p_q95"])      # promise unchanged
    assert row_b["realised"] > row_a["realised"] and row_b["breach95"]


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def test_binomial_p_directions():
    hi = st.binomial_p(10, 40)
    lo = st.binomial_p(0, 200)
    assert hi["p_greater"] < 0.01 and hi["p_less"] > 0.99
    assert lo["p_less"] < 0.01
    assert st.binomial_p(2, 40)["p_greater"] > 0.3


def test_multiplier_is_one_when_calibrated_and_scales_with_the_overrun():
    rng = np.random.default_rng(0)
    q = np.full(2000, 0.2)
    real = rng.uniform(0, 0.2 / 0.95, 2000)          # 5% above q
    assert st.multiplier(real, q) == pytest.approx(1.0, abs=0.02)
    assert st.multiplier(real * 1.5, q) == pytest.approx(1.5, abs=0.03)
    lo, hi = st.multiplier_ci(real[:200], q[:200], n_boot=200)
    assert lo < hi


def test_block_bootstrap_rate_on_a_null_series_is_quiet():
    b = (np.random.default_rng(1).random(400) < 0.05).astype(float)
    out = st.block_bootstrap_rate(b, block=5, n_boot=1000)
    assert out["p_greater"] > 0.05 and out["lo"] <= 0.05 <= out["hi"] + 0.02
    loud = st.block_bootstrap_rate(np.r_[np.ones(60), np.zeros(140)], block=5, n_boot=1000)
    assert loud["p_greater"] < 0.01


def test_summarise_counts_and_pit_ks():
    df = pd.DataFrame({"tape": "t", "horizon": 1, "model": "gauss",
                       "breach95": [True, False, False, False], "breach99": False,
                       "ratio_mean": 1.0, "pit": [0.99, 0.2, 0.5, 0.7],
                       "realised": [0.3, 0.1, 0.1, 0.1], "p_q95": 0.2})
    s = st.summarise(df).iloc[0]
    assert s["n"] == 4 and s["breaches95"] == 1 and s["rate95"] == 0.25
    assert st.ks_uniform_pit(np.linspace(0.01, 0.99, 50)) > 0.5


# --------------------------------------------------------------------------- #
# Synthetic: calibrated on the null, fires on the planted world
# --------------------------------------------------------------------------- #
def test_on_iid_gaussian_the_true_moment_formula_is_calibrated(null_cov):
    g = null_cov[null_cov["model"] == "gauss_oracle"]
    rate = g["breach95"].mean()
    assert 0.02 <= rate <= 0.08
    assert st.binomial_p(int(g["breach95"].sum()), len(g))["p_two"] > 0.05


def test_on_iid_gaussian_the_plugin_stays_quiet(null_cov):
    g = null_cov[null_cov["model"] == "gauss"]
    assert g["breach95"].mean() < 0.08
    assert st.binomial_p(int(g["breach95"].sum()), len(g))["p_greater"] > 0.05


def test_with_clustering_and_fat_tails_the_plugin_breaches_too_often(planted_cov, null_cov):
    g = planted_cov[planted_cov["model"] == "gauss"]
    assert g["breach95"].mean() > 0.10
    assert st.binomial_p(int(g["breach95"].sum()), len(g))["p_greater"] < 0.01
    g0 = null_cov[null_cov["model"] == "gauss"]
    assert g["breach95"].mean() > g0["breach95"].mean() + 0.05
    assert st.multiplier(g["realised"], g["p_q95"]) > 1.2


def test_with_clustering_the_typical_year_is_calmer_than_promised(planted_cov):
    """Wrong shape, not wrong scale: the median realised drawdown is BELOW the promise."""
    g = planted_cov[planted_cov["model"] == "gauss"]
    assert g["ratio_mean"].median() < 0.9


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def _h(rates=(0.22, 0.13, 0.15), p_hi=(0.002, 0.17, 0.001), p_lo=(1.0, 0.96, 1.0),
       null=0.058, k_range=1.87, fixes=(False, False)):
    prim = {t: {"rate": r, "breaches": int(r * 30), "n": 30, "p_greater": a, "p_less": b}
            for t, r, a, b in zip(data.INDEX_TAPES, rates, p_hi, p_lo)}
    return {"primary": prim, "null_oracle_rate": null, "null_plugin_rate": 0.077,
            "k95_range": k_range, "k95_min": 1.0, "k95_max": k_range, "k95_pooled": 1.6,
            "fixes": {"boot": fixes[0], "garch_t": fixes[1]},
            "tape_names": {"sp500": "S&P 500", "nasdaq": "Nasdaq", "ff_market": "FF market"},
            "disc_err": 0.01, "median_ratio": 0.68, "oracle_rate": 0.15, "pooled_rate": 0.16,
            "stocks_rate": 0.12, "stocks_k95_lo": 1.0, "stocks_k95_hi": 1.6,
            "boot_rate": 0.15, "garch_rate": 0.14, "gauss_mu0_pooled_rate": 0.08,
            "k95_mu0_pooled": 1.19}


def test_verdict_signal_needs_two_tapes_and_a_calibrated_null():
    assert st.verdict(_h())["signal"] == "Real"
    assert st.verdict(_h(null=0.12))["signal"] == "Weak"
    assert st.verdict(_h(p_hi=(0.002, 0.3, 0.3)))["signal"] == "Weak"
    assert st.verdict(_h(p_hi=(0.3, 0.3, 0.3)))["signal"] == "None"
    assert st.verdict(_h(p_hi=(0.002, 0.3, 0.3), p_lo=(1.0, 0.01, 0.9)))["signal"] == "Mixed"


def test_verdict_tradability_needs_a_stable_multiplier_or_a_working_model():
    assert st.verdict(_h())["trad"] == "Mirage"
    assert st.verdict(_h(k_range=1.3))["trad"] == "Fragile"
    assert st.verdict(_h(fixes=(False, True)))["trad"] == "Fragile"
    for kw in ({}, {"k_range": 1.1}, {"fixes": (True, True)}):
        assert st.verdict(_h(**kw))["trad"] != "Investable"


def test_verdict_prose_names_the_numbers():
    v = st.verdict(_h())
    assert "16%" in v["one_sentence"] and "5%" in v["one_sentence"]
    assert "S&P 500" in v["signal_why"] and "1.87" in v["trad_why"]


# --------------------------------------------------------------------------- #
# Real tape (gated)
# --------------------------------------------------------------------------- #
@needs_real
def test_real_sp500_gaussian_band_is_breached_too_often():
    r = data.load_sp500()
    c = st.coverage(r, "sp500", 252, (1,), 5, ("gauss",), n_sims=2000)
    assert len(c) == 27
    bp = st.binomial_p(int(c["breach95"].sum()), len(c))
    assert c["breach95"].mean() > 0.10 and bp["p_greater"] < 0.05
    assert c["ratio_mean"].median() < 1.0          # yet the typical year is calmer

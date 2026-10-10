"""Strategy tests for Study 1018 — is the crisis rise in correlation genuine or an artefact?"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from goestoone import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="skfolio tapes not cached — synthetic tests cover logic")


# --------------------------------------------------------------------------- #
# Correlation plumbing
# --------------------------------------------------------------------------- #
def test_corr_matrix_matches_numpy():
    X = np.random.default_rng(0).normal(size=(500, 5))
    assert np.allclose(st.corr_matrix(X), np.corrcoef(X.T))


def test_avg_pairwise_corr_recovers_a_known_level():
    rng = np.random.default_rng(1)
    f = rng.normal(size=20000)
    X = np.sqrt(0.4) * f[:, None] + np.sqrt(0.6) * rng.normal(size=(20000, 6))
    assert st.avg_pairwise_corr(X) == pytest.approx(0.4, abs=0.02)
    assert np.isnan(st.avg_pairwise_corr(X[:10]))         # too few days -> no number


# --------------------------------------------------------------------------- #
# The artefact in closed form
# --------------------------------------------------------------------------- #
def test_bgl_identity_and_direction():
    assert st.bgl_conditional_corr(0.3, 1.0) == pytest.approx(0.3)
    assert st.bgl_conditional_corr(0.3, 10.0) > st.bgl_conditional_corr(0.3, 4.0) > 0.3
    assert st.bgl_conditional_corr(0.3, 0.25) < 0.3


def test_bgl_formula_matches_selection_on_x_by_simulation():
    """Pick high-|x| days from a bivariate normal: the measured correlation rises exactly as
    the closed form says, though the true relationship never changed."""
    rng = np.random.default_rng(1018)
    n, rho = 400_000, 0.3
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    sel = np.abs(x) > 1.5
    k = x[sel].var() / x.var()
    measured = np.corrcoef(x[sel], y[sel])[0, 1]
    assert measured == pytest.approx(st.bgl_conditional_corr(rho, k), abs=0.01)
    assert measured > rho + 0.15


def test_fr_adjust_inverts_bgl():
    for rho in (0.1, 0.3, 0.6):
        for k in (2.0, 5.0, 12.0):
            assert st.fr_adjust(st.bgl_conditional_corr(rho, k), k - 1.0) == pytest.approx(rho)


# --------------------------------------------------------------------------- #
# Regimes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["past21", "past63", "drawdown"])
def test_past_only_labels_ignore_the_day_itself(null_panel, kind):
    _, m, _ = null_panel
    base = st.regime_labels(m, kind)
    t = 1500
    m2 = m.copy()
    m2.iloc[t] = -0.15                                     # a crash on day t...
    lab2 = st.regime_labels(m2, kind)
    # ...cannot change day t's label (the state variable for t stops at t-1). The full-sample
    # quantile thresholds move a hair, so only day t itself is compared.
    assert np.nan_to_num(base.iloc[t], nan=-9) == np.nan_to_num(lab2.iloc[t], nan=-9)


def test_contemporaneous_labels_do_see_the_day(null_panel):
    _, m, _ = null_panel
    t = 1500
    m2 = m.copy()
    m2.iloc[t] = -0.25
    assert st.regime_labels(m2, "big_down").iloc[t] == 1.0


def test_regime_shares_are_as_specified(null_panel):
    _, m, _ = null_panel
    lab = st.regime_labels(m, "past21")
    assert (lab == 1).mean() == pytest.approx(0.10, abs=0.01)
    assert (lab == 0).mean() == pytest.approx(0.50, abs=0.02)


def test_unknown_regime_is_refused(null_panel):
    with pytest.raises(KeyError):
        st.regime_labels(null_panel[1], "astrology")


# --------------------------------------------------------------------------- #
# Constant-correlation GARCH benchmark
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def null_fit(null_panel):
    R, m, _ = null_panel
    return st.fit_ccc_garch(R, m)


def test_ccc_fit_recovers_the_constant_correlation(null_fit):
    assert null_fit["avg_corr_stocks"] == pytest.approx(0.30, abs=0.04)
    assert np.allclose(np.diag(null_fit["corr"]), 1.0)
    pers = [p["alpha"] + p["beta"] for p in null_fit["params"]]
    assert np.median(pers) > 0.9                           # it found the volatility clustering


def test_ccc_simulation_is_deterministic_and_constant_correlation(null_fit):
    a, ma = st.simulate_ccc_garch(null_fit, seed=5)
    b, mb = st.simulate_ccc_garch(null_fit, seed=5)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    rc = st.regime_corr(a, st.regime_labels(ma, "past21"))
    assert abs(rc["diff"]) < 0.06                          # past-only: no artefact


def test_contemporaneous_conditioning_manufactures_correlation(null_fit):
    """The core artefact: in a constant-correlation world, selecting high-vol MONTHS by
    their own volatility reports a positive gap; selecting by PAST volatility does not."""
    art = st.artefact_benchmark(null_fit, kinds=("past21", "month_vol"), n_sims=6,
                                exceed=False)
    assert art["regimes"]["month_vol"]["diff_mean"] > 0.02
    assert abs(art["regimes"]["past21"]["diff_mean"]) < 0.02
    assert art["regimes"]["month_vol"]["diff_mean"] > art["regimes"]["past21"]["diff_mean"]


# --------------------------------------------------------------------------- #
# The detector — fires on the plant, quiet on the null
# --------------------------------------------------------------------------- #
def test_detector_fires_on_the_planted_world(planted):
    R, m, _ = planted
    d = st.crisis_correlation_test(R, m, n_sims=6, n_boot=120)
    assert d["significant"] and d["ci_lo"] > 0
    assert d["genuine"] > 0.10
    assert abs(d["artefact"]) < 0.03


def test_detector_stays_quiet_on_the_null(null_panel):
    R, m, _ = null_panel
    d = st.crisis_correlation_test(R, m, n_sims=6, n_boot=120)
    assert not d["significant"]
    assert abs(d["genuine"]) < 0.06


def test_detector_scales_with_signal_strength():
    g = []
    for s in (0.0, 0.5, 1.0):
        R, m, _ = data.synthetic_panel(n_assets=8, n_years=12, signal_strength=s, seed=7)
        g.append(st.regime_corr(R, st.regime_labels(m, "past21"))["diff"])
    assert g[0] < g[1] < g[2]


# --------------------------------------------------------------------------- #
# Forbes-Rigobon and the constant-beta benchmark
# --------------------------------------------------------------------------- #
def test_fr_removes_a_pure_factor_variance_rise():
    """One-factor world, constant beta, idio constant, factor variance x9 in 'crisis':
    raw correlation rises, the FR-adjusted correlation returns to the calm level, and the
    constant-beta benchmark predicts the crisis level."""
    rng = np.random.default_rng(3)
    n, k = 20000, 6
    crisis = np.zeros(n, bool)
    crisis[: n // 5] = True
    f = rng.normal(size=n) * np.where(crisis, 3.0, 1.0)
    X = f[:, None] * 1.0 + rng.normal(size=(n, k)) * 1.5
    R = pd.DataFrame(X)
    mk = pd.Series(f)
    lab = pd.Series(np.where(crisis, 1.0, 0.0))
    rc = st.regime_corr(R, lab)
    fr = st.fr_panel(R, mk, lab)
    cb = st.constant_beta_benchmark(R, mk, lab)
    assert rc["crisis"] > rc["calm"] + 0.3
    assert fr["mkt_corr_crisis_fr"] == pytest.approx(fr["mkt_corr_calm"], abs=0.02)
    assert cb["implied_crisis"] == pytest.approx(rc["crisis"], abs=0.02)
    iv = st.idio_vol_ratio(R, mk, lab)
    assert iv["mkt_vol_ratio"] == pytest.approx(3.0, rel=0.05)
    assert iv["idio_vol_ratio"] == pytest.approx(1.0, rel=0.05)


# --------------------------------------------------------------------------- #
# The holder
# --------------------------------------------------------------------------- #
def test_diversification_ratio_of_independent_assets_is_sqrt_n():
    X = np.random.default_rng(0).normal(size=(20000, 16))
    lab = np.r_[np.ones(5000), np.zeros(15000)]
    ps = st.portfolio_stats(X, lab)
    assert ps["dr_calm"] == pytest.approx(4.0, rel=0.05)
    assert ps["shortfall"] == pytest.approx(0.0, abs=0.05)


def test_shortfall_positive_when_correlation_truly_jumps(planted, null_panel):
    Rp, _, tp = planted
    Rn, _, tn = null_panel
    lab_p = pd.Series(tp["crisis"].astype(float).to_numpy(), index=Rp.index)
    lab_n = pd.Series(tn["crisis"].astype(float).to_numpy(), index=Rn.index)
    sp = st.portfolio_stats(Rp, lab_p)
    sn = st.portfolio_stats(Rn, lab_n)
    assert sp["shortfall"] > 0.15 and sp["dr_crisis"] < sp["dr_calm"]
    assert abs(sn["shortfall"]) < 0.08


def test_episode_table_on_synthetic(planted):
    R, _, _ = planted
    ep = st.episode_table(R, episodes=(("test", str(R.index[1000].date()),
                                        str(R.index[1100].date())),))
    assert len(ep) == 1 and ep["days"].iloc[0] == 101
    assert ep["ratio"].iloc[0] > 0


# --------------------------------------------------------------------------- #
# Exceedance correlations
# --------------------------------------------------------------------------- #
def test_gaussian_exceedance_is_symmetric_and_falls_in_the_tails():
    g = st.gaussian_exceedance(0.5, n=400_000)
    assert st.asymmetry(g) == pytest.approx(0.0, abs=0.02)
    assert g[-1.5] < g[-0.5] < 0.5 and g[1.5] < g[0.5] < 0.5


def test_vectorised_exceedance_matches_the_scalar_one():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5000, 3))
    y = X.mean(axis=1) + rng.normal(size=5000)
    E = st.exceedance_matrix(X, y)
    for j in range(3):
        s = st.exceedance_corr(X[:, j], y).to_numpy()
        assert np.allclose(E[:, j], s, equal_nan=True)


def test_asymmetry_detects_planted_downside_dependence():
    """Correlation 0.8 when the common factor is in its left tail, 0.2 otherwise."""
    rng = np.random.default_rng(4)
    n = 60000
    f = rng.normal(size=n)
    rho = np.where(f < -1.0, 0.8, 0.2)
    x = np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.normal(size=n)
    y = np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.normal(size=n)
    assert st.asymmetry(st.exceedance_corr(x, y)) > 0.1
    xs = np.sqrt(0.4) * f + np.sqrt(0.6) * rng.normal(size=n)
    ys = np.sqrt(0.4) * f + np.sqrt(0.6) * rng.normal(size=n)
    assert abs(st.asymmetry(st.exceedance_corr(xs, ys))) < 0.05


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def test_circular_block_indices():
    rng = np.random.default_rng(0)
    ii = st.circular_block_indices(1000, 63, rng)
    assert len(ii) == 1000 and ii.min() >= 0 and ii.max() < 1000
    assert (np.diff(ii[:63]) % 1000 == 1).all()            # blocks are contiguous


def test_bootstrap_shapes_and_ci(null_panel):
    R, m, _ = null_panel
    labs = {"past21": st.regime_labels(m, "past21")}
    bt = st.bootstrap(R, m, labs, n_boot=30, exceed=True)
    assert len(bt["past21"]["diff"]) == 30 and len(bt["_asym"]) == 30
    lo, hi = st.ci(bt["past21"]["diff"])
    assert lo < hi
    assert 0 < st.boot_p_positive(bt["past21"]["diff"]) <= 1


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"genuine": 0.26, "genuine_lo": 0.18, "genuine_hi": 0.33, "genuine_p": 0.002,
         "naive_calm": 0.21, "naive_crisis": 0.47, "artefact": -0.002, "n_sims": 40,
         "share_regimes_genuine_pos": 1.0, "n_regimes_genuine_pos": 6, "n_regimes": 6,
         "contemp_artefact": 0.08, "contemp_naive": 0.32, "max_rolling_corr": 0.73,
         "n_years": 32.9, "fr_crisis": 0.16, "var_ratio": 11.2, "idio_ratio": 1.56,
         "exc_down_1": 0.49, "exc_up_1": 0.51, "exc_gauss_1": 0.21, "asym": -0.02,
         "asym_lo": -0.06, "asym_hi": 0.02, "asym_devol": 0.14, "asym_devol_lo": 0.08,
         "asym_devol_hi": 0.2, "dr_calm": 2.14, "dr_crisis": 1.44, "vol_crisis": 0.37,
         "vol_pred": 0.26, "shortfall": 0.43, "shortfall_artefact": 0.0,
         "shortfall_lo": 0.25, "shortfall_hi": 0.6, "n_episodes": 10, "n_episodes_under": 10,
         "episode_ratio_median": 1.31}
    h.update(over)
    return h


def test_verdict_signal_requires_a_bootstrap_ci_above_zero():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(genuine_lo=-0.01))["signal"] == "Weak"
    assert st.verdict(_headline(genuine=-0.02, genuine_lo=-0.1))["signal"] == "None"
    assert st.verdict(_headline(share_regimes_genuine_pos=1 / 3))["signal"] == "Mixed"


def test_verdict_tradability_never_investable():
    assert st.verdict(_headline())["trad"] == "Fragile"
    assert st.verdict(_headline(shortfall_lo=-0.01))["trad"] == "Mirage"
    assert st.verdict(_headline(shortfall=0.03, shortfall_lo=0.01))["trad"] == "Mirage"
    for s in (0.05, 0.4, 2.0):
        assert st.verdict(_headline(shortfall=s, shortfall_lo=s / 2))["trad"] != "Investable"


def test_verdict_prose_and_keys():
    v = st.verdict(_headline())
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
    assert "do **not** go to one" in v["signal_why"]
    assert "interdependence" in v["signal_why"]               # FR-adjusted below calm
    assert "contagion in their sense" in st.verdict(_headline(fr_crisis=0.3))["signal_why"]
    assert "nowhere near one" in v["one_sentence"]
    assert "cannot be told apart" in st.verdict(
        _headline(genuine=-0.02, genuine_lo=-0.1))["one_sentence"]


# --------------------------------------------------------------------------- #
# Real tape (gated)
# --------------------------------------------------------------------------- #
@needs_real
def test_real_naive_gap_is_large_and_past_only_artefact_small():
    R, m = data.load_returns()
    lab = st.regime_labels(m, "past21")
    rc = st.regime_corr(R, lab)
    assert rc["diff"] > 0.1 and rc["crisis"] < 0.9         # rises, but not to one
    fit = st.fit_ccc_garch(R, m)
    art = st.artefact_benchmark(fit, kinds=("past21",), n_sims=2, exceed=False)
    assert abs(art["regimes"]["past21"]["diff_mean"]) < 0.03

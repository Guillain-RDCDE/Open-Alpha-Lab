"""Strategy tests for Study 977 — the ratio, the optimiser, and the degenerate case."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from max_div import data, strategy as st  # noqa: E402


def _cov(n=8, rho=0.3, vols=None, seed=977):
    vols = np.linspace(0.08, 0.30, n) if vols is None else np.asarray(vols, float)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return np.outer(vols, vols) * corr


# --------------------------------------------------------------------------- #
# The ratio
# --------------------------------------------------------------------------- #
def test_diversification_ratio_is_one_for_a_single_asset():
    cov = np.array([[0.04]])
    assert st.diversification_ratio(np.array([1.0]), cov) == pytest.approx(1.0)


def test_ratio_is_one_when_everything_is_perfectly_correlated():
    cov = _cov(n=5, rho=1.0)
    w = np.full(5, 0.2)
    assert st.diversification_ratio(w, cov) == pytest.approx(1.0, abs=1e-6)


def test_ratio_reaches_sqrt_n_for_independent_equal_assets():
    cov = np.eye(9) * 0.04
    w = np.full(9, 1 / 9)
    assert st.diversification_ratio(w, cov) == pytest.approx(3.0, rel=1e-9)
    assert st.effective_bets(w, cov) == pytest.approx(9.0, rel=1e-9)


def test_ratio_falls_as_correlation_rises():
    w = np.full(6, 1 / 6)
    drs = [st.diversification_ratio(w, _cov(n=6, rho=r)) for r in (0.0, 0.3, 0.6, 0.9)]
    assert drs == sorted(drs, reverse=True)


# --------------------------------------------------------------------------- #
# The optimiser
# --------------------------------------------------------------------------- #
def test_max_div_maximises_the_ratio_against_a_brute_force_search():
    """On three assets, check the closed form against a fine grid over the simplex."""
    cov = _cov(n=3, rho=0.25, vols=[0.10, 0.18, 0.30])
    w_star = st.max_div_weights(cov)
    best, best_dr = None, -np.inf
    grid = np.linspace(0, 1, 101)
    for a in grid:
        for b in grid:
            if a + b > 1:
                continue
            w = np.array([a, b, 1 - a - b])
            dr = st.diversification_ratio(w, cov)
            if dr > best_dr:
                best, best_dr = w, dr
    assert st.diversification_ratio(w_star, cov) >= best_dr - 1e-3


def test_max_div_beats_every_competitor_on_its_own_objective():
    cov = _cov(n=10, rho=0.35)
    dr = {m: st.diversification_ratio(st.weights_for(m, cov), cov) for m in st.METHODS}
    assert dr["max_div"] >= max(v for k, v in dr.items() if k != "max_div") - 1e-9


def test_min_variance_beats_every_competitor_on_its_own_objective():
    cov = _cov(n=10, rho=0.35)
    var = {m: (lambda w: float(w @ cov @ w))(st.weights_for(m, cov)) for m in st.METHODS}
    assert var["min_var"] <= min(v for k, v in var.items() if k != "min_var") + 1e-12


def test_the_two_objectives_coincide_when_volatilities_are_equal():
    """The sharpest test in the module: equal variances collapse DR into portfolio variance."""
    d = st.degenerate_case(n=8, rho=0.3)
    assert d["max_abs_diff"] < 1e-6


def test_they_diverge_when_volatilities_differ():
    cov = _cov(n=8, rho=0.3, vols=np.linspace(0.05, 0.4, 8))
    gap = np.abs(st.max_div_weights(cov) - st.min_variance_weights(cov)).sum() / 2
    assert gap > 0.1


@pytest.mark.parametrize("method", st.METHODS)
def test_every_method_is_long_only_and_normalised(method):
    cov = _cov(n=12, rho=0.4, vols=np.linspace(0.06, 0.35, 12))
    w = st.weights_for(method, cov)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()


def test_projection_onto_the_simplex_is_correct():
    v = np.array([0.5, 0.9, -0.4, 0.2])
    p = st._project_simplex(v)
    assert p.sum() == pytest.approx(1.0)
    assert (p >= 0).all()
    # the projection must preserve the ordering of the inputs
    assert list(np.argsort(p)) == list(np.argsort(v))


def test_under_equicorrelation_the_mdp_is_exactly_inverse_volatility():
    """The identity that decides how much the correlation matrix is worth.

    With a constant correlation matrix ``C``, ``C^-1 1`` is proportional to ``1``, so the
    closed form ``w ∝ diag(sigma)^-1 C^-1 1`` collapses to ``w ∝ 1/sigma`` — inverse
    volatility, exactly, for every level of correlation. The most diversified portfolio
    therefore differs from the free competitor *only* to the extent that correlations are
    **dispersed**, which is a far narrower claim than the marketing makes.
    """
    for rho in (0.0, 0.2, 0.5, 0.8):
        cov = _cov(n=6, rho=rho, vols=np.linspace(0.1, 0.3, 6))
        assert np.allclose(st.max_div_weights(cov), st.inverse_vol_weights(cov), atol=1e-8)


def test_max_div_responds_to_dispersed_correlations():
    """Give the matrix a block structure and the MDP stops being inverse volatility."""
    vols = np.linspace(0.1, 0.3, 6)
    corr = np.full((6, 6), 0.15)
    corr[:3, :3] = 0.85          # one tight cluster
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    w_md, w_iv = st.max_div_weights(cov), st.inverse_vol_weights(cov)
    assert np.abs(w_md - w_iv).sum() / 2 > 0.05
    # and the crowded cluster is the part that gets cut
    assert w_md[:3].sum() < w_iv[:3].sum()


# --------------------------------------------------------------------------- #
# The scoreboard
# --------------------------------------------------------------------------- #
def _frame(n_assets=10, n_obs=1600, seed=977):
    rng = np.random.default_rng(seed)
    cov = _cov(n=n_assets, rho=0.35, vols=np.linspace(0.06, 0.35, n_assets)) / 252
    L = np.linalg.cholesky(cov + np.eye(n_assets) * 1e-14)
    X = rng.normal(0, 1, (n_obs, n_assets)) @ L.T
    return pd.DataFrame(X, index=pd.bdate_range("2005-01-03", periods=n_obs),
                        columns=[f"A{i}" for i in range(n_assets)])


def test_walk_forward_is_out_of_sample():
    R = _frame()
    cut = 1000
    bad = R.copy(); bad.iloc[cut:] *= 5
    a = st.walk_forward(R, methods=("max_div",))
    b = st.walk_forward(bad, methods=("max_div",))
    a = a[a["date"] < R.index[cut - 63]]
    b = b[b["date"] < R.index[cut - 63]]
    assert np.allclose(a["max_weight"].to_numpy(), b["max_weight"].to_numpy())


def test_walk_forward_reports_in_and_out_of_sample_ratios():
    wf = st.walk_forward(_frame(), window=252, step=126)
    s = st.summarise(wf)
    assert list(s.index) == list(st.METHODS)
    assert (s["dr_in"] > 1).all()
    assert s.loc["max_div", "dr_in"] >= s["dr_in"].max() - 1e-9


def test_out_of_sample_ratio_slips_below_the_in_sample_one():
    """Optimising a quantity in sample and delivering it out of sample are different things."""
    wf = st.walk_forward(_frame(n_obs=2500), window=126, step=63)
    s = st.summarise(wf)
    assert s.loc["max_div", "dr_slippage"] < 0.02


def test_paired_test_is_antisymmetric():
    wf = st.walk_forward(_frame(), window=252, step=63)
    ab = st.paired_test(wf, "max_div", "inv_vol")
    ba = st.paired_test(wf, "inv_vol", "max_div")
    assert ab["t"] == pytest.approx(-ba["t"], abs=1e-9)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"weight_gap_multi": 0.28, "weight_gap_sectors": 0.19, "dr_in_maxdiv": 1.62,
         "dr_in_minvar": 1.41, "degenerate_gap": 1e-9, "dr_out_maxdiv": 1.48,
         "dr_slippage_maxdiv": -0.09, "vol_maxdiv": 0.096, "vol_invvol": 0.101,
         "vol_equal": 0.118, "best_t_vs_invvol": 2.4, "beats_invvol_panels": 2,
         "turnover_maxdiv": 0.35, "turnover_invvol": 0.10,
         "effective_n_maxdiv": 7.2, "effective_n_minvar": 4.1}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(weight_gap_sectors=0.02))["signal"] == "Weak"
    assert st.verdict(_headline(weight_gap_sectors=0.02, weight_gap_multi=0.03,
                                dr_in_maxdiv=1.0))["signal"] == "None"


def test_verdict_usefulness_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(best_t_vs_invvol=0.8))["trad"] == "Fragile"
    assert st.verdict(_headline(best_t_vs_invvol=0.8,
                                beats_invvol_panels=0))["trad"] == "Mirage"


def test_verdict_prose_quotes_the_free_competitor():
    v = st.verdict(_headline())
    assert "inverse" in v["trad_why"].lower() and "inverse" in v["one_sentence"].lower()
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

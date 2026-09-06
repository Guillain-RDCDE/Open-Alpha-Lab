"""Strategy tests for Study 978 — optimisers, the averaging, and a known truth."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from resampled import data, strategy as st  # noqa: E402


def _world(n=6, seed=978, mu_spread=0.0004, rho=0.3):
    """A small world with a known mean vector and covariance matrix."""
    vols = np.linspace(0.006, 0.02, n)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vols, vols) * corr
    mu = np.linspace(0.0, mu_spread, n)
    return mu, cov


def _sample(mu, cov, n_obs=500, seed=978):
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, cov, size=n_obs, method="cholesky")


# --------------------------------------------------------------------------- #
# The optimisers
# --------------------------------------------------------------------------- #
def test_projection_lands_on_the_simplex():
    v = np.array([0.7, -0.3, 0.9, 0.1])
    p = st.project_simplex(v)
    assert p.sum() == pytest.approx(1.0)
    assert (p >= 0).all()


def test_min_variance_is_the_minimum_among_random_long_only_portfolios():
    mu, cov = _world(n=5)
    w = st.min_variance_weights(mu, cov)
    rng = np.random.default_rng(0)
    for _ in range(300):
        r = rng.dirichlet(np.ones(5))
        assert w @ cov @ w <= r @ cov @ r + 1e-12


def test_max_sharpe_beats_random_long_only_portfolios():
    mu, cov = _world(n=5, mu_spread=0.001)
    w = st.max_sharpe_weights(mu, cov)
    s = (w @ mu) / np.sqrt(w @ cov @ w)
    rng = np.random.default_rng(0)
    best = max(((r @ mu) / np.sqrt(r @ cov @ r)) for r in rng.dirichlet(np.ones(5), 400))
    assert s >= best - 1e-3


def test_max_sharpe_matches_a_brute_force_grid_on_three_assets():
    mu, cov = _world(n=3, mu_spread=0.0008)
    w = st.max_sharpe_weights(mu, cov)
    grid = np.linspace(0, 1, 121)
    best = -np.inf
    for a in grid:
        for b in grid:
            if a + b > 1:
                continue
            r = np.array([a, b, 1 - a - b])
            v = r @ cov @ r
            if v > 0:
                best = max(best, (r @ mu) / np.sqrt(v))
    assert (w @ mu) / np.sqrt(w @ cov @ w) >= best - 1e-3


# --------------------------------------------------------------------------- #
# The averaging
# --------------------------------------------------------------------------- #
def test_resampling_is_deterministic_given_a_seed():
    mu, cov = _world()
    X = _sample(mu, cov)
    a = st.resampled_weights(X, "max_sharpe", n_resamples=20, seed=5)
    b = st.resampled_weights(X, "max_sharpe", n_resamples=20, seed=5)
    assert np.allclose(a, b)
    assert not np.allclose(a, st.resampled_weights(X, "max_sharpe", n_resamples=20, seed=6))


def test_resampled_weights_are_on_the_simplex():
    mu, cov = _world(n=8)
    X = _sample(mu, cov)
    for obj in st.OBJECTIVES:
        w = st.resampled_weights(X, obj, n_resamples=25)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()


def test_resampling_diversifies_relative_to_a_single_optimisation():
    """The averaging's headline property: fewer corners, more names held."""
    mu, cov = _world(n=10, mu_spread=0.0006)
    X = _sample(mu, cov, n_obs=300)
    w_plain = st.plain_weights(X, "max_sharpe")
    w_res = st.resampled_weights(X, "max_sharpe", n_resamples=60)
    assert st.concentration(w_res)["n_held"] >= st.concentration(w_plain)["n_held"]
    assert st.concentration(w_res)["max_weight"] <= st.concentration(w_plain)["max_weight"] + 1e-9


def test_resampling_converges_as_the_number_of_draws_grows():
    mu, cov = _world(n=6)
    X = _sample(mu, cov)
    a = st.resampled_weights(X, "min_var", n_resamples=40, seed=1)
    b = st.resampled_weights(X, "min_var", n_resamples=200, seed=1)
    c = st.resampled_weights(X, "min_var", n_resamples=200, seed=2)
    assert st.weight_distance(b, c) < st.weight_distance(a, c) + 0.05


def test_nonparametric_and_parametric_resampling_differ():
    mu, cov = _world(n=8, mu_spread=0.0008)
    X = _sample(mu, cov, n_obs=400)
    a = st.resampled_weights(X, "max_sharpe", n_resamples=40, parametric=True, seed=3)
    b = st.resampled_weights(X, "max_sharpe", n_resamples=40, parametric=False, seed=3)
    assert st.weight_distance(a, b) > 0.0


def test_shrinkage_also_diversifies_and_costs_one_pass():
    mu, cov = _world(n=10, mu_spread=0.0006)
    X = _sample(mu, cov, n_obs=300)
    w_plain = st.plain_weights(X, "max_sharpe")
    w_shrunk = st.shrunk_weights(X, "max_sharpe")
    assert st.concentration(w_shrunk)["n_held"] >= st.concentration(w_plain)["n_held"]


def test_resampled_and_shrunk_portfolios_are_close_to_each_other():
    """The study's central comparison, pinned: two very different procedures, similar answers."""
    mu, cov = _world(n=10, mu_spread=0.0006)
    X = _sample(mu, cov, n_obs=300)
    d_shrunk = st.weight_distance(st.resampled_weights(X, "max_sharpe", n_resamples=60),
                                  st.shrunk_weights(X, "max_sharpe"))
    d_plain = st.weight_distance(st.resampled_weights(X, "max_sharpe", n_resamples=60),
                                 st.plain_weights(X, "max_sharpe"))
    assert d_shrunk < d_plain


def test_minimum_variance_ignores_the_means_entirely():
    mu, cov = _world(n=6)
    X = _sample(mu, cov)
    shifted = X + 0.01
    assert np.allclose(st.plain_weights(X, "min_var"), st.plain_weights(shifted, "min_var"))


# --------------------------------------------------------------------------- #
# Against a known truth
# --------------------------------------------------------------------------- #
def test_truth_experiment_reports_every_method():
    mu, cov = _world(n=6, mu_spread=0.0008)
    out = st.truth_experiment(mu, cov, "max_sharpe", n_obs=250, n_trials=6, n_resamples=20)
    assert set(out["method"]) == set(st.METHODS)
    assert (out["utility_gap"] >= -1e-9).all()


def test_estimation_error_hurts_and_resampling_helps_on_max_sharpe():
    mu, cov = _world(n=10, mu_spread=0.001)
    out = st.truth_experiment(mu, cov, "max_sharpe", n_obs=250, n_trials=12, n_resamples=30)
    g = out.groupby("method")["utility_gap"].mean()
    assert g["plain"] > 0
    assert g["resampled"] <= g["plain"]


def test_with_identical_TRUE_means_the_two_objectives_are_the_same_problem():
    """The degenerate control, stated on the parameters rather than on a sample."""
    mu, cov = _world(n=8, mu_spread=0.0)
    a = st.optimise(mu, cov, "max_sharpe")
    b = st.optimise(mu, cov, "min_var")
    assert st.weight_distance(a, b) < 0.02


def test_but_the_SAMPLE_version_diverges_wildly_from_it():
    """And this is the whole study: identical true means, wildly different sample answers.

    With no dispersion in expected returns there is nothing for the mean vector to say, yet a
    maximum-Sharpe optimiser fitted to a sample chases the noise in it and lands a long way
    from the minimum-variance portfolio that the true parameters imply. Resampling and
    shrinkage both exist to close that distance — and both do.
    """
    mu, cov = _world(n=8, mu_spread=0.0)
    X = _sample(mu, cov, n_obs=500)
    target = st.optimise(mu, cov, "min_var")
    d_plain = st.weight_distance(st.plain_weights(X, "max_sharpe"), target)
    d_res = st.weight_distance(st.resampled_weights(X, "max_sharpe", n_resamples=40), target)
    d_shr = st.weight_distance(st.shrunk_weights(X, "max_sharpe"), target)
    assert d_plain > 0.2
    assert d_res < d_plain and d_shr < d_plain


# --------------------------------------------------------------------------- #
# The scoreboard
# --------------------------------------------------------------------------- #
def _frame(n_assets=8, n_obs=1600, seed=978):
    mu, cov = _world(n=n_assets, mu_spread=0.0006)
    X = _sample(mu, cov, n_obs=n_obs, seed=seed)
    return pd.DataFrame(X, index=pd.bdate_range("2005-01-03", periods=n_obs),
                        columns=[f"A{i}" for i in range(n_assets)])


def test_walk_forward_is_out_of_sample():
    R = _frame()
    cut = 1100
    bad = R.copy(); bad.iloc[cut:] *= 4
    a = st.walk_forward(R, "min_var", window=504, step=126, methods=("plain",))
    b = st.walk_forward(bad, "min_var", window=504, step=126, methods=("plain",))
    a = a[a["date"] < R.index[cut - 126]]
    b = b[b["date"] < R.index[cut - 126]]
    assert np.allclose(a["max_weight"].to_numpy(), b["max_weight"].to_numpy())


def test_walk_forward_covers_every_method():
    wf = st.walk_forward(_frame(), "min_var", window=504, step=252, n_resamples=15)
    s = st.summarise(wf)
    assert list(s.index) == list(st.METHODS)
    assert (s["realised_vol"] > 0).all()


def test_paired_test_is_antisymmetric():
    wf = st.walk_forward(_frame(n_obs=3000), "min_var", window=504, step=126, n_resamples=10)
    ab = st.paired_test(wf, "resampled", "plain")
    ba = st.paired_test(wf, "plain", "resampled")
    assert ab["t"] == pytest.approx(-ba["t"], abs=1e-9)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"weight_gap_vs_plain": 0.22, "weight_gap_vs_shrunk": 0.07,
         "utility_gap_plain": 0.06, "utility_gap_resampled": 0.03,
         "utility_gap_shrunk": 0.031, "utility_gap_equal": 0.05,
         "n_held_resampled": 8.0, "n_held_plain": 3.0, "max_weight_resampled": 0.25,
         "max_weight_plain": 0.62, "n_assets": 10, "ret_resampled": 0.07,
         "vol_resampled": 0.09, "sharpe_resampled": 0.78, "ret_plain": 0.06,
         "sharpe_plain": 0.65, "ret_shrunk": 0.069, "sharpe_shrunk": 0.77,
         "t_vs_plain": 2.1, "t_vs_shrunk": 0.4, "n_resamples": 60}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(weight_gap_vs_plain=0.02))["signal"] == "Weak"
    assert st.verdict(_headline(weight_gap_vs_plain=0.02,
                                utility_gap_resampled=0.09))["signal"] == "None"


def test_verdict_usefulness_needs_to_beat_the_cheap_fix():
    assert st.verdict(_headline())["trad"] == "Fragile"
    assert st.verdict(_headline(t_vs_shrunk=2.5))["trad"] == "Useful"
    assert st.verdict(_headline(t_vs_plain=-0.5))["trad"] == "Mirage"


def test_verdict_prose_names_the_competitor():
    v = st.verdict(_headline())
    assert "shrink" in v["trad_why"].lower()
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

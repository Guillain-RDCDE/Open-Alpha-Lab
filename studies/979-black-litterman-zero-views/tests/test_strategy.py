"""Strategy tests for Study 979 — the identity first, everything else afterwards."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bl_prior import data, strategy as st  # noqa: E402


def _cov(n=8, rho=0.3, seed=979):
    vols = np.linspace(0.005, 0.02, n)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    return np.outer(vols, vols) * corr


# --------------------------------------------------------------------------- #
# The identity — the whole point of the study
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("prior", st.PRIORS)
@pytest.mark.parametrize("tau", [0.01, 0.05, 0.5])
def test_zero_views_returns_the_prior_exactly(prior, tau):
    cov = _cov(n=9)
    wp = st.prior_weights(cov, prior)
    wb = st.posterior_weights(cov, wp, tau=tau)
    assert np.abs(wb - wp).sum() / 2 < 1e-10


def test_the_identity_holds_for_a_pathological_covariance():
    """Near-singular matrices are where implementations quietly stop being exact."""
    rng = np.random.default_rng(979)
    X = rng.normal(0, 0.01, (30, 25))          # 25 assets, 30 observations
    cov = np.cov(X, rowvar=False, ddof=1)
    wp = st.prior_weights(cov, "inverse_vol")
    assert np.abs(st.posterior_weights(cov, wp) - wp).sum() / 2 < 1e-8


def test_implied_returns_are_a_restatement_of_the_prior():
    cov = _cov(n=6)
    wp = st.prior_weights(cov, "equal")
    pi = st.implied_returns(cov, wp, delta=2.5)
    assert np.allclose(pi, 2.5 * cov @ wp)
    # and optimising on them returns the prior
    assert np.abs(st.optimal_weights(pi, cov, 2.5) - wp).sum() < 1e-10


def test_implied_returns_scale_linearly_with_risk_aversion():
    cov = _cov(n=5)
    wp = st.prior_weights(cov, "equal")
    assert np.allclose(st.implied_returns(cov, wp, delta=5.0),
                       2 * st.implied_returns(cov, wp, delta=2.5))
    # but the optimal portfolio does not move: delta cancels
    assert np.allclose(st.posterior_weights(cov, wp, delta=5.0),
                       st.posterior_weights(cov, wp, delta=2.5))


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def test_a_positive_view_raises_that_asset_weight():
    cov = _cov(n=8)
    wp = st.prior_weights(cov, "equal")
    P, q = st.single_view(8, asset=3, size_ann=0.05)
    wb = st.posterior_weights(cov, wp, P, q)
    assert wb[3] > wp[3]


def test_a_negative_view_lowers_it():
    cov = _cov(n=8)
    wp = st.prior_weights(cov, "equal")
    P, q = st.single_view(8, asset=3, size_ann=-0.05)
    assert st.posterior_weights(cov, wp, P, q)[3] < wp[3]


def test_a_view_equal_to_the_prior_belief_changes_nothing():
    """The genuinely neutral view is ``q = P pi``, and it must leave the portfolio alone."""
    cov = _cov(n=8)
    wp = st.prior_weights(cov, "equal")
    P, _ = st.single_view(8, asset=2, size_ann=0.0)
    q = st.implied_view(P, cov, wp)
    assert np.abs(st.posterior_weights(cov, wp, P, q) - wp).sum() / 2 < 1e-8


def test_a_zero_sized_view_is_NOT_neutral():
    """The point most descriptions of the model get wrong, pinned.

    A view that an asset will out-perform by *zero* contradicts a prior whose implied view is
    positive, so it moves the book — often more than a modest positive view does. The neutral
    statement is "I agree with the prior", not "I expect zero".
    """
    cov = _cov(n=8)
    wp = st.prior_weights(cov, "equal")
    P, q0 = st.single_view(8, asset=2, size_ann=0.0)
    implied = float(st.implied_view(P, cov, wp)[0])
    assert implied > 0
    assert np.abs(st.posterior_weights(cov, wp, P, q0) - wp).sum() / 2 > 1e-3


def test_views_further_from_the_prior_move_more_of_the_book():
    """Monotone in the DISTANCE from the prior's own implied view, not in the view's level."""
    cov = _cov(n=8)
    wp = st.prior_weights(cov, "equal")
    P, _ = st.single_view(8, asset=4, size_ann=0.0)
    implied_ann = float(st.implied_view(P, cov, wp)[0]) * st.TRADING_DAYS
    moved = []
    for extra in (0.0, 0.02, 0.05, 0.10, 0.20):
        _, q = st.single_view(8, asset=4, size_ann=implied_ann + extra)
        w = st.posterior_weights(cov, wp, P, q)
        moved.append(float(np.abs(w - wp).sum() / 2))
    assert np.all(np.diff(moved) >= -1e-12)
    assert moved[0] < 1e-8


def test_higher_tau_means_the_prior_is_held_less_tightly():
    cov = _cov(n=8)
    wp = st.prior_weights(cov, "equal")
    curve = st.view_strength_curve(cov, wp, asset=4, sizes=(0.05,), taus=(0.01, 0.05, 0.5))
    moved = curve.sort_values("tau")["book_moved"].to_numpy()
    assert moved[-1] >= moved[0]


def test_a_relative_view_is_funded_from_the_benchmark():
    cov = _cov(n=6)
    wp = st.prior_weights(cov, "equal")
    P, q = st.single_view(6, asset=1, size_ann=0.05, benchmark=4)
    wb = st.posterior_weights(cov, wp, P, q)
    assert wb[1] > wp[1] and wb[4] < wp[4]
    assert wb.sum() == pytest.approx(1.0)


def test_prior_sensitivity_compares_priors_and_views():
    cov = _cov(n=9)
    d = st.prior_sensitivity(cov, asset=2, size_ann=0.03)
    assert list(d.index) == list(st.PRIORS)
    assert (d["view_moved_book"] >= 0).all()
    assert d.filter(like="vs_").notna().to_numpy().sum() > 0


# --------------------------------------------------------------------------- #
# The mechanical view and the backtest
# --------------------------------------------------------------------------- #
def _frame(n_assets=8, n_obs=2500, seed=979):
    rng = np.random.default_rng(seed)
    cov = _cov(n=n_assets) / 252 * 252
    L = np.linalg.cholesky(cov + np.eye(n_assets) * 1e-14)
    X = rng.normal(0, 1, (n_obs, n_assets)) @ L.T + 0.0002
    return pd.DataFrame(X, index=pd.bdate_range("2005-01-03", periods=n_obs),
                        columns=[f"A{i}" for i in range(n_assets)])


def test_momentum_view_is_long_short_and_balanced():
    X = _frame().to_numpy()
    P, q = st.momentum_view(X, size_ann=0.03, top_k=2)
    assert P.shape == (1, X.shape[1])
    assert P.sum() == pytest.approx(0.0)
    assert q[0] == pytest.approx(0.03 / 252)


def test_momentum_view_skips_the_most_recent_month():
    X = _frame(n_obs=600).to_numpy()
    mangled = X.copy()
    mangled[-21:] *= 10            # only the skipped month is changed
    a, _ = st.momentum_view(X)
    b, _ = st.momentum_view(mangled)
    assert np.allclose(a, b)


def test_walk_forward_covers_every_method():
    wf = st.walk_forward(_frame(), window=504, step=126)
    s = st.summarise(wf)
    assert list(s.index) == list(st.METHODS)
    assert (s["realised_vol"] > 0).all()
    assert s.loc["prior", "tilt_from_prior"] == pytest.approx(0.0, abs=1e-12)


def test_the_tilt_grows_with_the_stated_view_size():
    tilts = {}
    for size in (0.0, 0.03, 0.10):
        wf = st.walk_forward(_frame(), window=504, step=126, size_ann=size)
        tilts[size] = st.summarise(wf).loc["black_litterman", "tilt_from_prior"]
    assert tilts[0.10] > tilts[0.03] > tilts[0.0] - 1e-12


def test_plain_mean_variance_is_the_wild_one():
    wf = st.walk_forward(_frame(), window=504, step=126)
    s = st.summarise(wf)
    assert s.loc["plain_mv", "max_weight"] > s.loc["black_litterman", "max_weight"]
    # Turnover is NOT reliably higher: a long-only projection can pin the plain optimiser in
    # a corner for quarters at a time, which is a different pathology, not a milder one.
    assert s.loc["plain_mv", "max_weight"] > s.loc["prior", "max_weight"]


def test_paired_test_is_antisymmetric():
    wf = st.walk_forward(_frame(n_obs=3000), window=504, step=126)
    ab = st.paired_test(wf, "black_litterman", "prior")
    ba = st.paired_test(wf, "prior", "black_litterman")
    assert ab["t"] == pytest.approx(-ba["t"], abs=1e-9)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"zero_view_error": 1e-14, "book_moved_3pct": 0.08, "book_moved_10pct": 0.22,
         "tau": 0.05, "prior_spread": 0.31, "view_move_mean": 0.08,
         "ret_bl": 0.075, "vol_bl": 0.10, "sharpe_bl": 0.75, "ret_prior": 0.07,
         "sharpe_prior": 0.71, "ret_mv": 0.05, "sharpe_mv": 0.42,
         "max_weight_mv": 0.55, "max_weight_bl": 0.18, "t_bl_vs_prior": 1.2,
         "n_rebalances": 60}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(book_moved_3pct=0.01))["signal"] == "Weak"
    assert st.verdict(_headline(zero_view_error=1e-3))["signal"] == "None"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Fragile"
    assert st.verdict(_headline(t_bl_vs_prior=2.6))["trad"] == "Investable"
    assert st.verdict(_headline(t_bl_vs_prior=-0.4))["trad"] == "Mirage"


def test_verdict_prose_states_the_identity():
    v = st.verdict(_headline())
    assert "prior" in v["one_sentence"] and "identity" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

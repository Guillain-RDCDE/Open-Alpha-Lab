"""Strategy tests for Study 976 — the tree, the bisection, and what each competitor does."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hrp import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The tree
# --------------------------------------------------------------------------- #
def test_correlation_distance_is_a_metric_at_the_extremes():
    corr = np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    d = st.correlation_distance(corr)
    assert d[0, 1] == pytest.approx(0.0)     # identical assets are at distance zero
    assert d[0, 2] == pytest.approx(1.0)     # perfectly opposed are at distance one
    assert np.allclose(np.diag(d), 0.0)


def test_single_linkage_returns_n_minus_one_merges():
    X, cov = st.block_panel(n_per_block=5, n_blocks=2, n_obs=400)
    order = st.cluster_order(np.cov(X, rowvar=False, ddof=1))
    assert sorted(order) == list(range(10))


def test_quasi_diagonal_order_puts_the_blocks_together():
    """Two planted blocks must come out as two contiguous runs."""
    X, cov = st.block_panel(n_per_block=6, n_blocks=2, n_obs=2000, rho_in=0.8, rho_out=0.05)
    order = st.cluster_order(np.cov(X, rowvar=False, ddof=1))
    labels = [0 if i < 6 else 1 for i in order]
    switches = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
    assert switches == 1          # exactly one boundary between the two families


def test_cluster_order_is_a_permutation_on_real_shaped_input():
    rng = np.random.default_rng(976)
    X = rng.normal(0, 0.01, (300, 14))
    order = st.cluster_order(np.cov(X, rowvar=False, ddof=1))
    assert sorted(order) == list(range(14))


# --------------------------------------------------------------------------- #
# The weightings
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", st.METHODS)
def test_every_method_returns_normalised_weights(method):
    X, cov = st.block_panel(n_per_block=6, n_blocks=2, n_obs=400)
    w = st.weights_for(method, np.cov(X, rowvar=False, ddof=1))
    assert w.shape == (12,)
    assert w.sum() == pytest.approx(1.0)


def test_hrp_is_long_only_by_construction():
    X, _ = st.block_panel(n_per_block=8, n_blocks=2, n_obs=200, rho_in=0.9, rho_out=-0.2)
    w = st.hrp_weights(np.cov(X, rowvar=False, ddof=1))
    assert (w > 0).all()


def test_min_variance_will_happily_short():
    X, _ = st.block_panel(n_per_block=8, n_blocks=2, n_obs=100, rho_in=0.9, rho_out=0.85)
    w = st.min_variance_weights(np.cov(X, rowvar=False, ddof=1))
    assert w.min() < 0


def test_inverse_variance_ignores_correlation_entirely():
    X, _ = st.block_panel(n_per_block=5, n_blocks=2, n_obs=800)
    cov = np.cov(X, rowvar=False, ddof=1)
    scrambled = cov.copy()
    off = ~np.eye(cov.shape[0], dtype=bool)
    scrambled[off] = scrambled[off][::-1]      # shuffle the off-diagonal, keep the variances
    assert np.allclose(st.inverse_variance_weights(cov),
                       st.inverse_variance_weights(scrambled))


def test_hrp_does_not_ignore_correlation():
    """The whole point: change the correlations, keep the variances, and HRP must move."""
    X, _ = st.block_panel(n_per_block=5, n_blocks=2, n_obs=1500, rho_in=0.85, rho_out=0.0)
    cov = np.cov(X, rowvar=False, ddof=1)
    flat = np.diag(np.diag(cov))               # same variances, no correlation at all
    assert not np.allclose(st.hrp_weights(cov), st.hrp_weights(flat), atol=1e-3)


def test_hrp_reduces_to_inverse_variance_without_structure():
    """With no correlation the tree has nothing to say and the two must nearly agree."""
    rng = np.random.default_rng(976)
    sd = np.linspace(0.005, 0.03, 12)
    X = rng.normal(0, 1, (4000, 12)) * sd
    cov = np.cov(X, rowvar=False, ddof=1)
    assert np.abs(st.hrp_weights(cov) - st.inverse_variance_weights(cov)).max() < 0.05


def test_risk_parity_equalises_risk_contributions():
    X, _ = st.block_panel(n_per_block=5, n_blocks=2, n_obs=1500)
    cov = np.cov(X, rowvar=False, ddof=1)
    w = st.risk_parity_weights(cov)
    rc = w * (cov @ w)
    assert rc.std() / rc.mean() < 0.05
    assert (w > 0).all()


def test_hrp_allocates_less_to_a_crowded_cluster():
    """Nine near-identical assets and one independent one: the loner must not be crushed."""
    n_obs = 3000
    rng = np.random.default_rng(976)
    f = rng.normal(0, 0.01, n_obs)
    crowd = np.column_stack([f + rng.normal(0, 0.002, n_obs) for _ in range(9)])
    loner = rng.normal(0, 0.01, n_obs).reshape(-1, 1)
    cov = np.cov(np.hstack([crowd, loner]), rowvar=False, ddof=1)
    w = st.hrp_weights(cov)
    # HRP overweights the loner — but only modestly, because recursive bisection splits the
    # ordered list by COUNT, not by cluster membership: with nine crowd members and one
    # outsider the first split is 5 / 5 and the loner is bundled with four of the crowd.
    # That is a real property of the published algorithm, not an implementation slip.
    assert w[-1] > 1.2 * w[:9].mean()
    assert w[-1] > st.inverse_variance_weights(cov)[-1]
    assert w[-1] < 0.5      # and it does NOT get the half-weight a cluster-aware split implies


def test_concentration_reports_effective_positions():
    c = st.concentration(np.full(10, 0.1))
    assert c["effective_n"] == pytest.approx(10.0)
    assert c["max_weight"] == pytest.approx(0.1)
    assert c["short"] == 0.0


# --------------------------------------------------------------------------- #
# The scoreboard
# --------------------------------------------------------------------------- #
def _frame(n_assets=16, n_obs=1600, seed=976):
    X, _ = st.block_panel(n_per_block=n_assets // 2, n_blocks=2, n_obs=n_obs, seed=seed)
    return pd.DataFrame(X, index=pd.bdate_range("2005-01-03", periods=n_obs),
                        columns=[f"A{i}" for i in range(n_assets)])


def test_walk_forward_is_out_of_sample():
    R = _frame()
    cut = 1000
    bad = R.copy(); bad.iloc[cut:] *= 5
    a = st.walk_forward(R, methods=("hrp",))
    b = st.walk_forward(bad, methods=("hrp",))
    a = a[a["date"] < R.index[cut - 63]]
    b = b[b["date"] < R.index[cut - 63]]
    assert np.allclose(a["max_weight"].to_numpy(), b["max_weight"].to_numpy())


def test_walk_forward_covers_every_method():
    s = st.summarise(st.walk_forward(_frame(), window=252, step=126))
    assert list(s.index) == list(st.METHODS)
    assert (s["realised_vol"] > 0).all()
    assert s.loc["equal", "max_weight"] == pytest.approx(1 / 16)


def test_hrp_turns_over_less_than_the_optimiser():
    wf = st.walk_forward(_frame(n_assets=20, n_obs=2000), window=126, step=63)
    s = st.summarise(wf)
    assert s.loc["hrp", "turnover"] < s.loc["min_var", "turnover"]


def test_paired_test_is_antisymmetric():
    wf = st.walk_forward(_frame(), window=252, step=63)
    ab = st.paired_test(wf, "hrp", "min_var")
    ba = st.paired_test(wf, "min_var", "hrp")
    assert ab["t"] == pytest.approx(-ba["t"], abs=1e-9)


def test_block_panel_plants_the_correlation_it_claims():
    X, cov = st.block_panel(n_per_block=6, n_blocks=2, n_obs=6000, rho_in=0.7, rho_out=0.1)
    C = np.corrcoef(X, rowvar=False)
    assert C[0, 1] == pytest.approx(0.7, abs=0.08)
    assert C[0, 8] == pytest.approx(0.1, abs=0.08)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_panels_hierarchy_matters": 3, "weight_gap_wide": 0.04, "weight_gap_multi": 0.05,
         "n_names": 40, "effective_n_hrp": 22.0, "effective_n_minvar": 8.0,
         "max_weight_hrp": 0.08, "max_weight_minvar": 0.40, "vol_hrp": 0.14,
         "vol_minvar": 0.16, "vol_equal": 0.17, "vol_invvar": 0.142,
         "t_vs_minvar": 3.1, "t_vs_equal": 4.0, "t_vs_invvar": -0.4,
         "turnover_hrp": 0.25, "turnover_minvar": 1.10}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(n_panels_hierarchy_matters=1))["signal"] == "Weak"
    assert st.verdict(_headline(n_panels_hierarchy_matters=0))["signal"] == "None"


def test_verdict_usefulness_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(t_vs_equal=0.4))["trad"] == "Fragile"
    assert st.verdict(_headline(t_vs_equal=0.4, t_vs_minvar=0.2))["trad"] == "Mirage"


def test_verdict_prose_quotes_the_control():
    v = st.verdict(_headline())
    assert "inverse variance" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

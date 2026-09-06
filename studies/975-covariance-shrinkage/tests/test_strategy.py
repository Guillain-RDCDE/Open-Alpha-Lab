"""Strategy tests for Study 975 — estimators against a known covariance matrix."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from shrinkage import data, strategy as st  # noqa: E402


def _factor_panel(n_assets=30, n_obs=200, rho=0.35, vol=0.02, seed=975, dispersion=0.5):
    """Returns with a KNOWN covariance: one common factor plus idiosyncratic noise.

    ``dispersion`` spreads the betas and the idiosyncratic volatilities. It matters more than
    it looks: at ``dispersion = 0`` every pair has *exactly* the same correlation, so the
    constant-correlation target is the truth and Ledoit-Wolf correctly shrinks all the way to
    it (delta = 1). A test of "does the intensity fall as the sample grows" therefore needs a
    heterogeneous panel, and a test of "does it go to 1 when the target is right" needs the
    homogeneous one. Both are below.
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 + dispersion * (np.linspace(-1, 1, n_assets))
    beta = np.sqrt(rho) * scale
    idio_sd = vol * np.sqrt(max(1 - rho, 1e-9)) * scale[::-1]
    f = rng.normal(0, vol, n_obs)
    X = np.outer(f, beta) + rng.normal(0, 1, (n_obs, n_assets)) * idio_sd
    truth = (vol ** 2) * np.outer(beta, beta) + np.diag(idio_sd ** 2)
    return X, truth


# --------------------------------------------------------------------------- #
# The estimators
# --------------------------------------------------------------------------- #
def test_sample_cov_matches_numpy():
    X, _ = _factor_panel(n_assets=6, n_obs=500)
    assert np.allclose(st.sample_cov(X), np.cov(X, rowvar=False, ddof=1))


def test_diagonal_cov_keeps_variances_and_kills_covariances():
    X, _ = _factor_panel(n_assets=6, n_obs=500)
    D = st.diagonal_cov(X)
    assert np.allclose(np.diag(D), np.var(X, axis=0, ddof=1))
    assert np.allclose(D - np.diag(np.diag(D)), 0.0)


@pytest.mark.parametrize("method", ["identity", "constant_corr"])
def test_shrinkage_intensity_is_between_zero_and_one(method):
    X, _ = _factor_panel(n_assets=25, n_obs=120)
    C, delta = st.estimate(X, method)
    assert 0.0 <= delta <= 1.0
    assert C.shape == (25, 25)
    assert np.allclose(C, C.T)


def test_shrinkage_intensity_falls_as_the_sample_grows():
    """More data, less need for a prior — the estimator must know that."""
    deltas = {}
    for n_obs in (80, 250, 2000):
        X, _ = _factor_panel(n_assets=25, n_obs=n_obs)
        deltas[n_obs] = st.estimate(X, "constant_corr")[1]
    assert deltas[80] > deltas[250] > deltas[2000]


def test_shrinkage_beats_the_sample_matrix_against_a_known_truth():
    X, truth = _factor_panel(n_assets=40, n_obs=120)
    e_sample = st.frobenius_error(st.sample_cov(X), truth)
    e_shrunk = st.frobenius_error(st.estimate(X, "constant_corr")[0], truth)
    assert e_shrunk < e_sample


def test_shrinkage_barely_moves_when_rows_dwarf_parameters():
    X, truth = _factor_panel(n_assets=8, n_obs=6000)
    e_sample = st.frobenius_error(st.sample_cov(X), truth)
    e_shrunk = st.frobenius_error(st.estimate(X, "constant_corr")[0], truth)
    assert abs(e_shrunk - e_sample) < 0.05
    assert st.estimate(X, "constant_corr")[1] < 0.35


def test_intensity_goes_to_one_when_the_target_is_the_truth():
    """A homogeneous panel IS a constant-correlation matrix, so shrink all the way to it."""
    X, _ = _factor_panel(n_assets=25, n_obs=250, dispersion=0.0)
    delta_cc = st.estimate(X, "constant_corr")[1]
    delta_id = st.estimate(X, "identity")[1]
    assert delta_cc > 0.9
    assert delta_id < delta_cc      # the identity target is plainly wrong here, so less of it


def test_constant_correlation_target_recovers_the_average_correlation():
    X, _ = _factor_panel(n_assets=20, n_obs=400, rho=0.5, dispersion=0.0)
    C, _ = st.estimate(X, "constant_corr")
    sd = np.sqrt(np.diag(C))
    R = C / np.outer(sd, sd)
    off = ~np.eye(20, dtype=bool)
    assert R[off].mean() == pytest.approx(0.5, abs=0.12)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_condition_number_explodes_when_assets_approach_observations():
    thin, _ = _factor_panel(n_assets=40, n_obs=45)
    fat, _ = _factor_panel(n_assets=40, n_obs=2000)
    assert st.condition_number(st.sample_cov(thin)) > 10 * st.condition_number(st.sample_cov(fat))


def test_shrinkage_repairs_the_condition_number():
    X, _ = _factor_panel(n_assets=40, n_obs=60)
    assert st.condition_number(st.estimate(X, "identity")[0]) < \
        st.condition_number(st.sample_cov(X))


def test_eigen_spread_reports_a_dominant_factor():
    X, _ = _factor_panel(n_assets=30, n_obs=800, rho=0.6)
    sp = st.eigen_spread(st.sample_cov(X))
    assert sp["top_share"] > 0.4
    assert sp["max"] > sp["min"]


def test_min_variance_weights_sum_to_one_and_long_only_has_no_shorts():
    X, _ = _factor_panel(n_assets=10, n_obs=500)
    C = st.sample_cov(X)
    w = st.min_variance_weights(C)
    assert w.sum() == pytest.approx(1.0)
    wl = st.min_variance_weights(C, long_only=True)
    assert (wl >= 0).all() and wl.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The walk-forward scoreboard
# --------------------------------------------------------------------------- #
def _panel_frame(n_assets=30, n_obs=1500, seed=975):
    X, _ = _factor_panel(n_assets=n_assets, n_obs=n_obs, seed=seed)
    idx = pd.bdate_range("2005-01-03", periods=n_obs)
    return pd.DataFrame(X, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_walk_forward_is_out_of_sample():
    """Mangling the future must leave every earlier estimate untouched."""
    R = _panel_frame()
    cut = 900
    bad = R.copy()
    bad.iloc[cut:] *= 6
    a = st.walk_forward(R, window=252, step=63, methods=("sample",))
    b = st.walk_forward(bad, window=252, step=63, methods=("sample",))
    a = a[a["date"] < R.index[cut - 63]]
    b = b[b["date"] < R.index[cut - 63]]
    assert np.allclose(a["promised_vol"].to_numpy(), b["promised_vol"].to_numpy())


def test_walk_forward_reports_optimism_on_a_thin_window():
    R = _panel_frame(n_assets=30, n_obs=1500)
    wf = st.walk_forward(R, window=126, step=63, methods=("sample", "constant_corr"))
    s = st.summarise(wf)
    assert s.loc["sample", "optimism"] < 0          # promised less than delivered
    assert s.loc["constant_corr", "realised_vol"] <= s.loc["sample", "realised_vol"]


def test_summarise_has_a_row_per_estimator():
    R = _panel_frame(n_assets=12, n_obs=1200)
    s = st.summarise(st.walk_forward(R, window=252, step=126))
    assert list(s.index) == list(st.ESTIMATORS)
    assert (s["realised_vol"] > 0).all()


def test_paired_test_is_antisymmetric():
    R = _panel_frame(n_assets=20, n_obs=1500)
    wf = st.walk_forward(R, window=126, step=63, methods=("sample", "identity"))
    ab = st.paired_vol_test(wf, "sample", "identity")
    ba = st.paired_vol_test(wf, "identity", "sample")
    assert ab["t"] == pytest.approx(-ba["t"], abs=1e-9)
    assert ab["n"] == ba["n"]


def test_costs_and_turnover_are_recorded():
    R = _panel_frame(n_assets=15, n_obs=1200)
    wf = st.walk_forward(R, window=252, step=63, methods=("sample",))
    assert (wf["turnover"] >= 0).all()
    assert wf["turnover"].iloc[0] == pytest.approx(1.0, abs=1.5)   # first trade from cash


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"wide_optimism_sample": -0.30, "wide_vol_saving": 0.12, "wide_paired_t": 3.4,
         "n_names": 40, "n_params_wide": 820, "window": 252,
         "wide_promised_sample": 0.09, "wide_realised_sample": 0.13,
         "wide_realised_best": 0.114, "wide_condition_sample": 90000,
         "wide_condition_best": 300, "sector_rows_per_param": 3.8,
         "sector_optimism_sample": -0.03, "best_method": "Ledoit-Wolf -> constant correlation",
         "n_rebalances": 60, "wide_win_rate": 0.78, "turnover_best": 0.4,
         "turnover_sample": 1.2, "max_weight_best": 0.12, "max_weight_sample": 0.45}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline(wide_optimism_sample=0.30))["signal"] == "Real"
    assert st.verdict(_headline(wide_optimism_sample=0.08))["signal"] == "Weak"
    assert st.verdict(_headline(wide_optimism_sample=0.01))["signal"] == "None"


def test_verdict_usefulness_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(wide_paired_t=0.5))["trad"] == "Fragile"
    assert st.verdict(_headline(wide_vol_saving=-0.02))["trad"] == "Mirage"


def test_verdict_prose_quotes_the_numbers():
    v = st.verdict(_headline(wide_vol_saving=0.17))
    assert "17%" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

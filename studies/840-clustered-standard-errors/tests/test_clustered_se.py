"""Offline, fixed-seed tests for the clustered-standard-errors machinery — Study 840.

The synthetic panel is deterministic; the common time effect leaves the pooled slope unbiased
but inflates the naive OLS SE by the Moulton factor; one-way FIRM clustering does NOT fix a TIME
effect; time-clustering and Fama-MacBeth restore calibration; the pitfall vanishes when the
common factor is switched off; Fama-MacBeth recovers a planted slope (power); costs make the
null a loser; the inference primitives behave. All offline, no network.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from clustered_se import data, strategy as st  # noqa: E402


# ---- the generator is deterministic and well-formed -----------------------
def test_panel_deterministic(null_panel):
    X, Y = null_panel
    X2, Y2 = data.panel(X.shape[0], X.shape[1], X.shape[2], rho_x=0.5, rho_e=0.5,
                        beta=0.0, seed=840)
    assert np.allclose(X, X2) and np.allclose(Y, Y2)


def test_panel_shapes_and_variance(null_panel):
    X, Y = null_panel
    assert X.shape == Y.shape
    # unit-variance construction (both x and e are standardised by design)
    assert abs(X.std() - 1.0) < 0.02


def test_intra_period_correlation_matches_rho():
    # A high rho_x must show up as strong cross-firm correlation within a period.
    X, _ = data.panel(1, 200, 30, rho_x=0.8, rho_e=0.0, beta=0.0, seed=1)
    x = X[0]                                   # (T, N)
    # average pairwise within-period correlation ~ rho_x
    corr = np.corrcoef(x.T)                     # (N, N) across periods
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert 0.7 < off.mean() < 0.9


# ---- the pooled slope is UNBIASED — the damage is only in the SE -----------
def test_pooled_slope_unbiased_under_null(null_panel):
    X, Y = null_panel
    c = st.calibration(*(X, Y))
    assert abs(c["b_mean"]) < 0.01            # point estimate centred at zero
    assert abs(c["b_fm_mean"]) < 0.01


# ---- the pitfall: naive OLS SE is far too small (Moulton inflation) --------
def test_naive_se_understates_truth(null_panel):
    X, Y = null_panel
    c = st.calibration(*(X, Y))
    assert c["ols_se_ratio"] < 0.5             # naive SE is < half the true variability
    # naive t SD ~ the Moulton closed form
    moult = data.theoretical_moulton(X.shape[2], 0.5, 0.5)
    assert abs(c["ols_t_sd"] - moult) / moult < 0.15


def test_naive_false_positive_rate_is_inflated(null_panel):
    X, Y = null_panel
    c = st.calibration(*(X, Y))
    assert c["ols_fp"] > 0.4                    # catastrophic over-rejection (nominal is 0.05)


# ---- firm clustering (WRONG dimension) does NOT help a time effect ---------
def test_firm_clustering_does_not_fix_time_effect(null_panel):
    X, Y = null_panel
    c = st.calibration(*(X, Y))
    assert c["firm_fp"] > 0.4                   # still badly over-sized


# ---- time clustering and Fama-MacBeth RESTORE calibration ------------------
def test_time_and_fama_macbeth_restore_calibration(null_panel):
    X, Y = null_panel
    c = st.calibration(*(X, Y))
    assert c["fm_fp"] < 0.10                    # Fama-MacBeth ~ nominal
    assert c["time_fp"] < 0.12                  # time clustering ~ nominal (few-cluster margin)
    assert 0.8 < c["fm_se_ratio"] < 1.2         # FM SE ~ the truth


# ---- the control: remove the common factor and the pitfall vanishes -------
def test_iid_control_all_calibrated(iid_panel):
    X, Y = iid_panel
    c = st.calibration(*(X, Y))
    for key in ("ols", "firm", "time", "fm"):
        assert c[f"{key}_fp"] < 0.09           # every estimator ~ nominal with no dependence
    assert abs(c["ols_t_sd"] - 1.0) < 0.1      # no inflation


# ---- Moulton closed form grows with N and with rho ------------------------
def test_moulton_monotone():
    assert data.theoretical_moulton(2, 0.5, 0.5) < data.theoretical_moulton(100, 0.5, 0.5)
    assert data.theoretical_moulton(50, 0.0, 0.5) == 1.0
    assert abs(data.theoretical_moulton(50, 0.5, 0.5) - np.sqrt(13.25)) < 1e-9


def test_inflation_curve_naive_rises_fm_flat():
    df = st.inflation_curve_rho([0.0, 0.4, 0.8], 400, 40, 40, rho_x=0.5, seed=840)
    assert df["ols_fp"].is_monotonic_increasing         # worse with more dependence
    assert df["fm_fp"].max() < 0.10                      # FM stays calibrated throughout


# ---- the positive control: Fama-MacBeth FIRES on a planted slope ----------
def test_fama_macbeth_has_power(edge_panel):
    X, Y = edge_panel
    inf = st.panel_inference(X, Y)
    power = float(np.mean(np.abs(inf["t_fm"]) > 1.96))
    assert power > 0.6                                   # detects a real effect
    assert abs(float(np.mean(inf["b_fm"])) - 0.06) < 0.01   # recovers the planted slope


def test_planted_slope_lifts_fm_t(null_panel, edge_panel):
    tn = float(np.mean(st.panel_inference(*null_panel)["t_fm"]))
    te = float(np.mean(st.panel_inference(*edge_panel)["t_fm"]))
    assert te > tn + 1.0                                 # real edge raises the FM t


# ---- the costed timer: a null minus costs is a loser ----------------------
def test_timer_costs_make_null_negative():
    Xo, Yo = data.one_panel(50, 50, rho_x=0.5, rho_e=0.5, beta=0.0, seed=840)
    tm = st.timer_stats(Xo, Yo, ret_scale=0.01, cost_bps=5.0, borrow_bps_yr=50.0)
    assert tm["net_bps"] < tm["gross_bps"]              # costs bite
    assert tm["net_bps"] < 0                            # net loser by construction


# ---- inference primitives behave ------------------------------------------
def test_one_sample_t_on_known_signal():
    rng = np.random.default_rng(0)
    x = 0.5 + rng.standard_normal(4000)
    assert st.one_sample_t(x) > 20                       # a strong mean is strongly significant


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_nominal_rate_is_five_percent():
    assert abs(st.nominal_rate(1.96) - 0.05) < 0.002


def test_fama_macbeth_slope_matches_manual_single_period():
    # FM per-period slope equals a plain OLS slope computed by hand on that period.
    X, Y = data.one_panel(3, 30, rho_x=0.5, rho_e=0.5, beta=0.0, seed=7)
    inf = st.panel_inference(X[None], Y[None])
    b_t = inf["b_t"][0]
    for t in range(X.shape[0]):
        xt, yt = X[t], Y[t]
        xd = xt - xt.mean(); yd = yt - yt.mean()
        manual = (xd * yd).sum() / (xd * xd).sum()
        assert abs(b_t[t] - manual) < 1e-10

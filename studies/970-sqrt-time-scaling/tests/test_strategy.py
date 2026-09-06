"""Strategy tests for Study 970 — variance ratios against closed-form truth."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sqrt_time import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The estimator against closed-form truth
# --------------------------------------------------------------------------- #
def test_variance_ratio_is_one_on_iid_data(iid_path):
    r, truth = iid_path
    for q in (5, 21, 63):
        assert st.variance_ratio(r, q) == pytest.approx(1.0, abs=0.12)


def test_variance_ratio_matches_the_ar1_closed_form(planted):
    r, truth = planted
    for q in (5, 21):
        assert st.variance_ratio(r, q) == pytest.approx(truth["vr_closed_form"][q], rel=0.20)


def test_positive_autocorrelation_lifts_the_variance_ratio_above_one():
    hi, _ = data.synthetic_ar1(n_years=25, ar1=0.20, seed=970)
    lo, _ = data.synthetic_ar1(n_years=25, ar1=-0.20, seed=970)
    assert st.variance_ratio(hi, 21) > 1.1
    assert st.variance_ratio(lo, 21) < 0.9


def test_bias_correction_matters_in_small_samples():
    """Without the Lo-MacKinlay constants the estimator drifts below 1 on short samples."""
    r, _ = data.synthetic_ar1(n_years=2, ar1=0.0, signal_strength=0.0, seed=970)
    raw = st.variance_ratio(r, 63, bias_correct=False)
    fixed = st.variance_ratio(r, 63, bias_correct=True)
    assert fixed > raw


def test_variance_ratio_is_nan_when_the_sample_is_too_short():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 20))
    assert np.isnan(st.variance_ratio(r, 21))


# --------------------------------------------------------------------------- #
# The test statistic
# --------------------------------------------------------------------------- #
def test_lo_mackinlay_does_not_reject_on_iid_data(iid_path):
    r, _ = iid_path
    out = st.lo_mackinlay_test(r, 21)
    assert abs(out["z"]) < 2.5
    assert 0.0 <= out["p_value"] <= 1.0


def test_lo_mackinlay_rejects_a_strong_ar1():
    r, _ = data.synthetic_ar1(n_years=25, ar1=0.20, seed=970)
    out = st.lo_mackinlay_test(r, 21)
    assert out["z"] > 2.0 and out["p_value"] < 0.05


def test_robust_test_survives_volatility_clustering_without_autocorrelation():
    """A GARCH-like tape with zero autocorrelation must not be called a trend.

    Twenty independent worlds; the robust statistic should reject at roughly its nominal rate,
    not systematically. This is the whole reason the heteroskedasticity-robust form is used.
    """
    rejects = 0
    for s in range(20):
        rng = np.random.default_rng(970 + s)
        n = 5000
        v = np.empty(n); e = np.empty(n); v[0] = 1e-4
        for t in range(1, n):
            v[t] = 1e-6 + 0.10 * e[t - 1] ** 2 + 0.88 * v[t - 1]
            e[t] = np.sqrt(v[t]) * rng.normal()
        r = pd.Series(e, index=pd.bdate_range("2000-01-03", periods=n))
        if abs(st.lo_mackinlay_test(r, 21)["z"]) >= 2.0:
            rejects += 1
    assert rejects <= 6


def test_vr_curve_has_one_row_per_horizon(planted):
    r, _ = planted
    c = st.vr_curve(r)
    assert list(c.index) == list(st.HORIZONS)
    assert c["vr"].notna().all()


# --------------------------------------------------------------------------- #
# The assumption-free check and the consequences
# --------------------------------------------------------------------------- #
def test_realised_scaling_agrees_with_the_estimator(planted):
    r, _ = planted
    p = 100 * (1 + r).cumprod()
    rs = st.realised_scaling(p, horizons=(5, 21))
    for q in (5, 21):
        assert rs.loc[q, "implied_vr"] == pytest.approx(st.variance_ratio(r, q), rel=0.45)


def test_realised_scaling_is_flat_on_iid_data(iid_path):
    r, _ = iid_path
    p = 100 * (1 + r).cumprod()
    rs = st.realised_scaling(p, horizons=(5, 21, 63))
    assert (rs["ratio"] - 1).abs().max() < 0.2


def test_vol_scaling_error_is_the_square_root_of_the_ratio():
    assert st.vol_scaling_error(1.44) == pytest.approx(0.2)
    assert st.vol_scaling_error(1.0) == pytest.approx(0.0)
    assert st.vol_scaling_error(0.64) == pytest.approx(-0.2)


def test_var_scaling_error_moves_with_the_variance_ratio():
    a = st.var_scaling_error(0.01, 1.0)
    b = st.var_scaling_error(0.01, 1.44)
    assert a["error_pct"] == pytest.approx(0.0)
    assert b["error_pct"] == pytest.approx(0.2)
    assert b["var_corrected"] > b["var_sqrt_rule"]


def test_sharpe_scaling_uses_the_lo_factor(planted):
    r, _ = planted
    out = st.sharpe_scaling_error(r)
    assert out["factor"] != pytest.approx(out["sqrt_factor"], rel=1e-6)
    assert out["sharpe_lo"] / out["sharpe_naive"] == pytest.approx(
        out["factor"] / out["sqrt_factor"], rel=1e-9)


def test_lo_factor_is_unbiased_but_very_noisy_at_the_annual_horizon():
    """The correction is honest on average and useless on one sample — both must be true.

    Lo's factor at q = 252 sums 251 estimated autocorrelations. On i.i.d. data its *mean*
    across draws sits on sqrt(252), but a single twenty-year sample lands tens of percent
    away. This test pins both halves, because a study that quoted the single-sample number as
    a correction would be adding noise and calling it precision.
    """
    errs = []
    for s in range(20):
        r, _ = data.synthetic_ar1(n_years=20, ar1=0.0, signal_strength=0.0, seed=970 + s)
        errs.append(st.sharpe_scaling_error(r)["relative_error"])
    errs = np.array(errs)
    assert abs(errs.mean()) < 0.10          # unbiased
    assert errs.std(ddof=1) > 0.05          # and noisy


def test_lo_factor_is_better_behaved_at_a_monthly_horizon():
    errs_q252, errs_q21 = [], []
    for s in range(15):
        r, _ = data.synthetic_ar1(n_years=20, ar1=0.0, signal_strength=0.0, seed=970 + s)
        errs_q252.append(st.sharpe_scaling_error(r, q=252)["relative_error"])
        errs_q21.append(st.sharpe_scaling_error(r, q=21)["relative_error"])
    assert np.std(errs_q21, ddof=1) < np.std(errs_q252, ddof=1)


def test_autocorrelation_profile_reads_the_planted_coefficient(planted):
    r, truth = planted
    prof = st.autocorrelation_profile(r, lags=3)
    assert prof[1] == pytest.approx(truth["ar1"], abs=0.06)
    assert len(prof) == 3


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_reject_annual": 4, "n_tickers": 10, "max_vr": 1.6, "max_vr_ticker": "SHY",
         "min_vr": 0.7, "min_vr_ticker": "TQQQ", "max_abs_vol_error": 0.26,
         "max_var_error": 0.26, "max_var_ticker": "SHY", "max_sharpe_error": 0.15,
         "max_sharpe_ticker": "SHY", "max_sharpe_naive": 1.2, "max_sharpe_lo": 1.4,
         "spy_vol_error": -0.01, "lo_factor_noise_iid": 0.22}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(n_reject_annual=1))["signal"] == "Weak"
    assert st.verdict(_headline(n_reject_annual=0))["signal"] == "None"


def test_verdict_usefulness_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(max_abs_vol_error=0.05))["trad"] == "Fragile"
    assert st.verdict(_headline(max_abs_vol_error=0.01))["trad"] == "Mirage"


def test_verdict_prose_quotes_the_numbers():
    v = st.verdict(_headline(max_vr=1.75))
    assert "1.75" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

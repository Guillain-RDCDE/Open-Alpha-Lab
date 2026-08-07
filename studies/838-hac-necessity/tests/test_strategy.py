"""The two spines of the study:
(1) a serially-correlated null makes the NAIVE t-stat over-reject wildly, while the
    Newey-West (HAC) t-stat stays near nominal — and the inflation matches its closed form;
(2) the HAC machinery still FIRES on a genuinely planted effect (unbiased, not merely
    conservative). Plus a statsmodels cross-check that our hand-rolled NW is correct."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hac_necessity import data as d, strategy as st  # noqa: E402


# ---- primitive sanity ------------------------------------------------------
def test_naive_t_matches_manual():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    manual = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(st.one_sample_t(x) - manual) < 1e-9


def test_naive_t_matrix_matches_scalar(iid_null):
    tm = st.one_sample_t_matrix(iid_null)
    for i in range(5):
        assert abs(tm[i] - st.one_sample_t(iid_null[i])) < 1e-9


def test_nw_t_matrix_matches_scalar(overlap_null):
    tm = st.newey_west_t_matrix(overlap_null, lags=42)
    for i in range(5):
        assert abs(tm[i] - st.newey_west_t(overlap_null[i], lags=42)) < 1e-9


def test_nw_matches_statsmodels():
    """Our hand-rolled Newey-West t equals statsmodels' HAC t on an intercept-only OLS."""
    sm = pytest.importorskip("statsmodels.api")
    x = d.overlap_returns(1500, window=21, seed=11).to_numpy()
    L = 42
    ours = st.newey_west_t(x, lags=L)
    model = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": False})
    theirs = float(model.tvalues[0])
    assert abs(ours - theirs) < 1e-6


def test_nw_auto_lags_grows():
    assert st.nw_auto_lags(100) <= st.nw_auto_lags(10000)
    assert st.nw_auto_lags(50) >= 1


# ---- spine #1: the naive t over-rejects; NW does not -----------------------
def test_naive_over_rejects_on_overlap(overlap_null):
    """On a mean-ZERO but serially-correlated null, the naive false-positive rate blows far
    past the nominal 5%, while Newey-West stays near it."""
    fp = st.false_positive_rate(overlap_null, lags=42, crit=1.96)
    assert fp["naive_fp"] > 0.40            # a wild false-positive rate
    assert fp["nw_fp"] < 0.18               # HAC keeps it near nominal
    assert fp["naive_fp"] > 3 * fp["nw_fp"]  # naive rejects several times more often


def test_inflation_factor_matches_sqrt_window(overlap_null):
    """The naive-t SD under the null equals the closed-form inflation factor sqrt(window)."""
    fp = st.false_positive_rate(overlap_null, lags=42, crit=1.96)
    assert abs(fp["naive_t_sd"] - np.sqrt(21)) < 0.6
    assert abs(fp["nw_t_sd"] - 1.0) < 0.35   # NW t SD ~ 1 (calibrated)


def test_control_removes_the_inflation(iid_null):
    """Switch autocorrelation OFF (window=1) and BOTH tests are calibrated — the serial
    correlation is the cause."""
    fp = st.false_positive_rate(iid_null, lags=42, crit=1.96)
    assert fp["naive_fp"] < 0.10
    assert fp["nw_fp"] < 0.12
    assert abs(fp["naive_t_sd"] - 1.0) < 0.15


def test_ar1_inflation_matches_theory():
    """AR(1) naive-t SD tracks sqrt((1+rho)/(1-rho)) and the naive FP rises with rho."""
    df = st.inflation_curve_ar1([0.0, 0.4, 0.8], 400, 2000, seed=838,
                                lags=st.nw_auto_lags(2000))
    assert (df["naive_fp"].values[0] < df["naive_fp"].values[-1])   # monotone in rho
    for _, r in df.iterrows():
        assert abs(r["naive_t_sd"] - r["theory_inflation"]) < 0.4 * r["theory_inflation"] + 0.15


def test_inflation_curve_overlap_monotone():
    df = st.inflation_curve_overlap([1, 10, 42], 400, 2000, seed=838)
    assert list(df["naive_fp"]) == sorted(df["naive_fp"])  # FP rises with the window


# ---- spine #2: the machinery FIRES on a planted effect (positive control) --
def test_power_on_planted_effect():
    """With a genuine positive mean injected, Newey-West rejects the null most of the time
    and with the correct sign — the correction is unbiased, not just conservative."""
    pw = st.power_check(400, 2520, window=21, mean=6e-4, seed=838, lags=42)
    assert pw["nw_power"] > 0.6
    assert pw["nw_t_mean"] > 2.0
    assert pw["nw_t_positive_share"] > 0.9


def test_null_has_no_power():
    """With no planted effect, Newey-West rejection is near nominal (no phantom signal)."""
    pw = st.power_check(400, 2520, window=21, mean=0.0, seed=838, lags=42)
    assert pw["nw_power"] < 0.15


# ---- the timer: nothing to trade -------------------------------------------
def test_timer_is_a_mirage():
    x = d.overlap_returns(2520, window=21, seed=838).to_numpy()
    tm = st.timer_stats(x)
    assert tm["net_bps"] < tm["gross_bps"]   # costs bite
    assert tm["net_bps"] < 0                 # a null minus costs is a loser


# ---- wilson interval sanity ------------------------------------------------
def test_wilson_brackets_point_estimate():
    lo, hi = st.wilson_interval(64, 100)
    assert lo < 0.64 < hi
    assert 0.0 <= lo <= hi <= 1.0

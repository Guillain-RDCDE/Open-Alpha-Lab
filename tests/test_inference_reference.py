"""Reference-value tests for the inferential core.

The other test files check shapes and directions; this one pins the numbers:
HAC t-stats against statsmodels, Sharpe standard errors against the closed
forms, block-bootstrap CI coverage against a known AR(1) truth, and the
stationary-bootstrap Reality Check against its anti-conservative iid variant.
All seeds are fixed; everything here is deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab import analytics, bayes, decompose, stats


# ---------------------------------------------------------------------------
# (a) mean_tstat_hac vs statsmodels OLS with HAC covariance
# ---------------------------------------------------------------------------
def test_hac_tstat_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(12345)
    r = rng.normal(2e-4, 1e-2, 500)
    lags = 5

    ours = analytics.mean_tstat_hac(pd.Series(r), lags=lags)
    fit = sm.OLS(r, np.ones((r.size, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": False}
    )
    np.testing.assert_allclose(ours["tstat"], float(fit.tvalues[0]), rtol=1e-10)
    np.testing.assert_allclose(ours["se_bps"], float(fit.bse[0]) * 1e4, rtol=1e-10)
    np.testing.assert_allclose(ours["mean_bps"], float(fit.params[0]) * 1e4, rtol=1e-10)


# ---------------------------------------------------------------------------
# (b) sharpe_with_se closed forms, one per method
# ---------------------------------------------------------------------------
def test_sharpe_se_closed_form_each_method():
    rng = np.random.default_rng(99)
    r = rng.standard_t(df=5, size=400) * 0.01 + 4e-4  # fat tails on purpose
    n = r.size
    sr = r.mean() / r.std(ddof=1)
    ann = np.sqrt(252)
    d = r - r.mean()

    # iid: Var = (1 + SR^2/2)/n
    s = analytics.sharpe_with_se(pd.Series(r), method="iid")
    np.testing.assert_allclose(s["se_ann"], np.sqrt((1 + 0.5 * sr**2) / n) * ann, rtol=1e-12)
    np.testing.assert_allclose(s["sharpe_ann"], sr * ann, rtol=1e-12)

    # mertens: Var = (1 + SR^2/2 - g3*SR + (g4-3)/4*SR^2)/n
    m2 = (d**2).mean()
    g3 = (d**3).mean() / m2**1.5
    g4 = (d**4).mean() / m2**2
    var_m = (1 + 0.5 * sr**2 - g3 * sr + (g4 - 3) / 4 * sr**2) / n
    s_m = analytics.sharpe_with_se(pd.Series(r), method="mertens")
    np.testing.assert_allclose(s_m["se_ann"], np.sqrt(var_m) * ann, rtol=1e-12)

    # lo: Var = (1 + 2*sum_k (1-k/(q+1)) rho_k + SR^2/2)/n, q lags
    q = 3
    g0 = d @ d / n
    factor = 1.0 + 2.0 * sum(
        (1 - k / (q + 1)) * (d[k:] @ d[:-k] / n) / g0 for k in range(1, q + 1)
    )
    var_lo = (factor + 0.5 * sr**2) / n
    s_lo = analytics.sharpe_with_se(pd.Series(r), method="lo", q=q)
    np.testing.assert_allclose(s_lo["se_ann"], np.sqrt(var_lo) * ann, rtol=1e-12)
    assert s_lo["q"] == q

    # fat tails (g4 > 3) must widen the Mertens SE vs the iid one
    assert s_m["se_ann"] > s["se_ann"]


def test_sharpe_se_rejects_unknown_method():
    with pytest.raises(ValueError):
        analytics.sharpe_with_se(pd.Series(np.ones(10) * 0.01 + np.arange(10) * 1e-4), method="bogus")


# ---------------------------------------------------------------------------
# (c) block-bootstrap CI coverage on AR(1) data with a known true Sharpe
# ---------------------------------------------------------------------------
def _ar1(rng, n, phi, mu, sig_e, burn=50):
    x = np.empty(n + burn)
    x[0] = mu
    eps = rng.normal(0, sig_e, n + burn)
    for t in range(1, n + burn):
        x[t] = mu + phi * (x[t - 1] - mu) + eps[t]
    return x[burn:]


def test_block_bootstrap_ci_coverage_on_ar1():
    phi, mu, sig_e = 0.3, 5e-4, 0.01
    sig_x = sig_e / np.sqrt(1 - phi**2)  # stationary AR(1) sd
    true_sr = mu / sig_x * np.sqrt(252)
    n, n_sims = 250, 200

    rng = np.random.default_rng(7)
    hits = 0
    for s in range(n_sims):
        r = pd.Series(_ar1(rng, n, phi, mu, sig_e))
        ci = stats.sharpe_ci_bootstrap(r, n_boot=300, seed=s)
        if ci["ci_low"] <= true_sr <= ci["ci_high"]:
            hits += 1
    coverage = hits / n_sims
    assert 0.88 <= coverage <= 0.99, f"95% CI covered the truth {coverage:.1%} of the time"


def test_block_bootstrap_reports_block_size_and_valid_count():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(3e-4, 0.01, 512))
    res = stats.sharpe_ci_bootstrap(r, n_boot=200, seed=1)
    assert res["method"] == "cbb"
    assert res["block_size"] == round(512 ** (1 / 3))
    assert 0 < res["n_boot_valid"] <= res["n_boot"]
    # explicit block size and the iid fallback both still work
    res8 = stats.sharpe_ci_bootstrap(r, n_boot=200, seed=1, block_size=8)
    assert res8["block_size"] == 8
    res_iid = stats.sharpe_ci_bootstrap(r, n_boot=200, seed=1, method="iid")
    assert res_iid["block_size"] == 1


# ---------------------------------------------------------------------------
# (d) reality_check: stationary bootstrap is more conservative than iid
#     on autocorrelated data
# ---------------------------------------------------------------------------
def test_stationary_bootstrap_more_conservative_than_iid_on_ar1_panel():
    rng = np.random.default_rng(5)
    n, m, phi = 750, 6, 0.6
    cols = {}
    for j in range(m):
        eps = rng.normal(0, 0.01, n)
        x = np.empty(n)
        x[0] = eps[0]
        for t in range(1, n):
            x[t] = phi * x[t - 1] + eps[t]
        cols[f"s{j}"] = x + (4e-4 if j == 0 else 0.0)  # one marginal "signal"
    panel = pd.DataFrame(cols)

    p_stat = bayes.reality_check(panel, n_boot=500, seed=3)
    p_iid = bayes.reality_check(panel, n_boot=500, seed=3, method="iid")
    assert p_stat["bootstrap_method"] == "stationary"
    assert p_stat["mean_block_length"] > 1.0
    # iid resampling destroys serial dependence -> too-small p-values
    assert p_stat["reality_check_pvalue"] > p_iid["reality_check_pvalue"]


# ---------------------------------------------------------------------------
# Lo annualisation factor and risk-free plumbing
# ---------------------------------------------------------------------------
def test_lo_annualization_factor_white_noise_is_sqrt_q():
    # Small q keeps the sampling noise of the 251-lag tail out of the test:
    # the estimator sums (q-k)*rho_k, whose variance grows fast with q.
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0, 0.01, 20000))
    f = analytics.lo_annualization_factor(r, q=12)
    assert abs(f - np.sqrt(12)) / np.sqrt(12) < 0.05  # ~sqrt(q) for iid noise


def test_lo_annualization_factor_below_sqrt_q_when_autocorrelated():
    rng = np.random.default_rng(13)
    x = _ar1(rng, 20000, phi=0.3, mu=0.0, sig_e=0.01)
    f = analytics.lo_annualization_factor(pd.Series(x), q=252)
    assert f < np.sqrt(252)  # positive autocorrelation shrinks the factor


def test_risk_free_lowers_sharpe_consistently():
    rng = np.random.default_rng(17)
    r = pd.Series(rng.normal(8e-4, 0.01, 2000))
    # scalar annualised rf
    s0 = analytics.sharpe_with_se(r)
    s_rf = analytics.sharpe_with_se(r, rf=0.05)
    assert s_rf["sharpe_ann"] < s0["sharpe_ann"]
    # an aligned per-period series with the same values matches the scalar
    rf_series = pd.Series(0.05 / 252, index=r.index)
    s_rf2 = analytics.sharpe_with_se(r, rf=rf_series)
    np.testing.assert_allclose(s_rf["sharpe_ann"], s_rf2["sharpe_ann"], rtol=1e-12)
    # same plumbing in the bootstrap CI and the point estimate
    b0 = stats.sharpe_ci_bootstrap(r, n_boot=100, seed=0)
    b_rf = stats.sharpe_ci_bootstrap(r, n_boot=100, seed=0, rf=0.05)
    assert b_rf["sharpe"] < b0["sharpe"]
    assert b_rf["sharpe"] == pytest.approx(stats.annualized_sharpe(r, rf=0.05))


def test_summary_rf_only_moves_sharpe():
    from quantlab import diagnostics

    dec = decompose.decompose(diagnostics.synthetic_ohlc(seed=3, n_days=1000))
    s0 = decompose.summary(dec)
    s_rf = decompose.summary(dec, rf=0.05)
    assert (s_rf["sharpe"] < s0["sharpe"]).all()
    pd.testing.assert_series_equal(s_rf["mean_bps_day"], s0["mean_bps_day"])
    pd.testing.assert_series_equal(s_rf["cum_return"], s0["cum_return"])

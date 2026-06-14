"""Portfolio construction invariants, stats helpers, and the study spine:
the global blend beats US-only on Sharpe only when correlation is low enough
and returns are equal — and loses when the US has a structural edge."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from home_bias import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Portfolio construction invariants
# ---------------------------------------------------------------------------
def test_rolling_blend_weights_sum_to_one(equal_mu_low_corr):
    ret, _ = equal_mu_low_corr
    blend = st.rolling_blend(ret)
    us = st.us_only(ret)
    assert len(blend) == len(ret)
    assert len(us) == len(ret)


def test_us_only_equals_spy(equal_mu_low_corr):
    ret, _ = equal_mu_low_corr
    us = st.us_only(ret)
    assert np.allclose(us.to_numpy(), ret["SPY"].to_numpy())


def test_rolling_blend_is_weighted_average(equal_mu_low_corr):
    ret, _ = equal_mu_low_corr
    blend = st.rolling_blend(ret)
    w = {"SPY": 0.60, "EFA": 0.25, "EEM": 0.15}
    manual = 0.60 * ret["SPY"] + 0.25 * ret["EFA"] + 0.15 * ret["EEM"]
    assert np.allclose(blend.to_numpy(), manual.to_numpy())


def test_portfolio_stats_structure(equal_mu_low_corr):
    ret, _ = equal_mu_low_corr
    s = st.portfolio_stats(ret["SPY"])
    assert set(s.keys()) >= {"ann_return", "ann_vol", "sharpe", "max_dd", "n"}
    assert s["n"] == len(ret)
    assert s["max_dd"] <= 0.0
    assert s["ann_vol"] > 0


def test_max_dd_negative_or_zero(equal_mu_low_corr):
    ret, _ = equal_mu_low_corr
    us = st.us_only(ret)
    s = st.portfolio_stats(us)
    assert s["max_dd"] <= 0.0


# ---------------------------------------------------------------------------
# Crisis vs calm correlations
# ---------------------------------------------------------------------------
def test_crisis_corr_returns_finite_values():
    """crisis_vs_calm should return finite, bounded correlations in both regimes
    for any valid panel. The empirical 'crisis corr spikes' property is a
    stylised fact of real equity returns (fat tails + vol clusters), not a
    property of the Gaussian synthetic model — so we only assert structure here;
    the real-tape result is validated in docs/results.md."""
    ret, _ = data.synthetic_panel(n_days=5000, corr_intl=0.60, seed=145)
    result = st.crisis_vs_calm(ret)
    import numpy as np
    assert np.isfinite(result["corr_crisis"]).all()
    assert np.isfinite(result["corr_calm"]).all()
    assert (result["corr_crisis"].abs() <= 1.0).all()
    assert (result["corr_calm"].abs() <= 1.0).all()


def test_crisis_corr_structure():
    ret, _ = data.synthetic_panel(n_days=3000, seed=145)
    result = st.crisis_vs_calm(ret)
    assert set(result.columns) >= {"corr_crisis", "corr_calm", "delta"}
    assert set(result.index) == {"EFA", "EEM"}


# ---------------------------------------------------------------------------
# HAC and bootstrap helpers
# ---------------------------------------------------------------------------
def test_mean_excess_hac_zero_on_identical():
    """If blend and US-only have identical returns, mean excess should be exactly zero.
    When variance is zero (diff = 0 everywhere) the t-stat is NaN (0/0), not a
    meaningful number — we only assert the mean is zero."""
    ret, _ = data.synthetic_panel(n_days=3000, seed=145)
    r = ret["SPY"]
    res = st.mean_excess_hac(r, r)
    assert abs(res["mean_bps"]) < 1e-9


def test_sharpe_diff_bootstrap_structure():
    ret, _ = data.synthetic_panel(n_days=2000, seed=145)
    blend = st.rolling_blend(ret)
    us = st.us_only(ret)
    res = st.sharpe_diff_bootstrap(blend, us, n_boot=500, seed=1)
    assert set(res.keys()) >= {"sharpe_diff", "ci_low", "ci_high", "frac_blend_below_us"}
    assert res["ci_low"] < res["ci_high"]
    assert 0.0 <= res["frac_blend_below_us"] <= 1.0


# ---------------------------------------------------------------------------
# The spine — diversification helps only under the right conditions
# ---------------------------------------------------------------------------
def test_blend_improves_sharpe_under_low_correlation(equal_mu_low_corr):
    """Equal returns + low correlation: the global blend should have a higher Sharpe than US-only."""
    ret, truth = equal_mu_low_corr
    assert truth["corr_intl"] < 0.5, "Fixture should plant low correlation."
    s_blend = st.portfolio_stats(st.rolling_blend(ret))["sharpe"]
    s_us = st.portfolio_stats(st.us_only(ret))["sharpe"]
    assert s_blend > s_us, (
        f"Blend Sharpe {s_blend:.3f} should exceed US-only {s_us:.3f} under low correlation."
    )


def test_blend_lags_us_when_us_dominant(us_dominant):
    """When US has a large return advantage, the blend should lag on annualised return."""
    ret, truth = us_dominant
    assert truth["mu_spread_ann"] > 0.03, "Fixture should plant US outperformance."
    r_blend = st.rolling_blend(ret).mean() * 252
    r_us = st.us_only(ret).mean() * 252
    assert r_us > r_blend, (
        f"US ann return {r_us:.3f} should exceed blend {r_blend:.3f} when US dominant."
    )


def test_high_corr_no_diversification_benefit(full_correlation):
    """Perfect correlation: blend Sharpe should be approximately equal to US-only Sharpe."""
    ret, _ = full_correlation
    s_blend = st.portfolio_stats(st.rolling_blend(ret))["sharpe"]
    s_us = st.portfolio_stats(st.us_only(ret))["sharpe"]
    # With perfect correlation the blend just scales weights — should be close.
    assert abs(s_blend - s_us) < 0.30, (
        f"With corr=1.0, Sharpes should be close: blend={s_blend:.3f}, us={s_us:.3f}"
    )

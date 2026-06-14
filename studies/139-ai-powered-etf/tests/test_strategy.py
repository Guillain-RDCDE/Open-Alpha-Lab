"""The CAPM decomposition, HAC inference, and the study's spine: the engine recovers
planted alpha when it exists, and reads near-zero when the null holds."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_powered_etf import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# daily_returns
# ---------------------------------------------------------------------------
def test_daily_returns_shape(null_tape):
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    assert len(rets) == len(prices) - 1
    assert list(rets.columns) == ["aieq", "spy", "ivv"]


def test_daily_returns_no_lookahead(null_tape):
    """Return at t uses prices[t] and prices[t-1] only; no future info."""
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    # First return: prices[1] / prices[0] - 1
    for col in ["aieq", "spy"]:
        expected = prices[col].iloc[1] / prices[col].iloc[0] - 1
        assert abs(rets[col].iloc[0] - expected) < 1e-12


# ---------------------------------------------------------------------------
# capm_decompose
# ---------------------------------------------------------------------------
def test_capm_decompose_null_alpha_near_zero(null_tape):
    """With alpha_ann=0 planted, recovered CAPM alpha must be plausibly near zero.

    With 8%/yr idiosyncratic vol and 600 days, the standard error of the annualised
    alpha estimate is ~5%, so we can only demand the t-stat stays within reason —
    the point estimate can look large by chance on a single seed.
    """
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    decomp = st.capm_decompose(rets["aieq"], rets["spy"])
    # Check the alpha residual series directly: its daily mean / SE should be small
    alpha_series = decomp["alpha_series"]
    tstat = float(alpha_series.mean() / (alpha_series.std(ddof=1) / np.sqrt(len(alpha_series))))
    # With 600 obs, |t| < 3 is consistent with null — a |t|>5 would be suspicious
    assert abs(tstat) < 3.5


def test_capm_decompose_beta_close_to_planted(null_tape):
    prices, truth = null_tape
    rets = st.daily_returns(prices)
    decomp = st.capm_decompose(rets["aieq"], rets["spy"])
    # Beta should be close to the planted 1.05 (some estimation noise on 600 days)
    assert abs(decomp["beta"] - truth["beta"]) < 0.15


def test_capm_decompose_positive_alpha_detected(positive_tape):
    """With +5%/yr planted alpha on a long tape, t-stat on alpha should be positive.

    With only 600 days and 8% idio vol, we cannot always detect 5%/yr reliably in point
    estimate — but use a 2000-day tape to confirm detection works in principle.
    """
    prices_long, _ = data.synthetic_daily(n_days=2000, alpha_ann=0.05, beta=1.05, seed=42)
    rets_long = st.daily_returns(prices_long)
    decomp_long = st.capm_decompose(rets_long["aieq"], rets_long["spy"])
    # With 2000 days, a 5%/yr planted alpha should be recoverable
    assert decomp_long["alpha_ann_pct"] > 1.0


def test_capm_decompose_negative_alpha_detected(negative_tape):
    """With -4.4%/yr planted alpha, recovered alpha should be clearly negative."""
    prices, _ = negative_tape
    rets = st.daily_returns(prices)
    decomp = st.capm_decompose(rets["aieq"], rets["spy"])
    assert decomp["alpha_ann_pct"] < -1.0


def test_capm_decompose_r_squared_between_0_and_1(null_tape):
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    decomp = st.capm_decompose(rets["aieq"], rets["spy"])
    assert 0.0 <= decomp["r_squared"] <= 1.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(null_tape):
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    result = st.summarize(rets["aieq"], rets["spy"])
    required = [
        "n", "n_years", "cagr_fund_pct", "cagr_market_pct",
        "sharpe_fund", "sharpe_market",
        "alpha_daily_bps", "alpha_ann_pct", "beta", "r_squared",
        "tstat_alpha_hac", "excess_mean_bps", "tstat_excess_hac",
        "max_dd_fund_pct", "max_dd_market_pct",
    ]
    for k in required:
        assert k in result, f"Missing key: {k}"


def test_summarize_higher_alpha_gives_higher_cagr_fund():
    """CAGR of fund must be monotone in planted alpha (market held fixed, seed fixed)."""
    res = []
    for alpha in (-0.05, 0.0, 0.05):
        prices, _ = data.synthetic_daily(n_days=500, alpha_ann=alpha, seed=42)
        rets = st.daily_returns(prices)
        r = st.summarize(rets["aieq"], rets["spy"])
        res.append(r["cagr_fund_pct"])
    assert res[0] < res[1] < res[2]


def test_summarize_tstat_more_negative_with_larger_negative_alpha():
    """Larger negative alpha planted -> more negative t-stat (on long-enough tape)."""
    prices_mild, _ = data.synthetic_daily(n_days=2000, alpha_ann=-0.03, seed=99)
    prices_severe, _ = data.synthetic_daily(n_days=2000, alpha_ann=-0.10, seed=99)
    r_mild = st.daily_returns(prices_mild)
    r_severe = st.daily_returns(prices_severe)
    t_mild = st.summarize(r_mild["aieq"], r_mild["spy"])["tstat_alpha_hac"]
    t_severe = st.summarize(r_severe["aieq"], r_severe["spy"])["tstat_alpha_hac"]
    assert t_severe < t_mild


def test_max_drawdown_is_non_positive(null_tape):
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    r = st.summarize(rets["aieq"], rets["spy"])
    assert r["max_dd_fund_pct"] <= 0.0
    assert r["max_dd_market_pct"] <= 0.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_for_zero_mean():
    arr = np.zeros(200)
    assert not np.isfinite(st._hac_tstat(arr))  # se=0 → nan


def test_hac_tstat_significant_for_large_mean():
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=0.005, scale=0.01, size=500)
    t = st._hac_tstat(arr)
    assert abs(t) > 2.0


def test_rolling_alpha_length(null_tape):
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    ra = st.rolling_alpha(rets["aieq"], rets["spy"], window=252)
    # Should have n - window + 1 points (after daily_returns, len = n_days-1)
    assert len(ra) == (len(prices) - 1) - 252 + 1

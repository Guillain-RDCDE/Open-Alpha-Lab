"""The CAPM decomposition, HAC inference, and the study's spine: the engine recovers
planted alpha when it exists, and reads near-zero when the null holds."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spac_performance import data, strategy as st  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
SPAK_CACHE = os.path.join(STUDY_CACHE, "spak_daily.parquet")


# ---------------------------------------------------------------------------
# daily_returns
# ---------------------------------------------------------------------------
def test_daily_returns_shape(null_tape):
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    assert len(rets) == len(prices) - 1
    assert list(rets.columns) == ["spac", "spy"]


def test_daily_returns_no_lookahead(null_tape):
    """Return at t uses prices[t] and prices[t-1] only; no future info."""
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    for col in ["spac", "spy"]:
        expected = prices[col].iloc[1] / prices[col].iloc[0] - 1
        assert abs(rets[col].iloc[0] - expected) < 1e-12


# ---------------------------------------------------------------------------
# capm_decompose
# ---------------------------------------------------------------------------
def test_capm_decompose_null_alpha_near_zero(null_tape):
    """With alpha_ann=0 planted, recovered CAPM alpha is plausibly near zero.

    With 45%/yr idiosyncratic vol and 600 days, the SE of the annualised alpha
    estimate is very large — we only demand the t-stat stays within reason.
    """
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    decomp = st.capm_decompose(rets["spac"], rets["spy"])
    alpha_series = decomp["alpha_series"]
    tstat = float(alpha_series.mean() / (alpha_series.std(ddof=1) / np.sqrt(len(alpha_series))))
    assert abs(tstat) < 4.0


def test_capm_decompose_beta_close_to_planted(null_tape):
    prices, truth = null_tape
    rets = st.daily_returns(prices)
    decomp = st.capm_decompose(rets["spac"], rets["spy"])
    # With high idio vol (45%/yr), beta estimate has wide SE; allow ±0.5
    assert abs(decomp["beta"] - truth["beta"]) < 0.5


def test_capm_decompose_positive_alpha_detected():
    """With +10%/yr planted alpha on a long tape, recovered alpha must be positive."""
    prices_long, _ = data.synthetic_daily(n_days=2000, alpha_ann=0.10, beta=1.5, seed=42)
    rets_long = st.daily_returns(prices_long)
    decomp_long = st.capm_decompose(rets_long["spac"], rets_long["spy"])
    assert decomp_long["alpha_ann_pct"] > 1.0


def test_capm_decompose_negative_alpha_detected(negative_tape):
    """With -30%/yr planted alpha, recovered alpha should be clearly negative."""
    prices, _ = negative_tape
    rets = st.daily_returns(prices)
    decomp = st.capm_decompose(rets["spac"], rets["spy"])
    assert decomp["alpha_ann_pct"] < -5.0


def test_capm_decompose_r_squared_between_0_and_1(null_tape):
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    decomp = st.capm_decompose(rets["spac"], rets["spy"])
    assert 0.0 <= decomp["r_squared"] <= 1.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(null_tape):
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    result = st.summarize(rets["spac"], rets["spy"])
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
    for alpha in (-0.20, 0.0, 0.10):
        prices, _ = data.synthetic_daily(n_days=500, alpha_ann=alpha, seed=42)
        rets = st.daily_returns(prices)
        r = st.summarize(rets["spac"], rets["spy"])
        res.append(r["cagr_fund_pct"])
    assert res[0] < res[1] < res[2]


def test_summarize_tstat_more_negative_with_larger_negative_alpha():
    """Larger negative alpha planted -> more negative t-stat (on long-enough tape)."""
    prices_mild, _ = data.synthetic_daily(n_days=2000, alpha_ann=-0.10, seed=99)
    prices_severe, _ = data.synthetic_daily(n_days=2000, alpha_ann=-0.30, seed=99)
    r_mild = st.daily_returns(prices_mild)
    r_severe = st.daily_returns(prices_severe)
    t_mild = st.summarize(r_mild["spac"], r_mild["spy"])["tstat_alpha_hac"]
    t_severe = st.summarize(r_severe["spac"], r_severe["spy"])["tstat_alpha_hac"]
    assert t_severe < t_mild


def test_max_drawdown_is_non_positive(null_tape):
    prices, _ = null_tape
    rets = st.daily_returns(prices)
    r = st.summarize(rets["spac"], rets["spy"])
    assert r["max_dd_fund_pct"] <= 0.0
    assert r["max_dd_market_pct"] <= 0.0


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_for_zero_mean():
    arr = np.zeros(200)
    assert not np.isfinite(st._hac_tstat(arr))


def test_hac_tstat_significant_for_large_mean():
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=0.005, scale=0.01, size=500)
    t = st._hac_tstat(arr)
    assert abs(t) > 2.0


# ---------------------------------------------------------------------------
# Real-tape smoke test (skipped offline/CI when cache absent)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(SPAK_CACHE), reason="real-tape cache absent offline/CI")
def test_spak_alpha_is_clearly_negative():
    """On the real tape, SPAK Jensen alpha must be clearly negative (well below -10%/yr)."""
    from spac_performance.data import load_spak_vs_spy
    prices = load_spak_vs_spy(fetch=False, cache_dir=STUDY_CACHE)
    rets = st.daily_returns(prices)
    res = st.summarize(rets["SPAK"], rets["SPY"])
    assert res["alpha_ann_pct"] < -10.0
    assert res["cagr_fund_pct"] < res["cagr_market_pct"]

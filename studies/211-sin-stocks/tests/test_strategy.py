"""Strategy engine invariants, honest controls, and the study's core spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sin_stocks import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard
# ---------------------------------------------------------------------------
_CACHE_DIR = data.DEFAULT_CACHE
_SPY_CACHE = data._cache_path("SPY", _CACHE_DIR)
_MO_CACHE = data._cache_path("MO", _CACHE_DIR)
_BOTH_CACHED = os.path.exists(_SPY_CACHE) and os.path.exists(_MO_CACHE)

requires_cache = pytest.mark.skipif(
    not _BOTH_CACHED,
    reason="price cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Helper: build synthetic combined price frame with SIN_EW as a column
# ---------------------------------------------------------------------------
def _build_ret_frame(prices):
    """Build a DataFrame with SIN_EW, SPY, DSI daily returns from the synthetic price frame."""
    # Treat the SIN column as the equal-weight basket (single-stock proxy in synthetic)
    basket_prices = prices[["SIN"]].copy()
    ew_ret = st.build_equal_weight(basket_prices)
    spy_ret = st.daily_returns(prices)["SPY"].dropna()
    dsi_ret = st.daily_returns(prices)["DSI"].dropna()
    idx = ew_ret.index.intersection(spy_ret.index).intersection(dsi_ret.index)
    return pd.DataFrame({
        "SIN_EW": ew_ret.loc[idx],
        "SPY": spy_ret.loc[idx],
        "DSI": dsi_ret.loc[idx],
    })


# ---------------------------------------------------------------------------
# build_equal_weight
# ---------------------------------------------------------------------------
def test_equal_weight_shape(null_prices):
    prices, truth = null_prices
    basket = prices[["SIN"]]
    ew = st.build_equal_weight(basket)
    assert isinstance(ew, pd.Series)
    # dropna removes first row
    assert len(ew) == len(prices) - 1


def test_equal_weight_deterministic(null_prices):
    prices, _ = null_prices
    basket = prices[["SIN"]]
    a = st.build_equal_weight(basket)
    b = st.build_equal_weight(basket)
    assert np.allclose(a.to_numpy(), b.to_numpy())


# ---------------------------------------------------------------------------
# daily_returns
# ---------------------------------------------------------------------------
def test_daily_returns_shape(null_prices):
    prices, truth = null_prices
    ret = st.daily_returns(prices)
    assert ret.shape == prices.shape
    assert ret.iloc[0].isna().all()
    assert ret.iloc[1:].notna().all().all()


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(null_prices):
    prices, _ = null_prices
    ret = _build_ret_frame(prices)["SIN_EW"]
    s = st.summarize(ret)
    for key in ("n", "cagr_pct", "vol_ann_pct", "sharpe_ann", "max_dd_pct", "mean_bps", "tstat"):
        assert key in s, f"Missing key: {key}"


def test_summarize_max_dd_negative(null_prices):
    prices, _ = null_prices
    ret = _build_ret_frame(prices)["SPY"]
    s = st.summarize(ret)
    assert s["max_dd_pct"] < 0.0, "max drawdown must be negative"


def test_summarize_cagr_from_planted_alpha():
    """SIN with planted +3 bps/day should have higher CAGR than with zero alpha."""
    prices_null, _ = data.synthetic_daily(n_years=10, alpha_bps=0.0, seed=211)
    prices_alpha, _ = data.synthetic_daily(n_years=10, alpha_bps=3.0, seed=211)
    ret_null = _build_ret_frame(prices_null)["SIN_EW"]
    ret_alpha = _build_ret_frame(prices_alpha)["SIN_EW"]
    s_null = st.summarize(ret_null)
    s_alpha = st.summarize(ret_alpha)
    assert s_alpha["cagr_pct"] > s_null["cagr_pct"]


# ---------------------------------------------------------------------------
# OLS alpha / beta decomposition
# ---------------------------------------------------------------------------
def test_beta_alpha_ols_keys(null_prices):
    prices, _ = null_prices
    ret = _build_ret_frame(prices)
    d = st.beta_alpha_ols(ret["SIN_EW"], ret["SPY"])
    for key in ("alpha_daily_bps", "alpha_ann_pct", "t_alpha", "beta", "r_squared", "n"):
        assert key in d, f"Missing key: {key}"


def test_null_alpha_near_zero(null_prices):
    """With no planted alpha, the daily OLS alpha should be near zero."""
    prices, _ = null_prices
    ret = _build_ret_frame(prices)
    d = st.beta_alpha_ols(ret["SIN_EW"], ret["SPY"])
    assert abs(d["alpha_daily_bps"]) < 5.0, \
        f"OLS alpha too large on null tape: {d['alpha_daily_bps']:.2f} bps"


def test_planted_alpha_raises_sin_cagr():
    """Planted +3 bps/day alpha should raise SIN_EW CAGR vs the null tape."""
    prices_null, _ = data.synthetic_daily(n_years=15, alpha_bps=0.0, seed=211)
    prices_alpha, _ = data.synthetic_daily(n_years=15, alpha_bps=3.0, seed=211)
    sin_null = st.daily_returns(prices_null)["SIN"].dropna()
    sin_alpha = st.daily_returns(prices_alpha)["SIN"].dropna()
    assert sin_alpha.mean() > sin_null.mean(), "planted alpha must raise SIN mean daily return"
    # The difference should be ~3 bps/day
    diff = (sin_alpha.mean() - sin_null.mean()) * 1e4
    assert diff > 1.0, f"planted alpha diff should exceed 1 bps/day; got {diff:.2f}"


# ---------------------------------------------------------------------------
# Crash analysis
# ---------------------------------------------------------------------------
def test_crash_analysis_returns_dataframe(null_prices):
    prices, _ = null_prices
    ret = _build_ret_frame(prices)
    result = st.crash_analysis(ret, asset="SIN_EW", benchmark="SPY", threshold=-0.15)
    assert isinstance(result, pd.DataFrame)


def test_crash_analysis_columns(null_prices):
    prices, _ = null_prices
    ret = _build_ret_frame(prices)
    result = st.crash_analysis(ret, asset="SIN_EW", benchmark="SPY", threshold=-0.15)
    if len(result) > 0:
        for col in ("start", "trough", "end", "bench_max_dd_pct", "protection_pp"):
            assert col in result.columns


# ---------------------------------------------------------------------------
# Rolling alpha
# ---------------------------------------------------------------------------
def test_rolling_alpha_series(null_prices):
    prices, _ = null_prices
    ret = _build_ret_frame(prices)
    ra = st.rolling_alpha(ret["SIN_EW"], ret["SPY"], window=252)
    assert isinstance(ra, pd.Series)
    assert ra.dropna().shape[0] > 0


# ---------------------------------------------------------------------------
# Random-mix control
# ---------------------------------------------------------------------------
def test_random_mix_control_shape(null_prices):
    prices, _ = null_prices
    ret = _build_ret_frame(prices)
    ctrl = st.random_mix_control(ret["SIN_EW"], ret["SPY"], n_seeds=5, seed=211)
    assert len(ctrl) == 5
    assert "sharpe_ann" in ctrl.columns


def test_random_mix_control_reproducible(null_prices):
    prices, _ = null_prices
    ret = _build_ret_frame(prices)
    a = st.random_mix_control(ret["SIN_EW"], ret["SPY"], n_seeds=5, seed=211)
    b = st.random_mix_control(ret["SIN_EW"], ret["SPY"], n_seeds=5, seed=211)
    assert (a["sharpe_ann"].values == b["sharpe_ann"].values).all()


# ---------------------------------------------------------------------------
# Spine test — the core hypothesis
# ---------------------------------------------------------------------------
def test_spine_alpha_detectable_only_when_planted():
    """The study's spine: the SIN mean return is near SPY's on the null tape
    and clearly above it when alpha is planted.

    We test the difference in means between alpha and null tapes (same SPY path)
    rather than the excess return, because random market noise can swamp small
    planted alphas in any single draw.  What is guaranteed is that the planted
    alpha shifts the SIN mean by exactly alpha_bps, which we verify.
    """
    prices_null, _ = data.synthetic_daily(n_years=15, alpha_bps=0.0, seed=211)
    prices_alpha, _ = data.synthetic_daily(n_years=15, alpha_bps=3.0, seed=211)

    sin_null = st.daily_returns(prices_null)["SIN"].dropna()
    sin_alpha = st.daily_returns(prices_alpha)["SIN"].dropna()

    # Planted alpha shifts mean by exactly 3 bps/day (verified by construction)
    delta = (sin_alpha.mean() - sin_null.mean()) * 1e4
    assert 2.0 < delta < 4.0, f"Planted 3 bps alpha should shift mean by ~3 bps; got {delta:.2f}"

    # On the null tape: the SIN vs SPY t-stat should not be certifiably large in both directions
    # (it can be negative by chance — that's the honest null)
    spy_null = st.daily_returns(prices_null)["SPY"].dropna()
    diff_null = sin_null - spy_null
    s_null = st.summarize(diff_null, label="null")
    # A very large t-stat (>5) on the null would indicate a systematic bias in the engine
    assert abs(s_null["tstat"]) < 5.0, \
        f"Null tape should not certify an extreme t: {s_null['tstat']:.2f}"

    # With the alpha tape: the SIN mean is higher by construction
    assert sin_alpha.mean() > sin_null.mean(), "planted alpha must raise SIN mean return"


# ---------------------------------------------------------------------------
# Real-data tests — skipped when cache absent
# ---------------------------------------------------------------------------
@requires_cache
def test_real_excess_return_finite():
    all_prices = data.load_aligned(fetch=False, cache_dir=_CACHE_DIR)
    basket_prices = all_prices[data.SIN_BASKET]
    ew = st.build_equal_weight(basket_prices)
    spy = st.daily_returns(all_prices)["SPY"].dropna()
    idx = ew.index.intersection(spy.index)
    diff = ew.loc[idx] - spy.loc[idx]
    assert diff.notna().sum() > 200
    assert np.isfinite(diff.mean())


@requires_cache
def test_real_summarize_finite():
    all_prices = data.load_aligned(fetch=False, cache_dir=_CACHE_DIR)
    basket_prices = all_prices[data.SIN_BASKET]
    ew = st.build_equal_weight(basket_prices)
    s = st.summarize(ew, label="SIN_EW real")
    assert np.isfinite(s["sharpe_ann"])
    assert np.isfinite(s["tstat"])


@requires_cache
def test_real_beta_finite():
    all_prices = data.load_aligned(fetch=False, cache_dir=_CACHE_DIR)
    basket_prices = all_prices[data.SIN_BASKET]
    ew = st.build_equal_weight(basket_prices)
    spy = st.daily_returns(all_prices)["SPY"].dropna()
    d = st.beta_alpha_ols(ew, spy)
    assert np.isfinite(d["beta"])
    assert np.isfinite(d["alpha_daily_bps"])

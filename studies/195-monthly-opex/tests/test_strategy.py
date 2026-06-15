"""Strategy engine invariants and the study's spine: the vol signal appears only when
we plant it; the return signal is absent on both synthetic and real tapes."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monthly_opex import data, strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard
# ---------------------------------------------------------------------------
_CACHE_PATH = data._cache_path("SPY", data.DEFAULT_CACHE)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_CACHE_PATH),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def test_daily_range_positive():
    bars, _ = data.synthetic_daily(n_years=5, seed=195)
    rng = st.daily_range_pct(bars)
    assert (rng >= 0).all()
    assert rng.name == "range_pct"


def test_log_volume_demeaned_mean_near_zero():
    """Year-demeaning should give an annual mean ≈ 0 within each year."""
    bars, _ = data.synthetic_daily(n_years=10, seed=195)
    lv_dm = st.log_volume_demeaned(bars)
    year = pd.DatetimeIndex(bars.index).year
    annual_means = lv_dm.groupby(year).mean()
    assert (annual_means.abs() < 0.05).all(), f"Year-demeaned means not near zero: {annual_means}"


def test_daily_return_close_to_close():
    bars, _ = data.synthetic_daily(n_years=5, seed=195)
    ret = st.daily_return(bars)
    # First return is NaN (no prior close)
    assert np.isnan(ret.iloc[0])
    # Subsequent values match close-to-close ratio
    r2 = bars["close"].iloc[2] / bars["close"].iloc[1] - 1.0
    assert abs(ret.iloc[2] - r2) < 1e-10


# ---------------------------------------------------------------------------
# compare_opex_vs_baseline invariants
# ---------------------------------------------------------------------------

def test_compare_output_keys():
    bars, _ = data.synthetic_daily(n_years=10, seed=195)
    idx = pd.DatetimeIndex(bars.index)
    mask = data.opex_week_mask(idx)
    rng = st.daily_range_pct(bars)
    res = st.compare_opex_vs_baseline(rng, mask, "test_label")
    expected = {"label", "n_opex", "n_baseline", "mean_opex", "mean_baseline",
                "std_opex", "std_baseline", "diff", "tstat"}
    assert set(res.keys()) == expected


def test_compare_diff_equals_mean_difference():
    bars, _ = data.synthetic_daily(n_years=10, seed=195)
    idx = pd.DatetimeIndex(bars.index)
    mask = data.opex_week_mask(idx)
    rng = st.daily_range_pct(bars)
    res = st.compare_opex_vs_baseline(rng, mask)
    assert abs(res["diff"] - (res["mean_opex"] - res["mean_baseline"])) < 1e-10


def test_compare_baseline_count():
    """n_opex + n_baseline must equal total non-NaN observations."""
    bars, _ = data.synthetic_daily(n_years=10, seed=195)
    idx = pd.DatetimeIndex(bars.index)
    mask = data.opex_week_mask(idx)
    rng = st.daily_range_pct(bars).dropna()
    res = st.compare_opex_vs_baseline(rng, mask)
    assert res["n_opex"] + res["n_baseline"] == len(rng)


# ---------------------------------------------------------------------------
# The synthetic positive control — the engine finds the effect only when planted
# ---------------------------------------------------------------------------

def test_vol_signal_appears_only_when_planted():
    """On a null tape vol t-stat should be small; on a vol_tape it should be large."""
    null_bars, _ = data.synthetic_daily(n_years=30, vol_premium=0.0, seed=195)
    vol_bars, _ = data.synthetic_daily(n_years=30, vol_premium=0.50, seed=195)

    def vol_t(bars):
        idx = pd.DatetimeIndex(bars.index)
        mask = data.opex_week_mask(idx)
        lv = st.log_volume_demeaned(bars)
        return st.compare_opex_vs_baseline(lv, mask)["tstat"]

    t_null = vol_t(null_bars)
    t_planted = vol_t(vol_bars)
    assert abs(t_null) < 2.0, f"Null tape should have |t| < 2, got {t_null:.2f}"
    assert t_planted > 3.0, f"Planted vol tape should have t > 3, got {t_planted:.2f}"


def test_ret_signal_appears_only_when_planted():
    """On a null tape the return t-stat should be noise; on a ret_tape it should be large."""
    null_bars, _ = data.synthetic_daily(n_years=30, vol_premium=0.0, ret_premium_bps=0.0, seed=195)
    ret_bars, _ = data.synthetic_daily(n_years=30, vol_premium=0.0, ret_premium_bps=15.0, seed=195)

    def ret_t(bars):
        idx = pd.DatetimeIndex(bars.index)
        mask = data.opex_week_mask(idx)
        r = st.daily_return(bars).dropna()
        return st.compare_opex_vs_baseline(r, mask)["tstat"]

    t_null = ret_t(null_bars)
    t_planted = ret_t(ret_bars)
    assert abs(t_null) < 2.0, f"Null tape should have |t| < 2, got {t_null:.2f}"
    assert t_planted > 3.0, f"Planted return tape should have t > 3, got {t_planted:.2f}"


# ---------------------------------------------------------------------------
# Three-leg test functions
# ---------------------------------------------------------------------------

def test_opex_vol_test_keys(null_tape):
    bars, _ = null_tape
    res = st.opex_vol_test(bars)
    assert set(res.keys()) == {"range_day", "range_week", "vol_day", "vol_week"}


def test_opex_return_test_keys(null_tape):
    bars, _ = null_tape
    res = st.opex_return_test(bars)
    assert set(res.keys()) == {"day", "week"}


def test_opex_incremental_monthly_keys(null_tape):
    bars, _ = null_tape
    res = st.opex_incremental_monthly_test(bars)
    assert "range_monthly_only" in res
    assert "vol_monthly_only" in res
    assert "ret_monthly_only" in res
    assert "range_quarterly" in res
    assert "vol_quarterly" in res


def test_opex_pre_post_split_keys(null_tape):
    bars, _ = null_tape
    res = st.opex_pre_post_2010_split(bars)
    assert "pre_2010" in res and "post_2010" in res
    for period in ("pre_2010", "post_2010"):
        assert "range" in res[period] and "vol" in res[period] and "ret" in res[period]


# ---------------------------------------------------------------------------
# summarize function
# ---------------------------------------------------------------------------

def test_summarize_keys():
    bars, _ = data.synthetic_daily(n_years=10, seed=195)
    wr = st.opex_week_trade_returns(bars)
    s = st.summarize(wr[wr != 0])
    assert set(s.keys()) == {"n", "mean_bps", "sharpe_ann", "cagr", "tstat"}


def test_summarize_short_returns_nan():
    s = st.summarize(pd.Series([0.01, 0.02]))
    assert all(np.isnan(v) for v in s.values())


def test_opex_week_trade_returns_only_in_opex_weeks(null_tape):
    """The long-week strategy returns should be zero on non-OpEx days."""
    bars, _ = null_tape
    ret_series = st.opex_week_trade_returns(bars, long_week=True)
    idx = pd.DatetimeIndex(ret_series.index)
    mask = data.opex_week_mask(idx).reindex(ret_series.index).fillna(False)
    # Where mask is False, return must be zero
    non_opex = ret_series[~mask.values]
    assert (non_opex == 0).all()


# ---------------------------------------------------------------------------
# Cache-gated real-data tests
# ---------------------------------------------------------------------------

@requires_cache
def test_real_vol_signal_is_significant():
    """On the real SPY tape, OpEx-day volume signal must clear |t| >= 2 (driven by quarterly)."""
    bars = data.fetch_daily("SPY", start="1993-01-01", fetch=False)
    res = st.opex_vol_test(bars)
    t = res["vol_day"]["tstat"]
    assert abs(t) >= 2.0, f"Expected |t| >= 2 for OpEx-day volume, got t={t:.2f}"


@requires_cache
def test_real_return_signal_is_not_significant_for_week():
    """On the real tape, OpEx *week* return HAC t should be well below 2."""
    bars = data.fetch_daily("SPY", start="1993-01-01", fetch=False)
    res = st.opex_return_test(bars)
    t_week = res["week"]["tstat"]
    assert abs(t_week) < 2.0, f"OpEx-week return t={t_week:.2f} unexpectedly significant"


@requires_cache
def test_real_incremental_monthly_vol_is_null():
    """The monthly-only (non-quarterly) OpEx vol signal should be insignificant."""
    bars = data.fetch_daily("SPY", start="1993-01-01", fetch=False)
    res = st.opex_incremental_monthly_test(bars)
    t = res["vol_monthly_only"]["tstat"]
    assert abs(t) < 2.0, f"Monthly-only vol t={t:.2f} unexpectedly significant"

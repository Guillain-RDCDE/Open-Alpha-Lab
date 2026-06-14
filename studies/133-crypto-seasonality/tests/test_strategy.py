"""Monthly-return engine invariants, per-month stats, the seasonal rule, and the spine:
the engine finds a planted October boost and reads ≈ 0 when no boost is present."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crypto_seasonality import strategy as st  # noqa: E402
from crypto_seasonality.data import synthetic_daily  # noqa: E402


# ---- monthly_returns invariants -------------------------------------------
def test_monthly_returns_shape(null_tape):
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    assert list(mret.columns) == ["year", "month", "month_name", "log_ret", "pct_ret", "n_days"]
    # Should have 12 months per year for ~11 years (some boundary months may be partial).
    assert len(mret) >= 11 * 11
    assert mret["month"].between(1, 12).all()


def test_monthly_returns_all_months_covered(null_tape):
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    # All 12 calendar months should appear in an 11-year tape.
    assert set(mret["month"].unique()) == set(range(1, 13))


def test_monthly_returns_no_lookahead(null_tape):
    """Entry price is the first-day open, not the prior close — no look-ahead."""
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    # pct_ret = exp(log_ret) - 1 must match the month's first open → last close ratio.
    bars_copy = bars.copy()
    bars_copy["ym"] = bars_copy.index.to_series().dt.to_period("M")
    for _, row in mret.head(20).iterrows():
        grp = bars_copy[bars_copy["ym"] == pd.Period(year=int(row["year"]), month=int(row["month"]), freq="M")]
        entry = grp["open"].iloc[0]
        exit_ = grp["close"].iloc[-1]
        expected = np.log(exit_ / entry)
        assert abs(row["log_ret"] - expected) < 1e-9


def test_pct_ret_equals_expm1_log_ret(null_tape):
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    assert np.allclose(mret["pct_ret"], np.expm1(mret["log_ret"]), atol=1e-10)


# ---- month_stats -----------------------------------------------------------
def test_month_stats_has_all_months(null_tape):
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    stats = st.month_stats(mret)
    assert len(stats) == 12
    assert list(stats["month"]) == list(range(1, 13))


def test_month_stats_null_has_no_bonferroni_sig(null_tape):
    """On a null tape (no planted effect) no month should clear the Bonferroni bar."""
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    stats = st.month_stats(mret)
    assert not stats["bonferroni_sig"].any()


def test_month_stats_boost_detected(uptober_tape):
    """With a large planted October boost, October's t-stat should be clearly elevated."""
    bars, _ = uptober_tape
    mret = st.monthly_returns(bars)
    stats = st.month_stats(mret)
    oct_t = stats.loc[stats["month"] == 10, "tstat_vs_baseline"].iloc[0]
    # With n=11 and a 50% log-return boost, the effect is large — t > 2 expected.
    assert oct_t > 2.0


# ---- seasonal_strategy and summarize ---------------------------------------
def test_seasonal_strategy_columns(null_tape):
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    df = st.seasonal_strategy(mret, long_months=[10])
    assert set(["year", "month", "bh_log_ret", "strategy_log_ret", "diff"]).issubset(df.columns)


def test_seasonal_strategy_in_month_equals_bh(null_tape):
    """In long months the strategy return must equal buy-and-hold."""
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    df = st.seasonal_strategy(mret, long_months=[10])
    in_long = df[df["month"] == 10]
    assert np.allclose(in_long["strategy_log_ret"], in_long["bh_log_ret"])


def test_seasonal_strategy_out_months_flat(null_tape):
    """Outside long months the strategy return must be 0 (flat / cash)."""
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    df = st.seasonal_strategy(mret, long_months=[10])
    out = df[df["month"] != 10]
    assert np.allclose(out["strategy_log_ret"], 0.0)


def test_summarize_basic(null_tape):
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    s = st.summarize(mret["log_ret"].to_numpy(), label="test")
    assert s["n_months"] == len(mret)
    assert np.isfinite(s["tstat"])
    assert np.isfinite(s["sharpe_ann"])


def test_bh_vs_strategy_returns_dict(null_tape):
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    result = st.bh_vs_strategy(mret, long_months=[10])
    assert "bh" in result and "strategy" in result
    assert "diff_tstat" in result
    assert np.isfinite(result["diff_tstat"])
    assert result["fraction_months_invested"] == pytest.approx(1 / 12)


# ---- the spine: engine finds planted effect, reads ~0 without it ----------
def test_engine_detects_planted_uptober(uptober_tape):
    """With a large October boost, October's t-stat exceeds the null tape's max."""
    bars, _ = uptober_tape
    mret = st.monthly_returns(bars)
    stats = st.month_stats(mret)
    oct_t = abs(stats.loc[stats["month"] == 10, "tstat_vs_baseline"].iloc[0])
    other_t = abs(stats.loc[stats["month"] != 10, "tstat_vs_baseline"]).max()
    assert oct_t > other_t


def test_engine_no_signal_on_null(null_tape):
    """On a null tape, no month's |t| should be very large (engine is not hallucinating)."""
    bars, _ = null_tape
    mret = st.monthly_returns(bars)
    stats = st.month_stats(mret)
    max_abs_t = stats["tstat_vs_baseline"].abs().max()
    # With n=11 per month on a pure random walk, |t| < 5 almost surely (3-sigma guard).
    assert max_abs_t < 5.0

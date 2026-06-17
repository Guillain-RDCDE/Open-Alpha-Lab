"""Signals, regime sort, predictive regression, timing overlay, and the study's spine:
the contrarian slope is significant only when a sentiment effect is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from baker_wurgler import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = pd.Series(np.random.default_rng(0).standard_normal(50) * 0.05)
    s = st.summarize(r)
    assert set(["mean", "vol", "sharpe", "tstat", "hit_rate", "n"]).issubset(s)


def test_summarize_short_series():
    s = st.summarize(pd.Series([0.01, -0.02]))
    assert np.isnan(s["tstat"]) or isinstance(s["tstat"], float)


def test_summarize_tstat_direction():
    rng = np.random.default_rng(10)
    r = pd.Series(rng.normal(0.05, 0.03, 100))
    s = st.summarize(r)
    assert s["tstat"] > 0


def test_annualise_monthly():
    s = st.summarize(pd.Series([0.01] * 60))
    out = st.annualise_monthly(s, periods=12)
    assert abs(out["mean_ann"] - 12 * 0.01) < 1e-10


# ---------------------------------------------------------------------------
# Regime sort
# ---------------------------------------------------------------------------
def test_regime_sort_labels(null_tape):
    df, _ = null_tape
    reg = st.regime_sort(df, "ret", n_bins=3)
    assert set(reg.index) <= {"low", "mid", "high"}
    assert "mean_ann" in reg.columns


def test_regime_sort_partitions_all(null_tape):
    df, _ = null_tape
    reg = st.regime_sort(df, "ret", n_bins=3)
    # tertiles of ~360 months -> ~120 each
    assert reg["n"].sum() == len(df.dropna(subset=["bw_sent", "ret"]))


def test_regime_sort_contrarian_direction(live_tape):
    """With a planted contrarian effect, low-sentiment months beat high-sentiment months."""
    df, _ = live_tape
    reg = st.regime_sort(df, "ret", n_bins=3)
    assert reg.loc["low", "mean_ann"] > reg.loc["high", "mean_ann"]


# ---------------------------------------------------------------------------
# low_minus_high
# ---------------------------------------------------------------------------
def test_low_minus_high_positive_when_planted(live_tape):
    df, _ = live_tape
    lmh = st.low_minus_high(df, "ret", n_bins=3)
    s = st.summarize(lmh)
    assert s["mean"] > 0.0


# ---------------------------------------------------------------------------
# Predictive regression -- the spine
# ---------------------------------------------------------------------------
def test_regression_keys(null_tape):
    df, _ = null_tape
    pr = st.predictive_regression(df, "ret")
    assert set(["alpha", "beta", "beta_t", "r2", "n"]).issubset(pr)


def test_regression_null_insignificant(null_tape):
    """On the null tape, the slope must be statistically indistinguishable from zero."""
    df, _ = null_tape
    pr = st.predictive_regression(df, "ret")
    assert abs(pr["beta_t"]) < 2.5, f"Null slope should be ~0, got t={pr['beta_t']:.2f}"


def test_regression_planted_negative_significant(live_tape):
    """With a planted contrarian effect, the slope is significantly NEGATIVE."""
    df, _ = live_tape
    pr = st.predictive_regression(df, "ret")
    assert pr["beta"] < 0.0, f"Planted contrarian slope should be negative, got {pr['beta']:.4f}"
    assert pr["beta_t"] < -2.0, f"Planted slope should clear t<-2, got t={pr['beta_t']:.2f}"


def test_regression_deterministic(live_tape):
    df, _ = live_tape
    a = st.predictive_regression(df, "ret")
    b = st.predictive_regression(df, "ret")
    assert a == b


# ---------------------------------------------------------------------------
# Timing overlay
# ---------------------------------------------------------------------------
def test_timing_overlay_columns(null_tape):
    df, _ = null_tape
    ov = st.timing_overlay(df, "ret", n_bins=3)
    assert set(["ret", "position", "gross", "cost", "net", "buyhold"]).issubset(ov.columns)


def test_timing_overlay_positions_valid(null_tape):
    df, _ = null_tape
    ov = st.timing_overlay(df, "ret", n_bins=3, short_high=False)
    assert set(ov["position"].unique()) <= {-1.0, 0.0, 1.0}
    # flat-high variant never shorts
    assert (ov["position"] >= 0).all()


def test_timing_overlay_costs_nonnegative(null_tape):
    df, _ = null_tape
    ov = st.timing_overlay(df, "ret", n_bins=3, one_way_bps=5.0)
    assert (ov["cost"] >= 0).all()
    assert ov["net"].equals(ov["gross"] - ov["cost"])


def test_turnover_count(null_tape):
    df, _ = null_tape
    ov = st.timing_overlay(df, "ret", n_bins=3)
    tc = st.turnover_count(ov)
    assert tc["switches"] >= 0
    assert tc["switches_per_year"] >= 0


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(data.GSPC_CACHE),
    reason="^GSPC cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_regression_is_float():
    mkt = data.fetch_market_monthly(fetch=False)
    sent = data.bw_monthly()
    df = data.merge_sentiment_returns(mkt, sent)
    pr = st.predictive_regression(df, "ret")
    assert isinstance(pr["beta_t"], float) and np.isfinite(pr["beta_t"])


@requires_cache
def test_real_regime_sort_nontrivial():
    mkt = data.fetch_market_monthly(fetch=False)
    sent = data.bw_monthly()
    df = data.merge_sentiment_returns(mkt, sent)
    reg = st.regime_sort(df, "ret", n_bins=3)
    assert len(reg) == 3
    assert reg["n"].min() > 50

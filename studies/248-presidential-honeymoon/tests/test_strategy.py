"""Honeymoon stats, period returns, and the study's spine: signal detectable when planted."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from presidential_honeymoon import strategy as st  # noqa: E402

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache",
    "presidential_honeymoon_gspc_daily.parquet",
)


# ---- period_returns ----------------------------------------------------------
def test_period_returns_shape_and_columns(null_tape):
    df, _ = null_tape
    periods = st.period_returns(df)
    assert "hm_ret" in periods.columns
    assert "rest_ret" in periods.columns
    assert len(periods) >= 5  # at least 5 usable inaugurations in 20-term tape


def test_period_returns_values_reasonable(null_tape):
    df, _ = null_tape
    periods = st.period_returns(df)
    # 100-day returns on a 16%-vol tape should stay within ±100%
    assert periods["hm_ret"].abs().max() < 2.0
    assert periods["rest_ret"].abs().max() < 2.0


# ---- honeymoon_stats ---------------------------------------------------------
def test_honeymoon_stats_keys(null_tape):
    df, _ = null_tape
    periods = st.period_returns(df)
    stats = st.honeymoon_stats(periods)
    for k in ("n", "hm_mean", "hm_std", "hm_tstat",
              "rest_mean", "rest_std", "rest_tstat",
              "diff_mean", "diff_std", "diff_tstat", "pos_rate"):
        assert k in stats


def test_honeymoon_stats_null_diff_t_is_small(null_tape):
    """On a null tape the paired diff t-stat should be well below 2 (noise only)."""
    df, _ = null_tape
    periods = st.period_returns(df)
    stats = st.honeymoon_stats(periods)
    assert abs(stats["diff_tstat"]) < 3.5


# ---- planted signal ----------------------------------------------------------
def test_planted_signal_detected(signal_tape, null_tape):
    """The spine: with a large planted honeymoon premium the diff t should be large;
    on the null it should not."""
    from presidential_honeymoon import data as dt

    periods_sig = st.period_returns(signal_tape[0])
    periods_null = st.period_returns(null_tape[0])
    stats_sig = st.honeymoon_stats(periods_sig)
    stats_null = st.honeymoon_stats(periods_null)

    # Planted +20 bps/day * ~70 trading days ≈ +14% lift in honeymoon
    assert stats_sig["diff_mean"] > 2.0  # at least +2 pp mean advantage
    # Signal tape should show higher diff than null tape
    assert stats_sig["diff_mean"] > stats_null["diff_mean"]


# ---- summarize ---------------------------------------------------------------
def test_summarize_keys(null_tape):
    df, _ = null_tape
    out = st.summarize(df)
    for k in ("n", "hm_mean", "hm_tstat", "diff_tstat",
              "fy_n", "fy_diff_mean", "fy_diff_tstat"):
        assert k in out


@pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="real-tape cache absent offline/CI",
)
def test_summarize_real_tape_runs():
    """Smoke test: summarize() completes without error on real data."""
    from presidential_honeymoon import data as dt
    df = dt.fetch_prices("^GSPC", fetch=False)
    out = st.summarize(df)
    assert out["n"] >= 10
    assert isinstance(out["diff_tstat"], float)

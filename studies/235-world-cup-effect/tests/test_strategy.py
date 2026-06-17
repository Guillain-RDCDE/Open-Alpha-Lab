"""Tests for the strategy layer of Study 235 (World-Cup-Effect).

All tests are offline and deterministic — they use the synthetic generator and
the hardcoded World Cup table only. No yfinance or network calls.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_cup_effect import data, strategy as st  # noqa: E402

_CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache", "sp500_daily.parquet")
)


# ---------------------------------------------------------------------------
# Helper: build a minimal tagged DataFrame (synthetic)
# ---------------------------------------------------------------------------
def _make_tagged(signal_bps: float = 0.0, seed: int = 235, n_days: int = 3000) -> pd.DataFrame:
    """Build a synthetic tagged DataFrame for strategy tests."""
    df, _ = data.synthetic_daily(n_days=n_days, signal_bps=signal_bps, seed=seed)
    df = df.rename(columns={"return": "sp500_return"})
    df = df.set_index("date")

    # Add ctrl_window: select same number of days, offset by 100 trading days
    rng = np.random.default_rng(seed + 99)
    n_wc = int(df["in_wc"].sum())
    non_wc_idx = np.where(~df["in_wc"].values)[0]
    if len(non_wc_idx) >= n_wc:
        ctrl_idx = rng.choice(non_wc_idx, size=n_wc, replace=False)
    else:
        ctrl_idx = non_wc_idx
    ctrl_mask = np.zeros(len(df), dtype=bool)
    ctrl_mask[ctrl_idx] = True
    df["ctrl_window"] = ctrl_mask

    # Assign edition numbers: group consecutive in_wc days
    df["edition"] = float("nan")
    in_wc_positions = np.where(df["in_wc"].values)[0]
    if len(in_wc_positions) > 0:
        # Assign first in_wc block as edition 1
        df.iloc[in_wc_positions[:min(30, len(in_wc_positions))],
                df.columns.get_loc("edition")] = 1.0

    return df


# ---------------------------------------------------------------------------
# window_returns
# ---------------------------------------------------------------------------
def test_window_returns_shape():
    df = _make_tagged()
    wr = st.window_returns(df)
    # Should have at most as many rows as unique editions
    n_editions = df["edition"].dropna().nunique()
    assert len(wr) == n_editions


def test_window_returns_columns():
    df = _make_tagged()
    wr = st.window_returns(df)
    assert "edition" in wr.columns
    assert "n_wc_days" in wr.columns
    assert "mean_wc_return_bps" in wr.columns


def test_window_returns_positive_days():
    df = _make_tagged()
    wr = st.window_returns(df)
    assert (wr["n_wc_days"] > 0).all()


# ---------------------------------------------------------------------------
# main_stats: output structure and invariants
# ---------------------------------------------------------------------------
def test_main_stats_keys():
    df = _make_tagged()
    result = st.main_stats(df, n_permutations=100, seed=99)
    required_keys = {
        "n_wc_days", "n_non_wc_days", "n_ctrl_days", "n_editions",
        "mean_wc_bps", "mean_non_wc_bps", "mean_ctrl_bps", "mean_all_bps",
        "diff_vs_non_wc_bps", "diff_vs_ctrl_bps",
        "t_vs_all", "p_vs_all", "t_vs_ctrl", "p_vs_ctrl",
        "t_edition_zero", "p_edition_zero",
        "perm_p", "obs_diff_bps",
    }
    assert required_keys.issubset(set(result.keys()))


def test_main_stats_n_adds_up():
    df = _make_tagged()
    result = st.main_stats(df, n_permutations=50, seed=1)
    # WC days + non-WC days should roughly match total
    total = result["n_wc_days"] + result["n_non_wc_days"]
    assert total == int(df["sp500_return"].notna().sum())


def test_main_stats_p_values_in_range():
    df = _make_tagged()
    result = st.main_stats(df, n_permutations=100, seed=1)
    assert 0.0 <= result["p_vs_all"] <= 1.0
    assert 0.0 <= result["p_vs_ctrl"] <= 1.0
    assert 0.0 <= result["perm_p"] <= 1.0


def test_perm_p_deterministic():
    df = _make_tagged()
    r1 = st.main_stats(df, n_permutations=200, seed=42)
    r2 = st.main_stats(df, n_permutations=200, seed=42)
    assert r1["perm_p"] == r2["perm_p"]


# ---------------------------------------------------------------------------
# Null vs planted signal: the spine of the study
# ---------------------------------------------------------------------------
def test_null_has_no_detectable_signal():
    """On a large null synthetic tape, the WC effect should be near zero."""
    df = _make_tagged(signal_bps=0.0, seed=42, n_days=10_000)
    result = st.main_stats(df, n_permutations=200, seed=42)
    # With no signal, the mean WC return should be close to the overall mean
    assert abs(result["diff_vs_non_wc_bps"]) < 20.0  # within 20 bps/day (noise)


def test_planted_signal_detectable():
    """With a very large planted signal (1000 bps/day), the engine finds it."""
    df = _make_tagged(signal_bps=1000.0, seed=235, n_days=5_000)
    result = st.main_stats(df, n_permutations=200, seed=235)
    # WC mean should be much higher than non-WC mean
    assert result["mean_wc_bps"] > result["mean_non_wc_bps"] + 500.0


def test_planted_signal_t_stat_large():
    """A large planted signal should produce a |t| >> 2."""
    df = _make_tagged(signal_bps=1000.0, seed=235, n_days=5_000)
    result = st.main_stats(df, n_permutations=100, seed=235)
    assert abs(result["t_vs_all"]) > 2.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _make_tagged()
    s = st.summarize(df, n_permutations=50, seed=1)
    required = {
        "n_editions", "n_wc_days", "n_non_wc_days",
        "mean_wc_bps", "mean_non_wc_bps",
        "t_vs_all", "p_vs_all", "perm_p",
        "window_returns",
    }
    assert required.issubset(set(s.keys()))


@pytest.mark.skipif(
    not os.path.exists(_CACHE_PATH),
    reason="real-tape cache absent offline/CI"
)
def test_summarize_real_data():
    """If the cache is present, run a sanity check on real S&P 500 data."""
    price_df = data.fetch_sp500_daily(cache_dir=os.path.dirname(_CACHE_PATH))
    tagged = data.tag_wc_windows(price_df)
    s = st.summarize(tagged, n_permutations=1_000, seed=235)
    assert s["n_editions"] == 19
    assert s["n_wc_days"] > 0
    # All 19 editions should have data
    assert len(s["window_returns"]) == 19

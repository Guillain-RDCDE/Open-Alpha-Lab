"""Tests for the strategy layer of Study 277 (Trading-Cards).

All tests are offline and deterministic -- they use the synthetic generator and
the hardcoded curated card index only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_cards import data, strategy as st  # noqa: E402


def _make_merged(signal_bps: float = 0.0, seed: int = 277, n_years: int = 36) -> pd.DataFrame:
    df, _ = data.synthetic_index(n_years=n_years, signal_bps=signal_bps, seed=seed)
    df["card_up"] = df["card_return"] > 0
    df["gspc_up"] = df["gspc_return"] > 0
    return df


# ---------------------------------------------------------------------------
# perf_stats
# ---------------------------------------------------------------------------
def test_perf_stats_keys():
    r = st.perf_stats(np.array([0.1, -0.05, 0.2, 0.0]))
    assert {"n", "mean", "vol", "cagr", "sharpe", "max_drawdown"}.issubset(r.keys())


def test_perf_stats_n():
    r = st.perf_stats(np.array([0.1, 0.2, -0.1]))
    assert r["n"] == 3


def test_perf_stats_drawdown_negative():
    r = st.perf_stats(np.array([0.1, -0.5, 0.1]))
    assert r["max_drawdown"] <= 0.0


def test_perf_stats_positive_returns_no_drawdown():
    r = st.perf_stats(np.array([0.05, 0.05, 0.05]))
    assert r["max_drawdown"] == 0.0


# ---------------------------------------------------------------------------
# hac_alpha
# ---------------------------------------------------------------------------
def test_hac_alpha_keys():
    df = _make_merged()
    r = st.hac_alpha(df["card_return"].values, df["gspc_return"].values, lags=2)
    required = {"n", "alpha", "alpha_ann_pct", "beta", "t_alpha", "t_beta",
                "p_alpha", "p_beta", "r2", "corr", "lags"}
    assert required.issubset(r.keys())


def test_hac_alpha_recovers_beta():
    """On a large synthetic null sample, the regression recovers the planted beta."""
    df = _make_merged(signal_bps=0.0, n_years=3000, seed=277)
    r = st.hac_alpha(df["card_return"].values, df["gspc_return"].values, lags=2)
    assert abs(r["beta"] - 0.6) < 0.1  # planted beta = 0.6


def test_hac_alpha_null_alpha_insignificant():
    """No planted alpha -> alpha t-stat should be small in expectation."""
    df = _make_merged(signal_bps=0.0, n_years=2000, seed=277)
    r = st.hac_alpha(df["card_return"].values, df["gspc_return"].values, lags=2)
    assert abs(r["t_alpha"]) < 2.0


def test_hac_alpha_detects_planted_signal():
    """A large planted alpha should clear the |t| >= 2 bar with enough data."""
    df = _make_merged(signal_bps=2_000.0, n_years=2000, seed=277)
    r = st.hac_alpha(df["card_return"].values, df["gspc_return"].values, lags=2)
    assert r["alpha"] > 0.10
    assert abs(r["t_alpha"]) >= 2.0


def test_hac_alpha_deterministic():
    df = _make_merged()
    r1 = st.hac_alpha(df["card_return"].values, df["gspc_return"].values, lags=2)
    r2 = st.hac_alpha(df["card_return"].values, df["gspc_return"].values, lags=2)
    assert r1["t_alpha"] == r2["t_alpha"]


# ---------------------------------------------------------------------------
# apply_frictions
# ---------------------------------------------------------------------------
def test_apply_frictions_reduces_return():
    raw = np.array([0.10, 0.10, 0.10])
    net = st.apply_frictions(raw, one_way_cost=0.15, holding_years=5.0)
    assert (net < raw).all()
    # 2*0.15 / 5 = 0.06 annual drag
    assert np.allclose(raw - net, 0.06)


def test_apply_frictions_zero_cost_is_identity():
    raw = np.array([0.1, -0.2, 0.05])
    net = st.apply_frictions(raw, one_way_cost=0.0, holding_years=5.0)
    assert np.allclose(net, raw)


# ---------------------------------------------------------------------------
# block_bootstrap_alpha
# ---------------------------------------------------------------------------
def test_bootstrap_keys():
    df = _make_merged()
    b = st.block_bootstrap_alpha(df["card_return"].values, df["gspc_return"].values,
                                 n_boot=500, block=3, seed=277)
    assert {"alpha_p2.5", "alpha_p50", "alpha_p97.5", "frac_alpha_pos",
            "sharpe_gap_p50", "n_boot"}.issubset(b.keys())


def test_bootstrap_ci_ordered():
    df = _make_merged()
    b = st.block_bootstrap_alpha(df["card_return"].values, df["gspc_return"].values,
                                 n_boot=500, block=3, seed=277)
    assert b["alpha_p2.5"] <= b["alpha_p50"] <= b["alpha_p97.5"]


def test_bootstrap_deterministic():
    df = _make_merged()
    b1 = st.block_bootstrap_alpha(df["card_return"].values, df["gspc_return"].values,
                                  n_boot=300, block=3, seed=42)
    b2 = st.block_bootstrap_alpha(df["card_return"].values, df["gspc_return"].values,
                                  n_boot=300, block=3, seed=42)
    assert b1["alpha_p50"] == b2["alpha_p50"]


def test_bootstrap_frac_pos_is_probability():
    df = _make_merged()
    b = st.block_bootstrap_alpha(df["card_return"].values, df["gspc_return"].values,
                                 n_boot=300, block=3, seed=1)
    assert 0.0 <= b["frac_alpha_pos"] <= 1.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _make_merged(n_years=36)
    s = st.summarize(df, n_boot=300, seed=1)
    required = {
        "n", "card_cagr_pct", "card_vol_pct", "card_sharpe",
        "card_net_cagr_pct", "gspc_cagr_pct", "gspc_sharpe",
        "alpha_ann_pct", "beta", "t_alpha", "corr",
        "boot_alpha_p50", "boot_frac_alpha_pos", "card_cagr_ex_boom_pct",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input():
    df = _make_merged(n_years=30)
    s = st.summarize(df, n_boot=200, seed=1)
    assert s["n"] == 30


def test_summarize_net_below_gross():
    """Net card CAGR must be below gross card CAGR (frictions always cost)."""
    df = _make_merged(n_years=36)
    s = st.summarize(df, n_boot=200, seed=1)
    assert s["card_net_cagr_pct"] < s["card_cagr_pct"]

"""Tests for the strategy layer of Study 273 (Lego-Returns). Offline & deterministic."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lego_returns import data, strategy as st  # noqa: E402


def _make_paired(premium_bps: float = 0.0, seed: int = 273, n_years: int = 38) -> pd.DataFrame:
    df, _ = data.synthetic_annual(n_years=n_years, premium_bps=premium_bps, seed=seed)
    return df


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def test_net_return_subtracts_drag():
    lr = np.array([0.10, 0.10, 0.10])
    net = st.net_lego_return(lr, one_way_cost=0.12, holding_years=1.0)
    assert np.allclose(net, lr - 0.12)


def test_net_return_amortises_over_hold():
    lr = np.array([0.10, 0.10])
    net5 = st.net_lego_return(lr, one_way_cost=0.10, holding_years=5.0)
    assert np.allclose(net5, lr - 0.02)


def test_net_return_zero_cost_identity():
    lr = np.array([0.05, -0.03, 0.20])
    assert np.allclose(st.net_lego_return(lr, one_way_cost=0.0), lr)


# ---------------------------------------------------------------------------
# HAC t-stat
# ---------------------------------------------------------------------------
def test_hac_zero_mean_small_t():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 500)
    mu, t = st.hac_tstat(x)
    assert abs(t) < 3.0


def test_hac_strong_mean_large_t():
    rng = np.random.default_rng(2)
    x = rng.normal(1.0, 0.5, 500)
    mu, t = st.hac_tstat(x)
    assert t > 5.0
    assert mu > 0.5


def test_hac_handles_tiny_sample():
    mu, t = st.hac_tstat(np.array([0.1, 0.2]))
    assert np.isnan(t)


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------
def test_beta_keys():
    df = _make_paired()
    b = st.beta_to_market(df["lego_return"].values, df["market_return"].values)
    assert {"beta", "alpha", "corr"}.issubset(set(b.keys()))


def test_beta_near_one_when_identical():
    rng = np.random.default_rng(3)
    m = rng.normal(0.08, 0.16, 200)
    b = st.beta_to_market(m, m)
    assert abs(b["beta"] - 1.0) < 1e-6
    assert abs(b["corr"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------
def test_compare_keys():
    df = _make_paired()
    s = st.compare(df, one_way_cost=0.12, holding_years=5.0)
    required = {
        "n", "lego_cagr", "lego_net_cagr", "market_cagr",
        "mean_excess_gross", "mean_excess_net", "hac_t_gross", "hac_t_net",
        "beta", "alpha", "corr", "lego_win_years", "lego_win_rate", "sign_test_p",
    }
    assert required.issubset(set(s.keys()))


def test_compare_n_matches_input():
    df = _make_paired(n_years=25)
    s = st.compare(df)
    assert s["n"] == 25


def test_net_cagr_below_gross_when_fee():
    df = _make_paired()
    s = st.compare(df, one_way_cost=0.12, holding_years=5.0)
    assert s["lego_net_cagr"] < s["lego_cagr"]


def test_win_rate_is_fraction():
    df = _make_paired()
    s = st.compare(df)
    assert 0.0 <= s["lego_win_rate"] <= 1.0
    assert 0.0 <= s["sign_test_p"] <= 1.0


# ---------------------------------------------------------------------------
# Null vs planted premium: the spine of the study
# ---------------------------------------------------------------------------
def test_null_no_significant_excess():
    """On the null tape, the HAC t on excess should be modest."""
    df = _make_paired(premium_bps=0.0, seed=42, n_years=300)
    s = st.compare(df, one_way_cost=0.0, holding_years=1.0)
    assert abs(s["hac_t_gross"]) < 3.0


def test_planted_premium_detected():
    """A large planted premium should clear the t >= 2 bar (gross)."""
    df = _make_paired(premium_bps=2_000.0, seed=273, n_years=200)
    s = st.compare(df, one_way_cost=0.0, holding_years=1.0)
    assert s["mean_excess_gross"] > 0
    assert s["hac_t_gross"] >= 2.0


def test_compare_deterministic():
    df = _make_paired()
    a = st.compare(df)
    b = st.compare(df)
    assert a["hac_t_gross"] == b["hac_t_gross"]
    assert a["sign_test_p"] == b["sign_test_p"]

"""Tests for the strategy layer of Study 300 (Sports-Sentiment).

All tests are offline and deterministic -- the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sports_sentiment import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# HAC mean t-stat
# ---------------------------------------------------------------------------
def test_hac_zero_mean_small_t():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 2000)
    r = st.hac_mean_tstat(x, lags=1)
    assert abs(r["t"]) < 3.0  # not significant for true-zero mean


def test_hac_strong_mean_large_t():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1.0, 2000)
    r = st.hac_mean_tstat(x, lags=1)
    assert r["t"] > 5.0


def test_hac_lag0_matches_plain_se():
    rng = np.random.default_rng(2)
    x = rng.normal(0.1, 1.0, 500)
    r = st.hac_mean_tstat(x, lags=0)
    plain_se = x.std(ddof=0) / np.sqrt(len(x))
    assert abs(r["se"] - plain_se) < 1e-9


def test_hac_handles_tiny_n():
    r = st.hac_mean_tstat(np.array([0.01]), lags=1)
    assert np.isnan(r["t"])


# ---------------------------------------------------------------------------
# loss_effect_stats
# ---------------------------------------------------------------------------
def test_loss_effect_keys():
    df, _ = data.synthetic_panel(n_events=60, loss_bps=0.0, seed=1)
    r = st.loss_effect_stats(df, n_boot=200, seed=1)
    required = {"n", "mean_bps", "hac_t", "plain_t", "boot_p_ge0",
                "frac_neg", "n_down", "n_up"}
    assert required.issubset(set(r.keys()))


def test_loss_effect_counts_add_up():
    df, _ = data.synthetic_panel(n_events=80, loss_bps=0.0, seed=3)
    r = st.loss_effect_stats(df, n_boot=200, seed=3)
    assert r["n_down"] + r["n_up"] == r["n"]


def test_loss_effect_boot_p_is_probability():
    df, _ = data.synthetic_panel(n_events=60, loss_bps=0.0, seed=4)
    r = st.loss_effect_stats(df, n_boot=500, seed=4)
    assert 0.0 <= r["boot_p_ge0"] <= 1.0


def test_loss_effect_deterministic():
    df, _ = data.synthetic_panel(n_events=60, loss_bps=0.0, seed=5)
    r1 = st.loss_effect_stats(df, n_boot=300, seed=5)
    r2 = st.loss_effect_stats(df, n_boot=300, seed=5)
    assert r1["boot_p_ge0"] == r2["boot_p_ge0"]
    assert r1["hac_t"] == r2["hac_t"]


# ---------------------------------------------------------------------------
# Null vs planted: the spine of the study
# ---------------------------------------------------------------------------
def test_null_panel_not_significant():
    df, _ = data.synthetic_panel(n_events=400, loss_bps=0.0, seed=300)
    r = st.loss_effect_stats(df, n_boot=1000, seed=300)
    assert abs(r["hac_t"]) < 2.0


def test_planted_loss_is_detected():
    """A planted -49 bps effect must show a strongly negative, significant t."""
    df, _ = data.synthetic_panel(n_events=400, loss_bps=49.0, seed=300)
    r = st.loss_effect_stats(df, n_boot=1000, seed=300)
    assert r["mean_bps"] < -20.0
    assert r["hac_t"] < -2.0
    assert r["boot_p_ge0"] < 0.05


# ---------------------------------------------------------------------------
# subgroup_summary
# ---------------------------------------------------------------------------
def test_subgroup_summary_all_row():
    df, _ = data.synthetic_panel(n_events=60, loss_bps=0.0, seed=6)
    df["shootout"] = (df["event_id"] % 2 == 0)
    df["tournament"] = np.where(df["event_id"] % 3 == 0, "WC", "EURO")
    sg = st.subgroup_summary(df)
    assert "All eliminations" in set(sg["group"])
    assert (sg["n"] > 0).all()


# ---------------------------------------------------------------------------
# tradable_pnl
# ---------------------------------------------------------------------------
def test_tradable_pnl_short_sign():
    """Short PnL is the negative of the underlying return."""
    df = pd.DataFrame({"event_id": [1, 2], "next_day_ret": [0.01, -0.02]})
    tr = st.tradable_pnl(df, cost_bps_oneway=0.0)
    # gross mean = -mean(ret) = -(-0.005) = +50 bps
    assert abs(tr["gross_mean_bps"] - 50.0) < 1e-6


def test_tradable_costs_reduce_pnl():
    df, _ = data.synthetic_panel(n_events=60, loss_bps=0.0, seed=7)
    free = st.tradable_pnl(df, cost_bps_oneway=0.0)
    costly = st.tradable_pnl(df, cost_bps_oneway=10.0)
    assert costly["net_mean_bps"] < free["net_mean_bps"]


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df, _ = data.synthetic_panel(n_events=60, loss_bps=0.0, seed=8)
    s = st.summarize(df, n_boot=200, seed=8)
    required = {"n", "mean_bps", "hac_t", "boot_p_ge0",
                "gross_mean_bps", "net_mean_bps", "net_hit_rate"}
    assert required.issubset(set(s.keys()))

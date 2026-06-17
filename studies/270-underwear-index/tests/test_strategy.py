"""Tests for the strategy layer of Study 270 (Underwear-Index).

All tests are offline and deterministic — they use the synthetic generator and
the merged_like fixture only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from underwear_index import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def test_dip_signal_values(merged_like):
    sig = st.dip_signal(merged_like)
    assert set(sig.unique()) <= {0, 1}
    assert (sig[merged_like["yoy"] < 0] == 1).all()


# ---------------------------------------------------------------------------
# recession_stats: structure and invariants
# ---------------------------------------------------------------------------
def test_recession_stats_keys(merged_like):
    r = st.recession_stats(merged_like, horizon="next", n_permutations=200, seed=99)
    required = {
        "n", "a_dip_rec", "b_dip_norec", "c_nodip_rec", "d_nodip_norec",
        "n_dip", "n_nodip", "rec_rate_dip", "rec_rate_nodip",
        "uncond_rec_rate", "fisher_p", "perm_p",
    }
    assert required.issubset(set(r.keys()))


def test_recession_stats_cells_add_up(merged_like):
    r = st.recession_stats(merged_like, horizon="next", n_permutations=100, seed=1)
    assert r["a_dip_rec"] + r["b_dip_norec"] + r["c_nodip_rec"] + r["d_nodip_norec"] == r["n"]
    assert r["n_dip"] + r["n_nodip"] == r["n"]


def test_fisher_and_perm_are_probabilities(merged_like):
    r = st.recession_stats(merged_like, horizon="next", n_permutations=300, seed=7)
    assert 0.0 <= r["fisher_p"] <= 1.0
    assert 0.0 <= r["perm_p"] <= 1.0


def test_permutation_deterministic(merged_like):
    r1 = st.recession_stats(merged_like, horizon="next", n_permutations=200, seed=42)
    r2 = st.recession_stats(merged_like, horizon="next", n_permutations=200, seed=42)
    assert r1["perm_p"] == r2["perm_p"]


# ---------------------------------------------------------------------------
# Null vs planted signal: the spine of the study
# ---------------------------------------------------------------------------
def _synth_merged(signal_bps: float, seed: int = 270, n_years: int = 400) -> pd.DataFrame:
    df, _ = data.synthetic_annual(n_years=n_years, signal_bps=signal_bps, seed=seed)
    df = df.sort_values("year").reset_index(drop=True)
    df["next_recession"] = df["recession"].shift(-1).fillna(False).astype(bool)
    rng = np.random.default_rng(seed)
    df["sp500_return"] = rng.normal(0.08, 0.17, len(df))
    df["next_sp500_return"] = df["sp500_return"].shift(-1)
    return df


def test_null_has_no_signal():
    """On the null tape (same-year horizon), dip and no-dip recession rates are close."""
    df = _synth_merged(signal_bps=0.0, seed=42)
    r = st.recession_stats(df, horizon="same", n_permutations=500, seed=42)
    assert abs(r["rec_rate_dip"] - r["rec_rate_nodip"]) < 0.20
    assert r["fisher_p"] > 0.05


def test_planted_signal_detectable():
    """A strong planted drag makes dips concentrate in recession years and fires Fisher."""
    df = _synth_merged(signal_bps=6_000.0, seed=270)
    r = st.recession_stats(df, horizon="same", n_permutations=500, seed=270)
    assert r["rec_rate_dip"] > r["rec_rate_nodip"]
    assert r["fisher_p"] < 0.05


# ---------------------------------------------------------------------------
# timing_backtest
# ---------------------------------------------------------------------------
def test_timing_backtest_keys(merged_like):
    bt = st.timing_backtest(merged_like, cost_bps=10.0)
    required = {
        "n_years", "cost_bps", "turnover", "bh_ann",
        "strat_gross_ann", "strat_net_ann", "spread_mean", "nw_t",
    }
    assert required.issubset(set(bt.keys()))


def test_timing_costs_reduce_net(merged_like):
    bt0 = st.timing_backtest(merged_like, cost_bps=0.0)
    bt1 = st.timing_backtest(merged_like, cost_bps=100.0)
    assert bt1["strat_net_ann"] <= bt0["strat_net_ann"] + 1e-9


def test_timing_turnover_in_range(merged_like):
    bt = st.timing_backtest(merged_like, cost_bps=10.0)
    assert 0.0 <= bt["turnover"] <= 2.0


def test_newey_west_zero_mean_series():
    x = np.array([1.0, -1.0] * 20)
    t = st._newey_west_t(x, lags=1)
    assert abs(t) < 1.0  # mean is ~0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(merged_like):
    s = st.summarize(merged_like, n_permutations=100, seed=1)
    required = {
        "n", "n_dips", "uncond_rec_pct", "rec_rate_dip_pct",
        "next_fisher_p", "next_perm_p", "same_fisher_p",
        "bh_ann_pct", "strat_net_pct", "turnover", "nw_t",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input(merged_like):
    s = st.summarize(merged_like, n_permutations=50, seed=1)
    assert s["n"] == len(merged_like)

"""Tests for the strategy layer of Study 281 (El-Nino).

All tests are offline and deterministic -- synthetic generator and hardcoded ONI
table only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from el_nino import data, strategy as st  # noqa: E402


def _make_merged(premium_bps: float = 0.0, seed: int = 281, n_years: int = 75) -> pd.DataFrame:
    """Build a synthetic merged frame that looks like fetch_real() output."""
    df, _ = data.synthetic_panel(n_years=n_years, premium_bps=premium_bps, seed=seed)
    df = df.rename(columns={"ret": "asset_return"})
    df["mkt_up"] = df["asset_return"] > 0
    return df


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def test_signal_warm_cold_values():
    df = _make_merged()
    sig = st.signal_warm_cold(df)
    assert set(sig.unique()) <= {1, -1, 0}
    assert (sig[df["phase"] == "El Nino"] == 1).all()
    assert (sig[df["phase"] == "La Nina"] == -1).all()
    assert (sig[df["phase"] == "Neutral"] == 0).all()


def test_signal_oni_sign():
    df = _make_merged()
    sig = st.signal_oni_sign(df)
    assert set(sig.unique()) <= {1, -1, 0}
    # positive ONI -> +1
    assert (sig[df["oni_ndj"] > 0] == 1).all()
    assert (sig[df["oni_ndj"] < 0] == -1).all()


# ---------------------------------------------------------------------------
# HAC t-stat
# ---------------------------------------------------------------------------
def test_hac_tstat_zero_mean_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    assert abs(st.hac_tstat(x, lags=1)) < 3.0


def test_hac_tstat_strong_mean_large():
    x = np.full(100, 0.5) + np.random.default_rng(1).normal(0, 0.05, 100)
    assert st.hac_tstat(x, lags=1) > 5.0


def test_hac_tstat_nan_on_tiny():
    assert np.isnan(st.hac_tstat(np.array([1.0]), lags=1))


# ---------------------------------------------------------------------------
# phase_stats: structure and invariants
# ---------------------------------------------------------------------------
def test_phase_stats_keys():
    df = _make_merged()
    r = st.phase_stats(df, n_permutations=200, seed=99)
    required = {
        "n_total", "n_elnino", "n_lanina", "n_neutral",
        "mean_elnino", "mean_lanina", "mean_neutral", "mean_all",
        "gap", "uncond_up_rate", "welch_t", "welch_p", "hac_t",
        "anova_f", "anova_p", "perm_p",
    }
    assert required.issubset(set(r.keys()))


def test_phase_stats_n_adds_up():
    df = _make_merged()
    r = st.phase_stats(df, n_permutations=100, seed=1)
    assert r["n_elnino"] + r["n_lanina"] + r["n_neutral"] == r["n_total"]


def test_phase_stats_gap_definition():
    df = _make_merged()
    r = st.phase_stats(df, n_permutations=100, seed=1)
    # each field is rounded to 4 dp independently, so allow rounding slack
    assert abs(r["gap"] - (r["mean_elnino"] - r["mean_lanina"])) < 2e-4


def test_perm_p_is_probability():
    df = _make_merged()
    r = st.phase_stats(df, n_permutations=500, seed=7)
    assert 0.0 <= r["perm_p"] <= 1.0


def test_phase_stats_deterministic():
    df = _make_merged()
    r1 = st.phase_stats(df, n_permutations=300, seed=42)
    r2 = st.phase_stats(df, n_permutations=300, seed=42)
    assert r1["perm_p"] == r2["perm_p"]


# ---------------------------------------------------------------------------
# Null vs planted signal: the spine of the study
# ---------------------------------------------------------------------------
def test_null_gap_not_significant():
    """On a large null tape the gap is tiny and perm p is large."""
    df = _make_merged(premium_bps=0.0, seed=281, n_years=600)
    r = st.phase_stats(df, n_permutations=500, seed=281)
    assert abs(r["gap"]) < 0.03
    assert r["perm_p"] > 0.10


def test_planted_signal_detectable():
    """A large planted premium is found: El Nino > La Nina, perm p tiny, |t| large."""
    df = _make_merged(premium_bps=1_000.0, seed=281, n_years=600)
    r = st.phase_stats(df, n_permutations=500, seed=281)
    assert r["mean_elnino"] > r["mean_lanina"]
    assert r["perm_p"] < 0.05
    assert abs(r["welch_t"]) > 2.0


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def test_backtest_keys():
    df = _make_merged()
    bt = st.backtest(df)
    required = {"n_years", "gross_ann", "net_ann", "buy_hold_ann",
                "turnover_per_yr", "gross_minus_bah", "net_minus_bah"}
    assert required.issubset(set(bt.keys()))


def test_backtest_net_le_gross():
    """Costs and borrow can only reduce the strategy return."""
    df = _make_merged(premium_bps=500.0)
    bt = st.backtest(df, cost_bps_oneway=20.0, short_borrow_bps=100.0)
    assert bt["net_ann"] <= bt["gross_ann"] + 1e-9


# ---------------------------------------------------------------------------
# multiple_comparisons_summary
# ---------------------------------------------------------------------------
def test_mc_summary_shape():
    a = _make_merged(seed=1)
    b = _make_merged(seed=2)
    mc = st.multiple_comparisons_summary({"a": a, "b": b}, n_permutations=100, seed=1)
    assert len(mc) == 2
    assert "bonferroni_perm_p" in mc.columns


def test_mc_bonferroni_ge_uncorrected():
    a = _make_merged(seed=1)
    b = _make_merged(seed=2)
    mc = st.multiple_comparisons_summary({"a": a, "b": b}, n_permutations=100, seed=1)
    for _, row in mc.iterrows():
        assert row["bonferroni_perm_p"] >= row["perm_p"] - 1e-12


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _make_merged()
    s = st.summarize(df, n_permutations=100, seed=1)
    required = {
        "n", "uncond_up_rate_pct", "mean_elnino_pct", "mean_lanina_pct",
        "gap_pct", "welch_t", "hac_t", "anova_p", "perm_p",
        "gross_ann_pct", "net_ann_pct", "buy_hold_ann_pct",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input():
    df = _make_merged(n_years=50)
    s = st.summarize(df, n_permutations=50, seed=1)
    assert s["n"] == 50

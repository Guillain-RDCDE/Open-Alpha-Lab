"""Tests for the strategy layer of Study 289 (Diwali-Muhurat).

All tests are offline and deterministic — they use the synthetic generator and
the hardcoded Diwali table only.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diwali_muhurat import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# event_windows
# ---------------------------------------------------------------------------
def test_event_windows_shape(synthetic_null, diwali_tbl):
    px, _ = synthetic_null
    ev = st.event_windows(px, diwali_tbl, hold_days=5, lag=1)
    # All 23 events should form a window on the synthetic tape (2003–2025 daily).
    assert len(ev) == 23
    assert {"entry_date", "exit_date", "window_ret", "n_sessions"}.issubset(ev.columns)


def test_event_windows_lag_is_after_event(synthetic_null, diwali_tbl):
    """Entry date must be strictly on/after each Diwali (no look-ahead)."""
    px, _ = synthetic_null
    ev = st.event_windows(px, diwali_tbl, hold_days=5, lag=1)
    for yr, row in ev.iterrows():
        d = diwali_tbl.loc[diwali_tbl["year"] == yr, "diwali"].iloc[0]
        assert row["entry_date"] >= d


def test_event_windows_n_sessions(synthetic_null, diwali_tbl):
    px, _ = synthetic_null
    ev = st.event_windows(px, diwali_tbl, hold_days=5, lag=1)
    assert (ev["n_sessions"] <= 5).all()
    assert (ev["n_sessions"] >= 1).all()


# ---------------------------------------------------------------------------
# event_stats: structure and invariants
# ---------------------------------------------------------------------------
def test_event_stats_keys(synthetic_null, diwali_tbl):
    px, _ = synthetic_null
    r = st.event_stats(px, diwali_tbl, hold_days=1, n_permutations=200, seed=99)
    required = {
        "n_events", "hold_days", "lag", "mean_event_ret", "baseline_window_ret",
        "excess_ret", "hit_rate", "t_plain", "t_nw", "perm_p",
    }
    assert required.issubset(set(r.keys()))


def test_event_stats_hit_rate_in_range(synthetic_null, diwali_tbl):
    px, _ = synthetic_null
    r = st.event_stats(px, diwali_tbl, hold_days=1, n_permutations=200, seed=1)
    assert 0.0 <= r["hit_rate"] <= 1.0


def test_event_stats_perm_p_is_probability(synthetic_null, diwali_tbl):
    px, _ = synthetic_null
    r = st.event_stats(px, diwali_tbl, hold_days=1, n_permutations=500, seed=7)
    assert 0.0 <= r["perm_p"] <= 1.0


def test_permutation_deterministic(synthetic_null, diwali_tbl):
    """Same seed → same permutation p-value."""
    px, _ = synthetic_null
    r1 = st.event_stats(px, diwali_tbl, hold_days=1, n_permutations=300, seed=42)
    r2 = st.event_stats(px, diwali_tbl, hold_days=1, n_permutations=300, seed=42)
    assert r1["perm_p"] == r2["perm_p"]


# ---------------------------------------------------------------------------
# Null vs planted signal: the spine of the study
# ---------------------------------------------------------------------------
def test_null_is_not_significant(synthetic_null, diwali_tbl):
    """On the null tape, the Diwali window does not beat baseline with |t|>=2."""
    px, _ = synthetic_null
    r = st.event_stats(px, diwali_tbl, hold_days=1, n_permutations=500, seed=289)
    assert abs(r["t_plain"]) < 2.0
    assert r["perm_p"] > 0.05  # one-sided 'greater': no positive edge


def test_planted_signal_detectable(synthetic_signal, diwali_tbl):
    """With a large planted Diwali premium, the engine finds it."""
    px, _ = synthetic_signal
    r = st.event_stats(px, diwali_tbl, hold_days=1, n_permutations=500, seed=289)
    assert r["excess_ret"] > 0.0
    assert r["t_plain"] > 2.0
    assert r["perm_p"] < 0.05


# ---------------------------------------------------------------------------
# net_edge
# ---------------------------------------------------------------------------
def test_net_edge_costs_reduce_return(synthetic_null, diwali_tbl):
    px, _ = synthetic_null
    ne = st.net_edge(px, diwali_tbl, hold_days=1, cost_bps_one_way=5.0)
    assert ne["net_mean"] < ne["gross_mean"]
    assert ne["round_trip_cost"] == pytest.approx(2 * 5.0 * 1e-4)


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(synthetic_null, diwali_tbl):
    px, _ = synthetic_null
    s = st.summarize(px, diwali_tbl, hold_days=1, n_permutations=200, seed=1)
    required = {
        "n_events", "hold_days", "mean_event_pct", "baseline_pct", "excess_pct",
        "hit_rate_pct", "t_plain", "t_nw", "perm_p", "gross_pct", "net_pct",
        "net_excess_pct",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_events(synthetic_null, diwali_tbl):
    px, _ = synthetic_null
    s = st.summarize(px, diwali_tbl, hold_days=1, n_permutations=100, seed=1)
    assert s["n_events"] == 23

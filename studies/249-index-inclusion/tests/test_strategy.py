"""Tests for strategy.py — HAC inference, positive/negative controls, period
breakdown, and cost-adjusted tradability."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from index_inclusion import data, strategy as st  # noqa: E402

CACHE_PATH = data.DEFAULT_CACHE


# ---------------------------------------------------------------------------
# summarize_window
# ---------------------------------------------------------------------------

def test_summarize_window_keys(null_events):
    events, _ = null_events
    s = st.summarize_window(events, col="ret_pop")
    for key in ("n", "mean_pct", "median_pct", "std_pct", "win_rate", "tstat"):
        assert key in s


def test_summarize_window_win_rate_range(null_events):
    events, _ = null_events
    s = st.summarize_window(events, col="ret_pop")
    assert 0.0 <= s["win_rate"] <= 1.0


def test_summarize_window_empty():
    empty = pd.DataFrame({"ret_pop": [float("nan")] * 3})
    s = st.summarize_window(empty, col="ret_pop")
    assert np.isnan(s["tstat"])


def test_summarize_window_count_matches(null_events):
    events, _ = null_events
    s = st.summarize_window(events, col="ret_pop")
    n_valid = events["ret_pop"].dropna().shape[0]
    assert s["n"] == n_valid


# ---------------------------------------------------------------------------
# compare_pop_giveback
# ---------------------------------------------------------------------------

def test_compare_pop_giveback_shape(null_events):
    events, _ = null_events
    comp = st.compare_pop_giveback(events)
    assert len(comp) == 3  # pop, giveback_1m, giveback_3m
    assert "mean_pct" in comp.columns
    assert "tstat" in comp.columns


def test_compare_pop_giveback_windows_present(null_events):
    events, _ = null_events
    comp = st.compare_pop_giveback(events)
    windows = comp["window"].values
    assert any("Pop" in w for w in windows)
    assert any("1m" in w for w in windows)
    assert any("3m" in w for w in windows)


# ---------------------------------------------------------------------------
# Positive control — the spine
# ---------------------------------------------------------------------------

def test_planted_pop_shows_higher_mean(null_events, planted_events):
    """With pop_bps=20, mean ret_pop is materially higher than at pop_bps=0."""
    ev_null, _ = null_events
    ev_plant, _ = planted_events
    mean_null = ev_null["ret_pop"].dropna().mean()
    mean_plant = ev_plant["ret_pop"].dropna().mean()
    assert mean_plant > mean_null


def test_planted_giveback_shows_lower_mean(null_events, giveback_events):
    """With giveback_bps=15, mean ret_giveback_1m is lower than at giveback_bps=0."""
    ev_null, _ = null_events
    ev_gb, _ = giveback_events
    mean_null = ev_null["ret_giveback_1m"].dropna().mean()
    mean_gb = ev_gb["ret_giveback_1m"].dropna().mean()
    assert mean_gb < mean_null


def test_pop_vs_random_edge_planted_exceeds_null():
    ev_planted, _ = data.synthetic_events(n_events=60, pop_bps=25.0, seed=1)
    ev_null, _ = data.synthetic_events(n_events=60, pop_bps=0.0, seed=1)
    p_mean, n_mean = st.pop_vs_random_edge(ev_planted, ev_null, col="ret_pop")
    assert p_mean > n_mean


# ---------------------------------------------------------------------------
# pop_by_period
# ---------------------------------------------------------------------------

def test_pop_by_period_returns_dataframe(null_events):
    events, _ = null_events
    # Use announce_date as proxy for period (synthetic data)
    events2 = events.copy()
    events2["effective_date"] = events2["effective_date"].dt.normalize()
    pp = st.pop_by_period(events2)
    assert isinstance(pp, pd.DataFrame)
    if not pp.empty:
        assert "period" in pp.columns
        assert "mean_pop_pct" in pp.columns


# ---------------------------------------------------------------------------
# cost_adjusted_pop
# ---------------------------------------------------------------------------

def test_cost_adjusted_pop_reduces_net():
    gross_net = st.cost_adjusted_pop(0.02, cost_bps=0.0)
    cost_net = st.cost_adjusted_pop(0.02, cost_bps=10.0)
    assert cost_net < gross_net


def test_cost_adjusted_pop_near_zero_edge_goes_negative():
    """A tiny gross pop is wiped out by 10 bps round-trip."""
    net = st.cost_adjusted_pop(0.0002, cost_bps=10.0)
    assert net < 0.0


@pytest.mark.skipif(
    not os.path.exists(CACHE_PATH) or
    not any(f.startswith("prices_249_") for f in os.listdir(CACHE_PATH)
            if os.path.isfile(os.path.join(CACHE_PATH, f))),
    reason="real-tape cache absent offline/CI"
)
def test_real_tape_pop_tstat_below_bar():
    """On the real tape the HAC t-stat for the pop should be below 2 (weak signal)."""
    from index_inclusion import data as d
    events, _ = d.fetch_inclusion_panel(fetch=False)
    s = st.summarize_window(events, col="ret_pop")
    # The signal is Weak — t should be below 2 on this sample
    assert abs(s["tstat"]) < 2.0, (
        f"Real-tape pop t = {s['tstat']:.2f} unexpectedly >= 2.0; verdict may need update"
    )

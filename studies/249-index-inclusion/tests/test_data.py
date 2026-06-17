"""Tests for the data layer — synthetic panel shape, determinism, knob wiring,
cache safety guard, and the INCLUSION_TABLE sanity checks."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from index_inclusion import data  # noqa: E402

CACHE_PATH = data.DEFAULT_CACHE


# ---------------------------------------------------------------------------
# Synthetic event panel
# ---------------------------------------------------------------------------

def test_synthetic_events_shape_and_columns(null_events):
    events, truth = null_events
    assert not events.empty
    expected_cols = {"ticker", "announce_date", "effective_date", "window_days",
                     "ret_pop", "ret_giveback_1m", "ret_giveback_3m"}
    assert expected_cols.issubset(set(events.columns))


def test_synthetic_events_returns_finite(null_events):
    events, _ = null_events
    for col in ("ret_pop", "ret_giveback_1m"):
        vals = events[col].dropna()
        assert len(vals) > 0
        assert np.all(np.isfinite(vals))


def test_synthetic_events_window_days_positive(null_events):
    events, _ = null_events
    assert (events["window_days"] >= 3).all()
    assert (events["window_days"] <= 15).all()


def test_synthetic_events_is_deterministic():
    a, _ = data.synthetic_events(n_events=30, seed=7)
    b, _ = data.synthetic_events(n_events=30, seed=7)
    assert list(a["ret_pop"].dropna().values) == list(b["ret_pop"].dropna().values)
    # Different seed -> different results
    c, _ = data.synthetic_events(n_events=30, seed=8)
    assert a["ret_pop"].dropna().mean() != c["ret_pop"].dropna().mean()


def test_pop_knob_shifts_mean_return():
    """Positive pop_bps should produce higher mean ret_pop."""
    null_ev, _ = data.synthetic_events(n_events=60, pop_bps=0.0, seed=42)
    planted_ev, _ = data.synthetic_events(n_events=60, pop_bps=30.0, seed=42)
    assert planted_ev["ret_pop"].dropna().mean() > null_ev["ret_pop"].dropna().mean()


def test_giveback_knob_shifts_mean_return():
    """Positive giveback_bps should produce lower (more negative) mean ret_giveback_1m."""
    null_ev, _ = data.synthetic_events(n_events=60, giveback_bps=0.0, seed=42)
    gb_ev, _ = data.synthetic_events(n_events=60, giveback_bps=20.0, seed=42)
    assert gb_ev["ret_giveback_1m"].dropna().mean() < null_ev["ret_giveback_1m"].dropna().mean()


def test_fingerprint_stable_and_content_sensitive(null_events):
    events, _ = null_events
    fp1 = data.fingerprint(events)
    fp2 = data.fingerprint(events)
    assert fp1 == fp2
    other, _ = data.synthetic_events(n_events=60, seed=999)
    assert data.fingerprint(events) != data.fingerprint(other)


def test_inclusion_table_has_entries():
    assert len(data.INCLUSION_TABLE) >= 40
    # Each row must have required keys
    for row in data.INCLUSION_TABLE:
        assert "ticker" in row
        assert "announce_date" in row
        assert "effective_date" in row
        # Effective date must be after announce date
        assert row["effective_date"] > row["announce_date"]


def test_inclusion_table_no_duplicates():
    keys = [(r["ticker"], str(r["announce_date"])) for r in data.INCLUSION_TABLE]
    assert len(keys) == len(set(keys)), "Duplicate (ticker, announce_date) in INCLUSION_TABLE"


@pytest.mark.skipif(
    not os.path.exists(CACHE_PATH) or
    not any(f.startswith("prices_249_") for f in os.listdir(CACHE_PATH)
            if os.path.isfile(os.path.join(CACHE_PATH, f))),
    reason="real-tape cache absent offline/CI"
)
def test_fetch_inclusion_panel_cache_only():
    """With cache present, fetch=False returns events with correct columns."""
    events, prices = data.fetch_inclusion_panel(fetch=False)
    assert not events.empty
    assert "ret_pop" in events.columns
    assert "ret_giveback_1m" in events.columns
    assert "ret_giveback_3m" in events.columns
    assert len(events) >= 30

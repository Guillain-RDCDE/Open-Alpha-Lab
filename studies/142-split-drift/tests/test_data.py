"""Tests for the data layer — synthetic panel shape, determinism, knob wiring,
and the cache-only safety guard."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from split_drift import data  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic event panel
# ---------------------------------------------------------------------------

def test_synthetic_events_shape_and_columns(null_events):
    events, truth = null_events
    assert not events.empty
    expected_cols = {"ticker", "event_date", "split_ratio",
                     "ret_1m", "ret_3m", "ret_6m", "ret_12m", "is_split"}
    assert expected_cols.issubset(set(events.columns))
    assert (events["is_split"] == True).all()  # noqa: E712


def test_synthetic_events_returns_are_finite(null_events):
    events, _ = null_events
    for col in ("ret_1m", "ret_3m"):
        vals = events[col].dropna()
        assert len(vals) > 0
        assert np.all(np.isfinite(vals))


def test_synthetic_events_is_deterministic():
    a, _ = data.synthetic_events(n_stocks=5, n_events=20, drift_bps=0.0, seed=7)
    b, _ = data.synthetic_events(n_stocks=5, n_events=20, drift_bps=0.0, seed=7)
    assert list(a["ret_3m"].dropna().values) == list(b["ret_3m"].dropna().values)
    # Different seed should produce different results
    c, _ = data.synthetic_events(n_stocks=5, n_events=20, drift_bps=0.0, seed=8)
    # The means should differ (different random draws); shapes may differ too
    assert a["ret_3m"].dropna().mean() != c["ret_3m"].dropna().mean()


def test_drift_knob_shifts_mean_return():
    """A positive drift_bps should shift mean forward returns upward."""
    null_ev, _ = data.synthetic_events(n_stocks=15, n_events=60, drift_bps=0.0, seed=42)
    planted_ev, _ = data.synthetic_events(n_stocks=15, n_events=60, drift_bps=30.0, seed=42)
    # The planted mean should be materially larger for 3m (63 days * 30 bps/day)
    null_mean = null_ev["ret_3m"].dropna().mean()
    planted_mean = planted_ev["ret_3m"].dropna().mean()
    assert planted_mean > null_mean


def test_split_ratios_are_positive(null_events):
    events, _ = null_events
    assert (events["split_ratio"] > 1.0).all()


def test_fingerprint_stable_and_content_sensitive(null_events):
    events, _ = null_events
    fp1 = data.fingerprint(events)
    fp2 = data.fingerprint(events)
    assert fp1 == fp2
    other, _ = data.synthetic_events(n_stocks=10, n_events=40, drift_bps=0.0, seed=999)
    assert data.fingerprint(events) != data.fingerprint(other)


def test_fetch_splits_panel_cache_only_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_splits_panel(tickers=["FAKE"], fetch=False, cache_dir=str(tmp_path))

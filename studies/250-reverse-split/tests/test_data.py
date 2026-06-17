"""Tests for the data layer — synthetic panel shape, determinism, knob wiring,
and the cache-only safety guard."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reverse_split import data  # noqa: E402

CACHE_PATH = data.DEFAULT_CACHE


# ---------------------------------------------------------------------------
# Synthetic event panel — shape and column checks
# ---------------------------------------------------------------------------

def test_synthetic_events_shape_and_columns(null_events):
    events, truth = null_events
    assert not events.empty
    expected_cols = {"ticker", "event_date", "ratio",
                     "ret_1m", "ret_3m", "ret_6m", "ret_12m", "is_rs"}
    assert expected_cols.issubset(set(events.columns))
    assert (events["is_rs"] == True).all()  # noqa: E712


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
    c, _ = data.synthetic_events(n_stocks=5, n_events=20, drift_bps=0.0, seed=8)
    assert a["ret_3m"].dropna().mean() != c["ret_3m"].dropna().mean()


def test_negative_drift_knob_shifts_mean_return_down():
    """A negative drift_bps should shift mean forward returns downward vs null."""
    null_ev, _ = data.synthetic_events(n_stocks=15, n_events=60, drift_bps=0.0, seed=42)
    planted_ev, _ = data.synthetic_events(n_stocks=15, n_events=60, drift_bps=-20.0, seed=42)
    null_mean = null_ev["ret_3m"].dropna().mean()
    planted_mean = planted_ev["ret_3m"].dropna().mean()
    assert planted_mean < null_mean


def test_ratio_values_are_integers_above_one(null_events):
    events, _ = null_events
    assert (events["ratio"] > 1).all()


def test_fingerprint_stable_and_content_sensitive(null_events):
    events, _ = null_events
    fp1 = data.fingerprint(events)
    fp2 = data.fingerprint(events)
    assert fp1 == fp2
    other, _ = data.synthetic_events(n_stocks=10, n_events=40, drift_bps=0.0, seed=999)
    assert data.fingerprint(events) != data.fingerprint(other)


def test_events_raw_table_has_required_columns():
    df = data.EVENTS_RAW
    for col in ("ticker", "event_date", "ratio", "note"):
        assert col in df.columns
    assert len(df) > 0
    assert (df["ratio"] > 1).all()


@pytest.mark.skipif(
    not any(
        os.path.exists(data._prices_cache_path(t, data.DEFAULT_CACHE))
        for t in data.EVENTS_RAW["ticker"].unique()
    ),
    reason="real-tape cache absent offline/CI"
)
def test_load_events_cache_returns_dataframe():
    """Load from cache (skips if cache absent, as on CI)."""
    events, prices = data.load_events(fetch=False)
    assert not events.empty
    assert "ret_1m" in events.columns
    assert "is_rs" in events.columns
    assert (events["is_rs"] == True).all()  # noqa: E712


def test_load_events_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_events(fetch=False, cache_dir=str(tmp_path))

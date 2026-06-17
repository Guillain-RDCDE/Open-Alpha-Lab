"""Tests for the data layer — synthetic panel shape, determinism, knob wiring,
and the cache-only safety guard."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spinoffs import data  # noqa: E402

# Path to study-local cache (absent on CI)
_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
_SPY_CACHE = os.path.join(_CACHE_DIR, "prices_239_SPY_1d.parquet")


# ---------------------------------------------------------------------------
# Synthetic event panel
# ---------------------------------------------------------------------------

def test_synthetic_events_shape_and_columns(null_events):
    events, truth = null_events
    assert not events.empty
    expected_cols = {"child", "parent", "ex_date",
                     "ret_6m", "ret_12m", "ret_18m", "ret_24m",
                     "spy_6m", "spy_12m", "spy_18m", "spy_24m",
                     "alpha_6m", "alpha_12m", "alpha_18m", "alpha_24m"}
    assert expected_cols.issubset(set(events.columns))


def test_synthetic_events_returns_finite(null_events):
    events, _ = null_events
    for col in ("ret_12m", "spy_12m", "alpha_12m"):
        vals = events[col].dropna()
        assert len(vals) > 0
        assert np.all(np.isfinite(vals))


def test_synthetic_events_deterministic():
    a, _ = data.synthetic_events(n_events=20, premium_bps=0.0, seed=77)
    b, _ = data.synthetic_events(n_events=20, premium_bps=0.0, seed=77)
    assert list(a["alpha_12m"].dropna().values) == list(b["alpha_12m"].dropna().values)
    c, _ = data.synthetic_events(n_events=20, premium_bps=0.0, seed=78)
    assert a["alpha_12m"].dropna().mean() != c["alpha_12m"].dropna().mean()


def test_premium_knob_shifts_alpha():
    """Positive premium_bps should shift child vs SPY alpha upward."""
    null_ev, _ = data.synthetic_events(n_events=40, premium_bps=0.0, seed=42)
    planted_ev, _ = data.synthetic_events(n_events=40, premium_bps=30.0, seed=42)
    null_mean = null_ev["alpha_12m"].dropna().mean()
    planted_mean = planted_ev["alpha_12m"].dropna().mean()
    assert planted_mean > null_mean


def test_alpha_equals_ret_minus_spy(null_events):
    """alpha = ret - spy for every row with finite values."""
    events, _ = null_events
    valid = events.dropna(subset=["ret_12m", "spy_12m", "alpha_12m"])
    diff = (valid["ret_12m"] - valid["spy_12m"] - valid["alpha_12m"]).abs()
    assert diff.max() < 1e-10


def test_fingerprint_stable_and_content_sensitive(null_events):
    events, _ = null_events
    fp1 = data.fingerprint(events)
    fp2 = data.fingerprint(events)
    assert fp1 == fp2
    other, _ = data.synthetic_events(n_events=20, premium_bps=0.0, seed=999)
    assert data.fingerprint(events) != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Real cache guard
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(_SPY_CACHE), reason="real-tape cache absent offline/CI")
def test_fetch_spinoff_panel_loads_from_cache():
    events, prices = data.fetch_spinoff_panel(fetch=False, cache_dir=_CACHE_DIR)
    assert not events.empty
    assert "SPY" in prices.columns


def test_fetch_spinoff_panel_cache_only_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_spinoff_panel(fetch=False, cache_dir=str(tmp_path))

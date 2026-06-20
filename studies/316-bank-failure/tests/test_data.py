"""The synthetic tape is well-formed and deterministic; the event table is sane;
the real loader is cache-safe (and cache-gated)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bank_failure import data  # noqa: E402


def test_event_table_is_sane():
    ev = data.events()
    assert list(ev.columns) == ["date", "label", "region"]
    assert len(ev) >= 8
    assert ev["date"].is_monotonic_increasing
    assert ev["date"].dt.tz is None
    # the canonical headline failures are present
    labels = " ".join(ev["label"])
    for name in ("Lehman", "Silicon Valley", "Credit Suisse", "Bear Stearns"):
        assert name in labels


def test_event_table_as_of_truncates():
    ev = data.events(as_of="2009-01-01")
    assert (ev["date"] <= pd.Timestamp("2009-01-01")).all()
    assert "Silicon Valley Bank (FDIC seizure)" not in set(ev["label"])  # 2023 dropped


def test_synthetic_shape_and_schema(null_tape):
    close, ev, truth = null_tape
    assert isinstance(close, pd.Series)
    assert len(close) == truth["n_days"]
    assert (close > 0).all()
    assert close.index.is_monotonic_increasing
    assert list(ev.columns) == ["date", "label", "region"]
    assert len(ev) == truth["n_events"]
    # planted event dates are inside the sample
    assert ev["date"].isin(close.index).all()


def test_synthetic_is_deterministic():
    a, _, _ = data.synthetic_tape(n_days=1500, drift=0.05, seed=7)
    b, _, _ = data.synthetic_tape(n_days=1500, drift=0.05, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _, _ = data.synthetic_tape(n_days=1500, drift=0.05, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_drift_actually_lifts_post_event_returns():
    """The drift knob really raises the cumulative move after planted events."""
    from bank_failure import strategy as st
    flat, ef, _ = data.synthetic_tape(n_days=6000, drift=0.0, seed=316)
    up, eu, _ = data.synthetic_tape(n_days=6000, drift=0.10, seed=316)
    car_flat = st.post_event_car(st.event_window_returns(flat, ef["date"]), 20).mean()
    car_up = st.post_event_car(st.event_window_returns(up, eu["date"]), 20).mean()
    assert car_up > car_flat + 0.02  # the +0.10 planted drift is clearly visible


def test_random_dates_reproducible_and_in_range(null_tape):
    close, _, _ = null_tape
    a = data.random_dates(close, n=12, seed=3)
    b = data.random_dates(close, n=12, seed=3)
    assert a.equals(b)
    assert a.isin(close.index).all()
    c = data.random_dates(close, n=12, seed=4)
    assert not a.equals(c)


def test_fingerprint_stable_and_content_sensitive(null_tape):
    close, _, _ = null_tape
    assert data.fingerprint(close) == data.fingerprint(close)
    other, _, _ = data.synthetic_tape(n_days=6000, drift=0.0, seed=99)
    assert data.fingerprint(close) != data.fingerprint(other)


def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real("NOPE_TICKER", fetch=False, cache_dir=str(tmp_path),
                       use_quantlab=False)


# --- cache-gated real-tape smoke test (skips offline) ----------------------
_SPY_CACHE = data._cache_path("SPY", data.DEFAULT_CACHE)


@pytest.mark.skipif(not os.path.exists(_SPY_CACHE), reason="offline CI — no real cache")
def test_load_real_cache_hit_well_formed():
    s = data.load_real("SPY", fetch=False)
    assert isinstance(s, pd.Series)
    assert s.index.tz is None
    assert (s > 0).all()

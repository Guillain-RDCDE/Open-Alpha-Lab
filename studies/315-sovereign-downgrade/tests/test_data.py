"""The synthetic tape is well-formed and deterministic; the downgrade table is sane; the
real loader is cache-safe (cache-gated so it skips offline)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sovereign_downgrade import data  # noqa: E402

CACHE = data._cache_path("SPY")


# ---- synthetic tape -------------------------------------------------------
def test_synthetic_shape_and_ohlc(planted):
    bars, events, truth = planted
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()
    assert events.size == truth["n_events"] > 0


def test_synthetic_is_deterministic():
    a, ea, _ = data.synthetic_spy(n_days=400, event_effect=0.03, seed=5)
    b, eb, _ = data.synthetic_spy(n_days=400, event_effect=0.03, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert list(ea) == list(eb)
    c, _, _ = data.synthetic_spy(n_days=400, event_effect=0.03, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_planted_dip_lowers_event_day_return(planted, null):
    """The event_effect knob really pushes the event-bar return negative (and 0 doesn't)."""
    from sovereign_downgrade import strategy as st

    pbars, pev, _ = planted
    nbars, nev, _ = null
    # canonical lag=1; the planted dip lands on the announcement bar (lag=0) and day after,
    # so the event-bar abnormal return at lag=0 should be clearly negative.
    p_abn = st.event_day_abnormal(pbars, pev, lag=0)["abn"].mean()
    n_abn = st.event_day_abnormal(nbars, nev, lag=0)["abn"].mean()
    assert p_abn < -0.010          # planted: a real announcement-day drop
    assert abs(n_abn) < 0.005      # null: no special move
    assert p_abn < n_abn


def test_null_events_are_well_formed(null):
    bars, _, _ = null
    ev = data.synthetic_null_events(bars, n_events=50, window=20, seed=3)
    assert isinstance(ev, pd.DatetimeIndex)
    assert len(ev) == 50
    assert ev.is_monotonic_increasing
    assert set(ev) <= set(bars.index)


def test_downgrade_table_is_sane():
    s = data.DOWNGRADES
    assert {"name", "agency", "announce", "from_", "to_"} <= set(s.columns)
    assert len(s) == 3                       # exactly the three US sovereign downgrades
    assert s["announce"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(s["announce"])
    dl = data.announcements()
    assert len(dl) == len(s)
    # the first downgrade is the 2011 S&P action
    assert dl.min() == pd.Timestamp("2011-08-05")
    # agencies are the three majors
    assert set(s["agency"]) == {"S&P", "Fitch", "Moody's"}


def test_fingerprint_is_stable_and_content_sensitive(planted):
    bars, _, _ = planted
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _, _ = data.synthetic_spy(event_effect=0.04, n_events=40, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# ---- real loader (cache-gated: skips offline) -----------------------------
@pytest.mark.skipif(not os.path.exists(CACHE), reason="offline CI: no shared SPY cache")
def test_load_real_reads_cache():
    bars = data.load_real("SPY", end="2026-05-31")
    assert list(bars.columns[:4]) == ["open", "high", "low", "close"]
    assert bars.index.is_monotonic_increasing
    assert bars.index.tz is None
    assert (bars["close"] > 0).all()

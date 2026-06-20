"""The synthetic tape is well-formed and deterministic; the shutdown table is sane; the
real loader is cache-safe."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from government_shutdown import data  # noqa: E402

CACHE = os.path.join(data.DEFAULT_CACHE, "SPY_total_return.parquet")


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
    a, ea, _ = data.synthetic_daily(n_days=400, event_effect=0.001, seed=5)
    b, eb, _ = data.synthetic_daily(n_days=400, event_effect=0.001, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert list(ea) == list(eb)
    c, _, _ = data.synthetic_daily(n_days=400, event_effect=0.001, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_planted_effect_lifts_post_event_returns(planted, null):
    """The event_effect knob really raises the post-event forward return (and 0 does not)."""
    from government_shutdown import strategy as st

    pbars, pev, _ = planted
    nbars, nev, _ = null
    pmean = st.event_forward_returns(pbars, pev, horizon=20)["ret_gross"].mean()
    nmean = st.event_forward_returns(nbars, nev, horizon=20)["ret_gross"].mean()
    assert pmean > nmean
    assert pmean > 0.02  # a clearly positive planted bounce over 20 days


def test_null_events_are_well_formed(null):
    bars, _, _ = null
    ev = data.synthetic_null_events(bars, n_events=50, window=20, seed=3)
    assert isinstance(ev, pd.DatetimeIndex)
    assert len(ev) == 50
    assert ev.is_monotonic_increasing
    assert set(ev) <= set(bars.index)


def test_shutdown_table_is_sane():
    s = data.SHUTDOWNS
    assert {"name", "start", "days"} <= set(s.columns)
    assert len(s) >= 4
    assert s["start"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(s["start"])
    starts = data.shutdown_starts()
    assert len(starts) == len(s)
    # every shutdown is in the SPY era
    assert starts.min() >= pd.Timestamp("1993-01-01")


def test_fingerprint_is_stable_and_content_sensitive(planted):
    bars, _, _ = planted
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _, _ = data.synthetic_daily(event_effect=0.002, n_events=40, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# ---- real loader (cache-gated: skips offline) -----------------------------
@pytest.mark.skipif(not os.path.exists(CACHE), reason="offline CI: no shared SPY cache")
def test_load_real_reads_cache():
    bars = data.load_real("SPY", end="2026-05-31")
    assert list(bars.columns[:4]) == ["open", "high", "low", "close"]
    assert bars.index.is_monotonic_increasing
    assert bars.index.tz is None
    assert (bars["close"] > 0).all()
    assert bars.index[0] >= pd.Timestamp("1993-01-01")

"""The synthetic tape is well-formed and deterministic; the X-date table is sane; the
real loader is cache-safe (cache-gated so it skips offline)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debt_ceiling import data  # noqa: E402

CACHE = data._cache_path("^VIX")


# ---- synthetic tape -------------------------------------------------------
def test_synthetic_shape_and_ohlc(planted):
    bars, events, truth = planted
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume", "vix"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()
    assert (bars["vix"] > 0).all()
    assert events.size == truth["n_events"] > 0


def test_synthetic_is_deterministic():
    a, ea, _ = data.synthetic_vix(n_days=400, event_effect=0.3, seed=5)
    b, eb, _ = data.synthetic_vix(n_days=400, event_effect=0.3, seed=5)
    assert np.allclose(a["vix"].to_numpy(), b["vix"].to_numpy())
    assert list(ea) == list(eb)
    c, _, _ = data.synthetic_vix(n_days=400, event_effect=0.3, seed=6)
    assert not np.allclose(a["vix"].to_numpy(), c["vix"].to_numpy())


def test_planted_hump_lifts_pre_and_drops_post(planted, null):
    """The event_effect knob really raises pre-deadline vol and lowers post (and 0 doesn't)."""
    from debt_ceiling import strategy as st

    pbars, pev, _ = planted
    nbars, nev, _ = null
    p_pre = st.pre_ramp(pbars, pev, pre=20)["d_log"].mean()
    p_post = st.post_collapse(pbars, pev, post=20)["d_log"].mean()
    n_pre = st.pre_ramp(nbars, nev, pre=20)["d_log"].mean()
    # planted: vol clearly rises into the deadline and falls after it
    assert p_pre > 0.10
    assert p_post < -0.10
    # null: the pre-ramp is small (no planted hump)
    assert abs(n_pre) < 0.10
    assert p_pre > n_pre


def test_null_events_are_well_formed(null):
    bars, _, _ = null
    ev = data.synthetic_null_events(bars, n_events=50, window=20, seed=3)
    assert isinstance(ev, pd.DatetimeIndex)
    assert len(ev) == 50
    assert ev.is_monotonic_increasing
    assert set(ev) <= set(bars.index)


def test_xdate_table_is_sane():
    s = data.X_DATES
    assert {"name", "deadline", "x_date", "ramp"} <= set(s.columns)
    assert len(s) >= 5
    assert s["deadline"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(s["deadline"])
    dl = data.deadlines()
    assert len(dl) == len(s)
    # every episode is in the VIX era (post-1990)
    assert dl.min() >= pd.Timestamp("1990-01-01")


def test_fingerprint_is_stable_and_content_sensitive(planted):
    bars, _, _ = planted
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _, _ = data.synthetic_vix(event_effect=0.5, n_events=40, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# ---- real loader (cache-gated: skips offline) -----------------------------
@pytest.mark.skipif(not os.path.exists(CACHE), reason="offline CI: no shared VIX cache")
def test_load_real_reads_cache():
    bars = data.load_real("^VIX", end="2026-05-31")
    assert list(bars.columns[:4]) == ["open", "high", "low", "close"]
    assert bars.index.is_monotonic_increasing
    assert bars.index.tz is None
    assert (bars["close"] > 0).all()
    assert bars.index[0] >= pd.Timestamp("1990-01-01")

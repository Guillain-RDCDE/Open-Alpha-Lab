"""The synthetic tape is well-formed and deterministic; the calendar is sane; the real
fetch is cache-safe (and cache-gated)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jackson_hole import data  # noqa: E402


def test_synthetic_shape_and_dates(null_tape):
    ret, speech, truth = null_tape
    assert isinstance(ret, pd.Series)
    assert ret.name == "ret"
    assert len(ret) > 252 * 30
    assert truth.bump == 0.0 and not truth.has_effect
    # one speech date per simulated year, all in August, all Fridays
    assert len(speech) >= 30
    assert (speech.month == 8).all()
    assert (speech.weekday == 4).all()  # Friday


def test_synthetic_is_deterministic():
    a, _, _ = data.synthetic_world(n_years=20, bump=0.005, seed=7)
    b, _, _ = data.synthetic_world(n_years=20, bump=0.005, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _, _ = data.synthetic_world(n_years=20, bump=0.005, seed=8)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_bump_lifts_speech_day_mean(null_tape, planted_tape):
    """The bump knob really raises the planted speech-day return (and 0 does not)."""
    from jackson_hole import strategy as st
    r0, sp0, _ = null_tape
    r1, sp1, _ = planted_tape
    m0 = st.event_table(r0, sp0, 0)["event_mean"]
    m1 = st.event_table(r1, sp1, 0)["event_mean"]
    assert m1 - m0 > 0.005  # ~the planted +80 bps shows up


def test_jh_speech_dates_calendar():
    """The real keynote calendar: one per year, all in late August, no future dates."""
    d = data.JH_SPEECH_DATES
    assert len(d) >= 30
    assert (d.month == 8).all()
    assert d.is_monotonic_increasing
    # one per distinct year
    assert d.year.nunique() == len(d)
    # not in the future relative to a generous cutoff (study as-of 2026)
    assert d.max() <= pd.Timestamp("2026-12-31")


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    ret, _, _ = null_tape
    assert data.fingerprint(ret) == data.fingerprint(ret)
    other, _, _ = data.synthetic_world(n_years=33, bump=0.0, seed=99)
    assert data.fingerprint(ret) != data.fingerprint(other)


def test_fetch_spy_cache_only_returns_empty_without_cache(tmp_path):
    """With a fresh (empty) cache dir and no fetch, fetch_spy returns an empty Series."""
    out = data.fetch_spy(cache_dir=str(tmp_path), fetch=False)
    assert out.empty


@pytest.mark.skipif(
    not os.path.exists(data.SHARED_SPY_CACHE), reason="offline CI: shared SPY cache absent"
)
def test_shared_spy_cache_loads():
    """When the shared cache exists, it loads as a tz-naive daily-return Series."""
    ret = data.fetch_spy()
    assert not ret.empty
    assert ret.index.tz is None
    assert ret.index.is_monotonic_increasing

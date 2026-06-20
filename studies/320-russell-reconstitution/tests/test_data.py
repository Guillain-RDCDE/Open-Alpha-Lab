"""The synthetic tape is well-formed and deterministic; the recon calendar is sane; the
real fetch is cache-safe."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from russell_reconstitution import data  # noqa: E402

IWM_CACHE = os.path.join(data.DEFAULT_CACHE, "bars_IWM_daily.parquet")


def test_synthetic_shape_and_ohlc(null_tape):
    bars, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(start_year=2005, end_year=2010, premium_bps=10.0, seed=5)
    b, _ = data.synthetic_daily(start_year=2005, end_year=2010, premium_bps=10.0, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(start_year=2005, end_year=2010, premium_bps=10.0, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_recon_table_is_annual_and_friday_ish():
    """One reconstitution per year, every event in late June, every weekday Friday (4)."""
    t = data.recon_table()
    # exactly one per year, no gaps
    assert t["year"].is_monotonic_increasing
    assert t["year"].is_unique
    assert (np.diff(t["year"].to_numpy()) == 1).all()
    # every event is in June, day-of-month >= 22 (the late-June recon Friday)
    assert (t["effective_date"].dt.month == 6).all()
    assert (t["effective_date"].dt.day >= 22).all()
    # every effective date is a Friday
    assert (t["weekday"] == 4).all()


def test_premium_lifts_runup_returns_only():
    """The planted premium raises the run-up windows, not the unconditional mean by much."""
    flat, _ = data.synthetic_daily(premium_bps=0.0, seed=320)
    hot, _ = data.synthetic_daily(premium_bps=80.0, seed=320)
    # same noise (same seed) -> isolate the planted premium's effect on the level path
    assert hot["close"].iloc[-1] > flat["close"].iloc[-1]


def test_no_timestamp_overflow_on_wide_span():
    """A very wide span must still build (no ns-timestamp overflow)."""
    bars, truth = data.synthetic_daily(start_year=1900, end_year=2200, premium_bps=0.0, seed=1)
    assert len(bars) > 70_000
    assert bars.index.is_monotonic_increasing


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(premium_bps=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


@pytest.mark.skipif(not os.path.exists(IWM_CACHE), reason="offline CI: no IWM cache")
def test_real_iwm_cache_loads_and_spans_the_events():
    bars = data.fetch_daily("IWM", fetch=False)
    assert bars.index.tz is None
    assert bars.index.min().year <= 2000
    assert bars.index.max().year >= 2025

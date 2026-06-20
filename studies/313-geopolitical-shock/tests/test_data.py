"""The curated shock table is well-formed, the synthetic tape is deterministic and plants
what it claims, and the real loader is cache-safe (skips when the cache is absent)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geopolitical_shock import data  # noqa: E402

CACHE = data._cache_path("SPY", data.DEFAULT_CACHE)


# ---- the curated shock table ----------------------------------------------
def test_shock_table_well_formed():
    tbl = data.shock_table()
    assert list(tbl.columns) == ["trade_date", "label"]
    assert len(tbl) >= 20
    assert tbl["trade_date"].is_monotonic_increasing
    assert tbl["trade_date"].is_unique
    assert tbl["label"].str.len().gt(0).all()
    # All dates are real, parseable, and not in the future relative to the table's span.
    assert pd.api.types.is_datetime64_any_dtype(tbl["trade_date"])


def test_shock_table_dates_are_weekdays():
    """Each trade date should be a NYSE-tradable weekday (Mon-Fri), not a weekend."""
    tbl = data.shock_table()
    assert (tbl["trade_date"].dt.dayofweek < 5).all()


# ---- the synthetic tape ----------------------------------------------------
def test_synthetic_shape_and_ohlc(null_tape):
    bars, events, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()
    assert list(events.columns) == ["trade_date", "label"]
    assert len(events) == truth["n_shocks"]


def test_synthetic_is_deterministic():
    a, ea, _ = data.synthetic_daily(shock_drift=0.02, seed=7)
    b, eb, _ = data.synthetic_daily(shock_drift=0.02, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert (ea["trade_date"].to_numpy() == eb["trade_date"].to_numpy()).all()
    c, _, _ = data.synthetic_daily(shock_drift=0.02, seed=8)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_shock_drift_is_actually_planted():
    """A positive shock_drift must lift the average post-event log return above the null."""
    null, ev0, _ = data.synthetic_daily(shock_drift=0.0, seed=313)
    relief, ev1, _ = data.synthetic_daily(shock_drift=0.03, seed=313)
    # events are at the same locations (same seed), so compare the post-window means.
    def post_mean(bars, events, post=10):
        r = np.log(bars["close"]).diff()
        idx = bars.index
        vals = []
        for d in events["trade_date"]:
            p = idx.searchsorted(d)
            vals.append(r.to_numpy()[p + 1 : p + 1 + post].sum())
        return np.nanmean(vals)
    assert post_mean(relief, ev1) > post_mean(null, ev0) + 0.01


def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, _, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _, _ = data.synthetic_daily(shock_drift=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# ---- real tape: cache-gated, never skips the synthetic core -----------------
@pytest.mark.skipif(not os.path.exists(CACHE), reason="offline CI: no SPY cache")
def test_real_spy_loads_and_is_clean():
    bars = data.load_real("SPY")
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.is_monotonic_increasing
    assert bars.index.tz is None
    assert (bars["close"] > 0).all()

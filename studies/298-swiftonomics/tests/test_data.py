"""Tests for the data layer of Study 298 (Swiftonomics).

All tests are offline and deterministic -- they use the hardcoded Eras Tour event
table and the synthetic generator only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swiftonomics import data  # noqa: E402


# ---------------------------------------------------------------------------
# Hardcoded Eras Tour event table
# ---------------------------------------------------------------------------
def test_events_has_expected_rows(eras_events):
    assert len(eras_events) == 16


def test_events_columns(eras_events):
    assert {"date", "label", "kind"}.issubset(set(eras_events.columns))


def test_events_dates_are_datetime(eras_events):
    assert pd.api.types.is_datetime64_any_dtype(eras_events["date"])


def test_events_dates_sorted(eras_events):
    d = eras_events["date"]
    assert (d.values[1:] >= d.values[:-1]).all()


def test_events_kinds_valid(eras_events):
    assert set(eras_events["kind"].unique()).issubset({"announce", "milestone"})


def test_events_cover_tour_window(eras_events):
    """Events span the 2022 announcement through the 2024 finale."""
    assert eras_events["date"].min().year == 2022
    assert eras_events["date"].max().year == 2024


def test_events_include_ticketmaster_meltdown(eras_events):
    labels = " ".join(eras_events["label"]).lower()
    assert "ticketmaster" in labels
    assert "final show" in labels


def test_eras_events_is_copy():
    a = data.eras_events()
    b = data.eras_events()
    a.loc[0, "label"] = "XXX"
    assert b.loc[0, "label"] != "XXX"


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_shape(synthetic_null):
    prices, events, truth = synthetic_null
    assert prices.shape[0] == truth["n_days"]
    assert list(prices.columns) == [data.STOCK, data.MARKET]
    assert len(events) == truth["n_events"]


def test_synthetic_is_deterministic():
    a, _, _ = data.synthetic_panel(bump_bps=0.0, seed=42)
    b, _, _ = data.synthetic_panel(bump_bps=0.0, seed=42)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_different_seeds_differ():
    a, _, _ = data.synthetic_panel(bump_bps=0.0, seed=1)
    b, _, _ = data.synthetic_panel(bump_bps=0.0, seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_prices_positive(synthetic_null):
    prices, _, _ = synthetic_null
    assert (prices > 0).all().all()


def test_synthetic_event_dates_are_trading_days(synthetic_null):
    prices, events, _ = synthetic_null
    assert events["date"].isin(prices.index).all()


def test_synthetic_bump_changes_stock_only():
    """Planting a bump moves the stock returns but leaves the market untouched."""
    p0, _, _ = data.synthetic_panel(bump_bps=0.0, seed=298)
    p1, _, _ = data.synthetic_panel(bump_bps=500.0, seed=298)
    # Market index identical (same seed, same draws)
    assert np.allclose(p0[data.MARKET].to_numpy(), p1[data.MARKET].to_numpy())
    # Stock differs
    assert not np.allclose(p0[data.STOCK].to_numpy(), p1[data.STOCK].to_numpy())


def test_fingerprint_stable_and_content_sensitive():
    p1, _, _ = data.synthetic_panel(bump_bps=0.0, seed=10)
    p2, _, _ = data.synthetic_panel(bump_bps=0.0, seed=11)
    assert data.fingerprint(p1) == data.fingerprint(p1)
    assert data.fingerprint(p1) != data.fingerprint(p2)


def test_fetch_prices_raises_without_cache(tmp_path):
    """Without the cache, fetch_prices(fetch=False) raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        data.fetch_prices(fetch=False, cache_path=str(tmp_path / "nope.parquet"))

"""Synthetic data layer — deterministic, overflow-safe, and faithfully structured.

All offline on the seeded world; the one test that touches a real/shared cache parquet is
cache-gated so CI never needs the network."""

import os

import numpy as np
import pandas as pd
import pytest

from low_volatility_anomaly import data


def test_world_is_deterministic(edge_world):
    df, truth = edge_world
    df2, truth2 = data.synthetic_world(lowvol_edge=0.06, seed=330)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_edge and truth2.has_edge


def test_world_has_three_legs(edge_world):
    df, _ = edge_world
    assert list(df.columns) == ["SPLV", "SPHB", "SPY"]
    assert df.index.name == "date"
    assert df.notna().all().all()


def test_low_leg_is_calmer_than_high_leg(edge_world):
    df, _ = edge_world
    # By construction beta_low < beta_high, so SPLV vol < SPHB vol.
    assert df["SPLV"].std() < df["SPHB"].std()


def test_index_is_overflow_safe():
    # A long synthetic span must still produce valid, monotone ns-Timestamps
    # (period_range, never date_range(periods=BIG, freq="M")).
    df, _ = data.synthetic_world(n_years=120, seed=1)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df.index[-1].year < 2200


def test_null_world_has_no_planted_edge(null_world):
    _, truth = null_world
    assert not truth.has_edge


def test_fetch_pairs_offline_returns_empty_or_cache():
    # Offline (no fetch): either a cached frame or an empty one, never a crash.
    df = data.fetch_pairs(fetch=False)
    assert isinstance(df, pd.DataFrame)


def test_fingerprint_is_stable(edge_world):
    df, _ = edge_world
    assert data.fingerprint(df) == data.fingerprint(df.copy())
    assert len(data.fingerprint(df)) == 12


@pytest.mark.skipif(
    data._existing_cache() is None, reason="offline CI: no real SPLV/SPHB cache"
)
def test_real_cache_has_expected_columns():
    df = data.fetch_pairs(fetch=False)
    assert not df.empty
    for col in ("SPLV", "SPHB"):
        assert col in df.columns

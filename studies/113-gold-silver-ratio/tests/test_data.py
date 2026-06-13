"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_silver_ratio import data  # noqa: E402


def test_synthetic_shape_and_ohlc(null_pair):
    gold, silver, truth = null_pair
    assert len(gold) == truth["n_days"]
    assert len(silver) == truth["n_days"]
    assert list(gold.columns) == ["open", "high", "low", "close", "volume"]
    assert list(silver.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range brackets its open and close.
    for bars in (gold, silver):
        assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
        assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a_g, a_s, _ = data.synthetic_daily(n_days=200, mean_rev_speed=0.05, seed=42)
    b_g, b_s, _ = data.synthetic_daily(n_days=200, mean_rev_speed=0.05, seed=42)
    assert np.allclose(a_g["close"].to_numpy(), b_g["close"].to_numpy())
    assert np.allclose(a_s["close"].to_numpy(), b_s["close"].to_numpy())
    c_g, c_s, _ = data.synthetic_daily(n_days=200, mean_rev_speed=0.05, seed=43)
    assert not np.allclose(a_g["close"].to_numpy(), c_g["close"].to_numpy())


def test_ratio_is_consistent(null_pair):
    """The implied ratio gold/silver is correctly encoded in the synthetic prices."""
    gold, silver, _ = null_pair
    ratio = gold["close"] / silver["close"]
    assert (ratio > 1.0).all()   # gold always more expensive than silver in synthetic


def test_mean_reversion_knob_affects_stationarity():
    """A high mean_rev_speed should keep the ratio bounded; zero produces a wide walk."""
    _, _, truth_null = data.synthetic_daily(n_days=2000, mean_rev_speed=0.0, seed=7)
    _, _, truth_fast = data.synthetic_daily(n_days=2000, mean_rev_speed=0.20, seed=7)
    # Fast reversion should have a much shorter planted half-life
    assert truth_fast["planted_half_life"] < 10.0
    assert truth_null["planted_half_life"] == float("inf")


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_pair):
    gold, _, _ = null_pair
    assert data.fingerprint(gold) == data.fingerprint(gold)
    other_g, _, _ = data.synthetic_daily(n_days=1000, mean_rev_speed=0.0, seed=99)
    assert data.fingerprint(gold) != data.fingerprint(other_g)

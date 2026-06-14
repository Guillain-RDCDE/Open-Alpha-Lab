"""The synthetic tape is well-formed and deterministic; the real loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from permanent_portfolio import data  # noqa: E402


def test_synthetic_shape_and_columns(null_world):
    frame, truth = null_world
    assert set(frame.columns) == {"STK", "BOND", "GOLD", "CASH"}
    assert len(frame) == truth["n_days"]
    assert (frame > 0).all().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_four_asset(n_years=10, cycle_strength=0.2, seed=7)
    b, _ = data.synthetic_four_asset(n_years=10, cycle_strength=0.2, seed=7)
    assert np.allclose(a["STK"].to_numpy(), b["STK"].to_numpy())
    c, _ = data.synthetic_four_asset(n_years=10, cycle_strength=0.2, seed=8)
    assert not np.allclose(a["STK"].to_numpy(), c["STK"].to_numpy())


def test_synthetic_prices_strictly_positive(cycle_world):
    frame, _ = cycle_world
    assert (frame > 0).all().all()


def test_cycle_strength_affects_dispersion():
    """Higher cycle_strength should increase the spread of annual returns across legs."""
    flat, _ = data.synthetic_four_asset(n_years=20, cycle_strength=0.0, seed=144)
    cycled, _ = data.synthetic_four_asset(n_years=20, cycle_strength=0.8, seed=144)
    # In the cycle world the four legs should be more dispersed year-over-year
    flat_r = flat.pct_change().dropna()
    cycled_r = cycled.pct_change().dropna()
    flat_yr_std = flat_r.groupby(flat_r.index.year).sum().std(axis=1).mean()
    cycled_yr_std = cycled_r.groupby(cycled_r.index.year).sum().std(axis=1).mean()
    assert cycled_yr_std > flat_yr_std


def test_load_real_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(cache_dir=str(tmp_path), fetch=False)


def test_fingerprint_stable_and_content_sensitive(null_world):
    frame, _ = null_world
    assert data.fingerprint(frame) == data.fingerprint(frame)
    other, _ = data.synthetic_four_asset(n_years=20, cycle_strength=0.0, seed=999)
    assert data.fingerprint(frame) != data.fingerprint(other)

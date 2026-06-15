"""The synthetic tape is well-formed and deterministic; the real loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from golden_butterfly import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard — any test that reads the real cache must be skipped when absent
# ---------------------------------------------------------------------------
CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "_cache", "gb_panel.parquet",
)
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape — offline, no cache needed
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns(null_world):
    frame, truth = null_world
    assert set(frame.columns) == {"LCG", "SCV", "BOND", "CASH", "GOLD"}
    assert len(frame) == truth["n_days"]
    assert (frame > 0).all().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_five_asset(n_years=10, cycle_strength=0.2, seed=7)
    b, _ = data.synthetic_five_asset(n_years=10, cycle_strength=0.2, seed=7)
    assert np.allclose(a["LCG"].to_numpy(), b["LCG"].to_numpy())
    # Different seed gives different output
    c, _ = data.synthetic_five_asset(n_years=10, cycle_strength=0.2, seed=8)
    assert not np.allclose(a["LCG"].to_numpy(), c["LCG"].to_numpy())


def test_synthetic_prices_strictly_positive(cycle_world):
    frame, _ = cycle_world
    assert (frame > 0).all().all()


def test_cycle_strength_affects_dispersion():
    """Higher cycle_strength should increase the spread of annual returns across legs."""
    flat, _ = data.synthetic_five_asset(n_years=20, cycle_strength=0.0, seed=203)
    cycled, _ = data.synthetic_five_asset(n_years=20, cycle_strength=0.8, seed=203)
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
    other, _ = data.synthetic_five_asset(n_years=20, cycle_strength=0.0, seed=999)
    assert data.fingerprint(frame) != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Real tape — skip if cache absent
# ---------------------------------------------------------------------------
@requires_cache
def test_real_data_columns():
    """Real panel has all five GB tickers."""
    px = data.load_real(fetch=False)
    for t in ("SPY", "IWN", "TLT", "SHY", "GLD"):
        assert t in px.columns


@requires_cache
def test_real_data_starts_at_gld_inception():
    """Joint window begins at GLD inception (2004-11-18 or shortly after)."""
    import pandas as pd
    px = data.load_real(fetch=False)
    assert px.index[0] >= pd.Timestamp("2004-11-18")


@requires_cache
def test_real_data_no_nulls():
    """Price frame has no NaN values after the dropna() in load_real."""
    px = data.load_real(fetch=False)
    assert px.isnull().sum().sum() == 0


@requires_cache
def test_real_fingerprint_is_string(null_world):
    px = data.load_real(fetch=False)
    fp = data.fingerprint(px)
    assert isinstance(fp, str) and len(fp) == 12

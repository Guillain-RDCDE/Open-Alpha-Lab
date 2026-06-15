"""The synthetic tape is well-formed and deterministic; the real loader is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from talmud_portfolio import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard — real-data tests skip when the shared panel is absent
# ---------------------------------------------------------------------------
SHARED_PANEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache",
    "cross_asset_etfs.parquet",
)
STUDY_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "_cache", "talmud_prices.parquet",
)
requires_cache = pytest.mark.skipif(
    not os.path.exists(SHARED_PANEL) and not os.path.exists(STUDY_CACHE),
    reason="real price cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape tests (offline — no cache required)
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns(null_world):
    frame, truth = null_world
    assert list(frame.columns) == ["REIT", "STK", "BOND"]
    assert len(frame) == truth["n_days"]
    assert (frame > 0).all().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_three_asset(n_years=10, cycle_strength=0.0, seed=204)
    b, _ = data.synthetic_three_asset(n_years=10, cycle_strength=0.0, seed=204)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_three_asset(n_years=10, cycle_strength=0.0, seed=99)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_synthetic_different_seeds_differ():
    a, _ = data.synthetic_three_asset(n_years=10, seed=1)
    b, _ = data.synthetic_three_asset(n_years=10, seed=2)
    assert not np.allclose(a["STK"].to_numpy(), b["STK"].to_numpy())


def test_synthetic_prices_all_positive(cycle_world):
    frame, _ = cycle_world
    assert (frame > 0).all().all()


def test_synthetic_start_near_100(null_world):
    frame, _ = null_world
    # All columns start close to 100 (first day is the first compounded period)
    assert (frame.iloc[0] > 80).all()
    assert (frame.iloc[0] < 120).all()


def test_cycle_boosts_cross_asset_dispersion():
    """A planted cycle increases the cross-asset return dispersion relative to null."""
    null, _ = data.synthetic_three_asset(n_years=15, cycle_strength=0.0, seed=204)
    cycl, _ = data.synthetic_three_asset(n_years=15, cycle_strength=1.0, seed=204)
    # Compute annual returns for each leg
    null_ret = null.pct_change().dropna()
    cycl_ret = cycl.pct_change().dropna()
    # With planted cycle the cross-sectional dispersion of annual means should be larger
    null_disp = null_ret.mean().std()
    cycl_disp = cycl_ret.mean().std()
    assert cycl_disp > null_disp


def test_fingerprint_stable_and_content_sensitive(null_world, cycle_world):
    frame_n, _ = null_world
    frame_c, _ = cycle_world
    assert data.fingerprint(frame_n) == data.fingerprint(frame_n)
    assert data.fingerprint(frame_n) != data.fingerprint(frame_c)


def test_load_real_raises_without_cache(tmp_path):
    """Cache-gating: load_real raises FileNotFoundError when no cache is available."""
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Real-data tests (skipped on CI when cache is absent)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_prices_shape_and_columns():
    prices = data.load_real()
    assert set(data.ASSETS).issubset(set(prices.columns))
    assert len(prices) > 100


@requires_cache
def test_real_prices_all_positive():
    prices = data.load_real()
    assert (prices > 0).all().all()


@requires_cache
def test_real_fingerprint_stable():
    prices = data.load_real()
    fp1 = data.fingerprint(prices)
    fp2 = data.fingerprint(prices)
    assert fp1 == fp2
    assert len(fp1) == 12

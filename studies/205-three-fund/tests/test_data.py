"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from three_fund import data  # noqa: E402

_REAL_CACHE = data._cache_path(data.DEFAULT_CACHE)
requires_cache = pytest.mark.skipif(
    not os.path.exists(_REAL_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns(null_prices):
    prices, truth = null_prices
    assert list(prices.columns) == ["VTI", "VXUS", "BND", "SPY"]
    assert len(prices) == truth["n_days"]
    assert (prices > 0).all().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_prices(n_days=500, intl_alpha=0.01, seed=42)
    b, _ = data.synthetic_prices(n_days=500, intl_alpha=0.01, seed=42)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_prices(n_days=500, intl_alpha=0.01, seed=99)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_synthetic_null_seed_independence():
    """Different seeds produce genuinely different paths."""
    a, _ = data.synthetic_prices(n_days=252, seed=1)
    b, _ = data.synthetic_prices(n_days=252, seed=2)
    assert not np.allclose(a["VTI"].to_numpy(), b["VTI"].to_numpy())


def test_synthetic_intl_alpha_shifts_international(positive_intl_prices, null_prices):
    """Positive intl_alpha should produce higher average international returns."""
    pos_px, _ = positive_intl_prices
    null_px, _ = null_prices
    r_pos  = pos_px["VXUS"].pct_change().dropna().mean()
    r_null = null_px["VXUS"].pct_change().dropna().mean()
    assert r_pos > r_null


def test_spy_equals_vti_in_synthetic(null_prices):
    """Synthetic SPY is the same process as VTI (both track US equity)."""
    prices, _ = null_prices
    assert np.allclose(prices["VTI"].to_numpy(), prices["SPY"].to_numpy())


def test_fingerprint_stable_and_content_sensitive(null_prices):
    prices, _ = null_prices
    assert data.fingerprint(prices) == data.fingerprint(prices)
    other, _ = data.synthetic_prices(n_days=2520, intl_alpha=0.0, seed=99)
    assert data.fingerprint(prices) != data.fingerprint(other)


def test_load_real_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Real tape (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_data_has_expected_tickers():
    prices = data.load_real(fetch=False)
    for t in data.ALL_TICKERS:
        assert t in prices.columns, f"{t} missing from real prices"


@requires_cache
def test_real_data_starts_at_universe_start():
    prices = data.load_real(fetch=False)
    assert prices.index[0] >= pd.Timestamp(data.UNIVERSE_START)


@requires_cache
def test_real_data_all_positive():
    prices = data.load_real(fetch=False)
    assert (prices > 0).all().all()


@requires_cache
def test_real_fingerprint_is_stable():
    prices = data.load_real(fetch=False)
    fp = data.fingerprint(prices)
    assert isinstance(fp, str) and len(fp) == 12

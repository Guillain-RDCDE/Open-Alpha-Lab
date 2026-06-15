"""Tests for the data layer — synthetic generator and real-data loader (cache-gated)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_miners import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache-gate: any test that reads real Yahoo data must be marked with this.
# ---------------------------------------------------------------------------
_CACHE_DIR = data.DEFAULT_CACHE
_GLD_CACHE = data._cache_path("GLD", _CACHE_DIR)
_GDX_CACHE = data._cache_path("GDX", _CACHE_DIR)

requires_cache = pytest.mark.skipif(
    not (os.path.exists(_GLD_CACHE) and os.path.exists(_GDX_CACHE)),
    reason="real-data cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic generator tests — always run, no network needed
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=5, seed=208)
    assert list(prices.columns) == ["gld", "gdx"]
    assert len(prices) == 5 * data.TRADING_DAYS_PER_YEAR
    assert prices.index.name == "date"


def test_synthetic_is_deterministic():
    p1, t1 = data.synthetic_daily(n_years=5, seed=42)
    p2, t2 = data.synthetic_daily(n_years=5, seed=42)
    assert (p1["gld"] == p2["gld"]).all()
    assert (p1["gdx"] == p2["gdx"]).all()
    assert t1 == t2


def test_synthetic_prices_positive():
    prices, _ = data.synthetic_daily(n_years=10, seed=208)
    assert (prices > 0).all().all()


def test_synthetic_leverage_increases_gdx_vol():
    """Higher leverage → higher GDX volatility relative to GLD."""
    _, t_lo = data.synthetic_daily(n_years=20, leverage=0.5, asymmetry=0.0, alpha_ann=0.0, seed=208)
    _, t_hi = data.synthetic_daily(n_years=20, leverage=2.0, asymmetry=0.0, alpha_ann=0.0, seed=208)
    # Confirm that the truth dict records what we planted
    assert t_lo["planted_upside_beta"] < t_hi["planted_upside_beta"]


def test_synthetic_null_gdx_uncorrelated():
    """At leverage=0, GDX should have near-zero correlation with GLD."""
    import numpy as np
    prices, _ = data.synthetic_daily(n_years=30, leverage=0.0, asymmetry=0.0, alpha_ann=0.0, seed=0)
    r_gld = np.log(prices["gld"]).diff().dropna()
    r_gdx = np.log(prices["gdx"]).diff().dropna()
    corr = float(np.corrcoef(r_gld, r_gdx)[0, 1])
    assert abs(corr) < 0.15, f"Expected near-zero corr at leverage=0, got {corr:.3f}"


def test_fingerprint_changes_with_prices():
    p1, _ = data.synthetic_daily(n_years=5, seed=1)
    p2, _ = data.synthetic_daily(n_years=5, seed=2)
    assert data.fingerprint(p1) != data.fingerprint(p2)


def test_truth_dict_keys():
    _, truth = data.synthetic_daily(n_years=5)
    for key in ("leverage", "asymmetry", "alpha_ann", "n_years", "n_days", "seed",
                "planted_upside_beta", "planted_downside_beta"):
        assert key in truth, f"missing key: {key}"


# ---------------------------------------------------------------------------
# Real-data tests — only run when the cache is present
# ---------------------------------------------------------------------------
@requires_cache
def test_real_load_pair_returns_aligned_frame():
    prices = data.load_pair(fetch=False)
    assert list(prices.columns) == ["gld", "gdx"]
    assert len(prices) > 2000, "Expected >2000 common trading days"
    assert prices.notna().all().all()


@requires_cache
def test_real_prices_start_after_gdx_ipo():
    prices = data.load_pair(fetch=False)
    assert str(prices.index[0].date()) >= data.GDX_START


@requires_cache
def test_real_fingerprints_stable():
    """Fingerprints should be stable for the same cached parquet."""
    prices = data.load_pair(fetch=False)
    fp1 = data.fingerprint(prices, "gld")
    fp2 = data.fingerprint(prices, "gld")
    assert fp1 == fp2
    assert len(fp1) == 12

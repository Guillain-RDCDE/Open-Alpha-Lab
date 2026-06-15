"""The synthetic tape is well-formed, deterministic, and cache-gated for real data."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dividend_aristocrats import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard — real-data tests are skipped when the parquet is absent (CI)
# ---------------------------------------------------------------------------
_CACHE_DIR = data.DEFAULT_CACHE
_NOBL_CACHE = data._cache_path("NOBL", _CACHE_DIR)
_SPY_CACHE = data._cache_path("SPY", _CACHE_DIR)
_BOTH_CACHED = os.path.exists(_NOBL_CACHE) and os.path.exists(_SPY_CACHE)

requires_cache = pytest.mark.skipif(
    not _BOTH_CACHED,
    reason="price cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic tape tests — always run, no files required
# ---------------------------------------------------------------------------
def test_synthetic_shape(null_prices):
    prices, truth = null_prices
    assert set(prices.columns) == {"NOBL", "SPY"}
    assert len(prices) == truth["n_days"]
    assert prices.index.name == "date"


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=5, alpha_bps=0.0, seed=206)
    b, _ = data.synthetic_daily(n_years=5, alpha_bps=0.0, seed=206)
    assert np.allclose(a["NOBL"].to_numpy(), b["NOBL"].to_numpy())
    c, _ = data.synthetic_daily(n_years=5, alpha_bps=0.0, seed=99)
    assert not np.allclose(a["NOBL"].to_numpy(), c["NOBL"].to_numpy())


def test_synthetic_prices_positive(null_prices):
    prices, _ = null_prices
    assert (prices > 0).all().all()


def test_alpha_knob_shifts_mean_excess_return():
    """Planted alpha of +3 bps/day should produce a positive mean excess return."""
    from dividend_aristocrats import strategy as st
    prices_null, _ = data.synthetic_daily(n_years=10, alpha_bps=0.0, seed=206)
    prices_alpha, _ = data.synthetic_daily(n_years=10, alpha_bps=3.0, seed=206)
    diff_null = st.excess_return(prices_null).mean()
    diff_alpha = st.excess_return(prices_alpha).mean()
    assert diff_alpha > diff_null, "planted alpha should raise excess return mean"
    assert diff_alpha > 1e-5, "planted +3 bps/day alpha should be meaningfully positive"


def test_synthetic_correlation_is_high(null_prices):
    """NOBL and SPY should be highly correlated by construction (~0.97 target)."""
    from dividend_aristocrats import strategy as st
    prices, truth = null_prices
    ret = st.daily_returns(prices).dropna()
    corr = float(ret.corr().loc["NOBL", "SPY"])
    assert corr > 0.80, f"synthetic correlation too low: {corr:.3f}"


def test_fingerprint_stable_and_content_sensitive(null_prices):
    prices, _ = null_prices
    fp1 = data.fingerprint(prices)
    fp2 = data.fingerprint(prices)
    assert fp1 == fp2
    other, _ = data.synthetic_daily(n_years=10, alpha_bps=0.0, seed=99)
    assert fp1 != data.fingerprint(other)


def test_fetch_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_prices("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_load_aligned_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_aligned(["NOBL", "SPY"], fetch=False, cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Real-data tests — skipped when cache is absent
# ---------------------------------------------------------------------------
@requires_cache
def test_real_nobl_loaded():
    df = data.fetch_prices("NOBL", fetch=False, cache_dir=_CACHE_DIR)
    assert "close" in df.columns
    assert len(df) > 200
    assert (df["close"] > 0).all()


@requires_cache
def test_real_spy_loaded():
    df = data.fetch_prices("SPY", fetch=False, cache_dir=_CACHE_DIR)
    assert "close" in df.columns
    assert len(df) > 200


@requires_cache
def test_real_aligned_no_nans():
    prices = data.load_aligned(fetch=False, cache_dir=_CACHE_DIR)
    assert set(prices.columns) == {"NOBL", "SPY"}
    assert prices.isna().sum().sum() == 0

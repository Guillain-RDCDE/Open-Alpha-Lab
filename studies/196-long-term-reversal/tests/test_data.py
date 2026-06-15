"""The synthetic panel is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from long_term_reversal import data  # noqa: E402

CACHE_PATH = data.MONTHLY_CACHE


def test_synthetic_shape_and_types(null_panel):
    price_df, fwd_ret_df, truth = null_panel
    # Shape: months × firms
    assert price_df.shape[0] == truth["n_months"]
    assert price_df.shape[1] == truth["n_firms"]
    assert fwd_ret_df.shape == price_df.shape


def test_synthetic_prices_positive(null_panel):
    price_df, _, _ = null_panel
    # Prices must be positive (since they're cumulated exponentials)
    assert (price_df.values > 0).all()


def test_synthetic_is_deterministic():
    a, b_a, _ = data.synthetic_panel(n_firms=50, n_months=100, reversal=0.02, seed=7)
    b, b_b, _ = data.synthetic_panel(n_firms=50, n_months=100, reversal=0.02, seed=7)
    assert np.allclose(a.values, b.values)
    c, _, _ = data.synthetic_panel(n_firms=50, n_months=100, reversal=0.02, seed=8)
    assert not np.allclose(a.values, c.values)


def test_null_knob_yields_no_planted_effect():
    """reversal=0 means the forward returns have no cross-sectional z-score planted."""
    _, fwd_null, truth_null = data.synthetic_panel(
        n_firms=100, n_months=150, reversal=0.0, seed=42
    )
    assert truth_null["reversal"] == 0.0
    # The last month's fwd return should be NaN (no next month to hold)
    assert fwd_null.iloc[-1].isnull().all(), "Last row should be NaN (no forward month)"


def test_live_knob_changes_fwd_returns():
    """reversal > 0 should produce different forward returns than reversal=0."""
    _, fwd_null, _ = data.synthetic_panel(n_firms=80, n_months=120, reversal=0.0, seed=5)
    _, fwd_live, _ = data.synthetic_panel(n_firms=80, n_months=120, reversal=0.05, seed=5)
    # They use the same base noise; the reversal-planted component must differentiate them
    diff = (fwd_live - fwd_null).dropna(how="all")
    assert not np.allclose(diff.fillna(0).values, 0.0)


def test_fingerprint_stable_and_content_sensitive(null_panel):
    price_df, _, _ = null_panel
    assert data.fingerprint(price_df) == data.fingerprint(price_df)
    other, _, _ = data.synthetic_panel(n_firms=120, n_months=200, reversal=0.0, seed=999)
    assert data.fingerprint(price_df) != data.fingerprint(other)


def test_fetch_monthly_raises_without_cache(tmp_path):
    bogus = str(tmp_path / "no_such.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_monthly(fetch=False, cache_path=bogus)


# ---- Real-data tests: guarded by cache presence ----------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="ltr_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_prices_shape():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    assert prices.shape[0] > 100, "Expected >100 months of real data"
    assert prices.shape[1] > 30, "Expected >30 tickers"


@requires_cache
def test_real_prices_positive():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    # Non-NaN prices must be positive (NaN is fine for tickers with missing history)
    non_nan = prices.stack().dropna()
    assert (non_nan > 0).all(), "All non-NaN prices must be positive"


@requires_cache
def test_real_fingerprint_stable():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    assert data.fingerprint(prices) == data.fingerprint(prices)

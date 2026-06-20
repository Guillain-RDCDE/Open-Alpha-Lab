"""The synthetic panel is well-formed and deterministic; the cache-first loader is safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from one_month_reversal import data  # noqa: E402


def test_synthetic_shape_and_types(null_panel):
    price_df, fwd_ret_df, truth = null_panel
    assert price_df.shape[0] == truth["n_months"]
    assert price_df.shape[1] == truth["n_firms"]
    assert fwd_ret_df.shape == price_df.shape


def test_synthetic_prices_positive(null_panel):
    price_df, _, _ = null_panel
    assert (price_df.values > 0).all()


def test_synthetic_index_is_monthly_and_safe(null_panel):
    """The decorative label must be timestamps with a sane span (no ns overflow)."""
    price_df, _, _ = null_panel
    idx = price_df.index
    assert idx.is_monotonic_increasing
    span_years = (idx[-1] - idx[0]).days / 365.25
    assert span_years < 290, "synthetic span must stay well under the ns-timestamp horizon"


def test_synthetic_is_deterministic():
    a, _, _ = data.synthetic_panel(n_firms=50, n_months=100, reversal=0.03, seed=7)
    b, _, _ = data.synthetic_panel(n_firms=50, n_months=100, reversal=0.03, seed=7)
    assert np.allclose(a.values, b.values)
    c, _, _ = data.synthetic_panel(n_firms=50, n_months=100, reversal=0.03, seed=8)
    assert not np.allclose(a.values, c.values)


def test_null_knob_yields_no_planted_effect():
    _, fwd_null, truth = data.synthetic_panel(n_firms=100, n_months=150, reversal=0.0, seed=42)
    assert truth["reversal"] == 0.0
    assert fwd_null.iloc[-1].isnull().all(), "Last row should be NaN (no forward month)"


def test_live_knob_changes_fwd_returns():
    _, fwd_null, _ = data.synthetic_panel(n_firms=80, n_months=120, reversal=0.0, seed=5)
    _, fwd_live, _ = data.synthetic_panel(n_firms=80, n_months=120, reversal=0.05, seed=5)
    diff = (fwd_live - fwd_null).dropna(how="all")
    assert not np.allclose(diff.fillna(0).values, 0.0)


def test_fingerprint_stable_and_content_sensitive(null_panel):
    price_df, _, _ = null_panel
    assert data.fingerprint(price_df) == data.fingerprint(price_df)
    other, _, _ = data.synthetic_panel(n_firms=120, n_months=200, reversal=0.0, seed=999)
    assert data.fingerprint(price_df) != data.fingerprint(other)


def test_load_real_raises_without_any_cache(tmp_path, monkeypatch):
    """With every candidate cache hidden, load_real raises (offline CI is covered by synth)."""
    bogus = str(tmp_path / "no_such.parquet")
    monkeypatch.setattr(data, "MONTHLY_CACHE", bogus)
    monkeypatch.setattr(data, "SHARED_LTR_CACHE", str(tmp_path / "also_missing.parquet"))
    with pytest.raises(FileNotFoundError):
        data.load_real(cache_path=bogus)


# ---- Real-data tests: guarded by cache presence ----------------------------
def _any_cache() -> str | None:
    for p in (data.MONTHLY_CACHE, data.SHARED_LTR_CACHE):
        if os.path.exists(p):
            return p
    return None


requires_cache = pytest.mark.skipif(
    _any_cache() is None,
    reason="no monthly panel cache (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_prices_shape():
    prices = data.load_real()
    assert prices.shape[0] > 100, "Expected >100 months of real data"
    assert prices.shape[1] > 30, "Expected >30 tickers"


@requires_cache
def test_real_prices_positive():
    prices = data.load_real()
    non_nan = prices.stack().dropna()
    assert (non_nan > 0).all(), "All non-NaN prices must be positive"


@requires_cache
def test_real_no_partial_last_month():
    """The pinned run drops the partial month: last index <= AS_OF_LAST_MONTH."""
    import pandas as pd

    prices = data.load_real()
    assert prices.index[-1] <= pd.Timestamp(data.AS_OF_LAST_MONTH)

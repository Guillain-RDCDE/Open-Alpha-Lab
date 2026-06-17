"""Tests for the data layer (synthetic panel, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from same_month_seasonality import data  # noqa: E402

CACHE_PATH = data.MONTHLY_CACHE


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_shape():
    price_df, truth = data.synthetic_panel(n_firms=50, n_months=120, premium=0.01, seed=1)
    assert price_df.shape == (120, 50)
    assert truth["n_firms"] == 50
    assert truth["n_months"] == 120


def test_synthetic_panel_prices_positive():
    price_df, _ = data.synthetic_panel(n_firms=50, n_months=120, premium=0.01, seed=2)
    assert (price_df > 0).all().all()


def test_synthetic_panel_deterministic():
    a, _ = data.synthetic_panel(n_firms=50, n_months=120, premium=0.01, seed=42)
    b, _ = data.synthetic_panel(n_firms=50, n_months=120, premium=0.01, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_panel_null_zero_premium():
    """With premium=0, the planted alphas are all zero."""
    price_df, truth = data.synthetic_panel(n_firms=50, n_months=120, premium=0.0, seed=10)
    assert truth["premium"] == 0.0
    assert price_df.shape[0] == 120


def test_synthetic_panel_month_end_index():
    price_df, _ = data.synthetic_panel(n_firms=30, n_months=60, premium=0.0, seed=5)
    # All index dates should be month-end
    for ts in price_df.index:
        expected_end = ts + pd.offsets.MonthEnd(0)
        assert ts == expected_end, f"{ts} is not a month-end date"


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    price_df, _ = data.synthetic_panel(n_firms=30, n_months=60, premium=0.01, seed=7)
    fp = data.fingerprint(price_df)
    assert isinstance(fp, str)
    assert len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_panel(n_firms=30, n_months=60, premium=0.01, seed=7)
    b, _ = data.synthetic_panel(n_firms=30, n_months=60, premium=0.05, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="sms_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_panel_shape():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    assert prices.shape[0] > 100, "Expected at least 100 monthly rows"
    assert prices.shape[1] > 20, "Expected at least 20 tickers"


@requires_cache
def test_real_panel_month_end_index():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    for ts in prices.index[:5]:
        expected_end = ts + pd.offsets.MonthEnd(0)
        assert ts == expected_end

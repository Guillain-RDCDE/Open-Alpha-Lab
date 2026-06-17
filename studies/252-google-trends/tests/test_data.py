"""Tests for the data layer (curated Trends table, synthetic panel, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from google_trends import data  # noqa: E402

CACHE_PATH = data.MONTHLY_CACHE


# ---------------------------------------------------------------------------
# Curated Trends table
# ---------------------------------------------------------------------------
def test_trends_table_shape():
    tr = data.trends_table()
    assert tr.shape[0] == data._TRENDS_N_MONTHS
    assert tr.shape[1] == len(data._TREND_PARAMS)


def test_trends_table_scale_0_100():
    tr = data.trends_table()
    assert tr.min().min() >= 0.0
    assert tr.max().max() <= 100.0


def test_trends_table_rebased_to_100():
    """Each column is re-based so its own peak month = 100 (Trends convention)."""
    tr = data.trends_table()
    assert np.allclose(tr.max(axis=0).to_numpy(), 100.0)


def test_trends_table_deterministic():
    a = data.trends_table()
    b = data.trends_table()
    pd.testing.assert_frame_equal(a, b)


def test_trends_table_month_end_index():
    tr = data.trends_table()
    for ts in tr.index:
        assert ts == ts + pd.offsets.MonthEnd(0)


def test_trends_table_event_spikes_present():
    """GME spikes in Jan-2021 (squeeze); ZM spikes in Mar-2020 (lockdown)."""
    tr = data.trends_table()
    gme_jan21 = tr.loc[pd.Timestamp("2021-01-31"), "GME"]
    gme_base = tr.loc[pd.Timestamp("2019-06-30"), "GME"]
    assert gme_jan21 > gme_base, "GME should spike in Jan-2021"
    zm_mar20 = tr.loc[pd.Timestamp("2020-03-31"), "ZM"]
    zm_base = tr.loc[pd.Timestamp("2019-06-30"), "ZM"]
    assert zm_mar20 > zm_base, "ZM should spike in Mar-2020"


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_shape():
    attn, price, truth = data.synthetic_panel(n_firms=20, n_months=60, premium=0.02, seed=1)
    assert attn.shape == (60, 20)
    assert price.shape == (60, 20)
    assert truth["n_firms"] == 20
    assert truth["n_months"] == 60


def test_synthetic_panel_prices_positive():
    _, price, _ = data.synthetic_panel(n_firms=20, n_months=60, premium=0.02, seed=2)
    assert (price > 0).all().all()


def test_synthetic_panel_attn_in_band():
    attn, _, _ = data.synthetic_panel(n_firms=20, n_months=60, premium=0.02, seed=3)
    assert attn.min().min() >= 0.0
    assert attn.max().max() <= 100.0


def test_synthetic_panel_deterministic():
    a1, p1, _ = data.synthetic_panel(n_firms=20, n_months=60, premium=0.02, seed=42)
    a2, p2, _ = data.synthetic_panel(n_firms=20, n_months=60, premium=0.02, seed=42)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(p1, p2)


def test_synthetic_panel_null_zero_premium():
    attn, price, truth = data.synthetic_panel(n_firms=20, n_months=60, premium=0.0, seed=10)
    assert truth["premium"] == 0.0
    assert price.shape[0] == 60


def test_synthetic_panel_month_end_index():
    attn, price, _ = data.synthetic_panel(n_firms=10, n_months=36, premium=0.0, seed=5)
    for ts in price.index:
        assert ts == ts + pd.offsets.MonthEnd(0)


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    _, price, _ = data.synthetic_panel(n_firms=10, n_months=36, premium=0.02, seed=7)
    fp = data.fingerprint(price)
    assert isinstance(fp, str)
    assert len(fp) == 12


def test_fingerprint_changes_with_data():
    _, p1, _ = data.synthetic_panel(n_firms=10, n_months=36, premium=0.02, seed=7)
    _, p2, _ = data.synthetic_panel(n_firms=10, n_months=36, premium=0.05, seed=7)
    assert data.fingerprint(p1) != data.fingerprint(p2)


# ---------------------------------------------------------------------------
# Cache guard: offline default must NOT touch the network
# ---------------------------------------------------------------------------
def test_fetch_monthly_cache_guard(tmp_path):
    missing = str(tmp_path / "does_not_exist.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_monthly(fetch=False, cache_path=missing)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="gt_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_panel_shape():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    assert prices.shape[0] > 24, "Expected at least 24 monthly rows"
    assert prices.shape[1] > 10, "Expected at least 10 tickers"


@requires_cache
def test_real_panel_month_end_index():
    prices = data.fetch_monthly(fetch=False, cache_path=CACHE_PATH)
    for ts in prices.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)

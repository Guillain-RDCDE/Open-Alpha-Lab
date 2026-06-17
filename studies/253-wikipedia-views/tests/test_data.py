"""Tests for the data layer (curated anchors, synthetic panel, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wikipedia_views import data  # noqa: E402

CACHE_PATH = data.RETURNS_CACHE


# ---------------------------------------------------------------------------
# Curated anchor table
# ---------------------------------------------------------------------------
def test_anchor_table_columns():
    df = data.wiki_anchor_table()
    assert set(["ticker", "article", "base_kviews", "att_beta"]).issubset(df.columns)
    assert len(df) >= 20, "Expected at least 20 curated names"


def test_anchor_table_positive_levels():
    df = data.wiki_anchor_table()
    assert (df["base_kviews"] > 0).all()
    assert (df["att_beta"] > 0).all()


def test_anchor_tickers_unique():
    df = data.wiki_anchor_table()
    assert df["ticker"].is_unique


# ---------------------------------------------------------------------------
# Curated -> monthly panel
# ---------------------------------------------------------------------------
def test_wiki_views_panel_shape():
    panel = data.wiki_views_panel(n_months=60)
    assert panel.shape == (60, len(data.BASKET_TICKERS))


def test_wiki_views_panel_positive():
    panel = data.wiki_views_panel(n_months=48)
    assert (panel > 0).all().all()


def test_wiki_views_panel_deterministic():
    a = data.wiki_views_panel(n_months=48, seed=1)
    b = data.wiki_views_panel(n_months=48, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_wiki_views_panel_month_end_index():
    panel = data.wiki_views_panel(n_months=24)
    for ts in panel.index:
        assert ts == ts + pd.offsets.MonthEnd(0), f"{ts} not month-end"


# ---------------------------------------------------------------------------
# Synthetic attention panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_shapes():
    price_df, views_df, truth = data.synthetic_attention_panel(
        n_firms=20, n_months=90, premium=0.01, seed=7
    )
    assert price_df.shape == (90, 20)
    assert views_df.shape == (90, 20)
    assert truth["n_firms"] == 20


def test_synthetic_panel_prices_positive():
    price_df, _, _ = data.synthetic_attention_panel(n_firms=20, n_months=90, premium=0.01, seed=2)
    assert (price_df > 0).all().all()


def test_synthetic_panel_deterministic():
    a, av, _ = data.synthetic_attention_panel(n_firms=20, n_months=90, premium=0.01, seed=42)
    b, bv, _ = data.synthetic_attention_panel(n_firms=20, n_months=90, premium=0.01, seed=42)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(av, bv)


def test_synthetic_panel_null_premium():
    _, _, truth = data.synthetic_attention_panel(n_firms=20, n_months=90, premium=0.0, seed=10)
    assert truth["premium"] == 0.0


# ---------------------------------------------------------------------------
# Best-effort offline real tape + align
# ---------------------------------------------------------------------------
def test_best_effort_real_tape_aligned():
    views, prices = data.best_effort_real_tape(n_months=96)
    assert views.shape == prices.shape
    assert list(views.columns) == list(prices.columns)
    assert (prices > 0).all().all()


def test_align_panels_intersection():
    views = data.wiki_views_panel(n_months=60)
    _, prices = data.best_effort_real_tape(n_months=72)
    v, p = data.align_panels(views, prices)
    assert v.shape == p.shape
    assert (v.index == p.index).all()


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    panel = data.wiki_views_panel(n_months=36)
    fp = data.fingerprint(panel)
    assert isinstance(fp, str)
    assert len(fp) == 12


def test_fingerprint_changes_with_data():
    a = data.wiki_views_panel(n_months=36, seed=1)
    b = data.wiki_views_panel(n_months=36, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Real-data cache guard
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="wiki_returns_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


def test_fetch_returns_cache_guard():
    """With no cache and fetch=False, fetch_returns must raise (no network)."""
    if os.path.exists(CACHE_PATH):
        pytest.skip("cache present")
    with pytest.raises(FileNotFoundError):
        data.fetch_returns(fetch=False, cache_path=CACHE_PATH)


@requires_cache
def test_real_returns_shape():
    prices = data.fetch_returns(fetch=False, cache_path=CACHE_PATH)
    assert prices.shape[0] > 24
    assert prices.shape[1] > 5


@requires_cache
def test_real_returns_month_end_index():
    prices = data.fetch_returns(fetch=False, cache_path=CACHE_PATH)
    for ts in prices.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)

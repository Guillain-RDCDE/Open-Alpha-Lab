"""Tests for the data layer (mention table, synthetic panel, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from wsb_mentions import data  # noqa: E402

CACHE_PATH = data.MONTHLY_CACHE


# ---------------------------------------------------------------------------
# Hardcoded mention table
# ---------------------------------------------------------------------------
def test_mention_table_shape():
    mc = data.mention_counts()
    assert mc.shape == (36, len(data.MEME_TICKERS))
    assert list(mc.columns) == data.MEME_TICKERS


def test_mention_table_month_end_index():
    mc = data.mention_counts()
    for ts in mc.index:
        assert ts == ts + pd.offsets.MonthEnd(0), f"{ts} is not month-end"
    assert mc.index[0] == pd.Timestamp("2021-01-31")
    assert mc.index[-1] == pd.Timestamp("2023-12-31")


def test_mention_table_non_negative():
    mc = data.mention_counts()
    assert (mc.to_numpy() >= 0).all(), "mention counts must be non-negative"


def test_mention_table_gme_peak_january_2021():
    """The canonical squeeze: GME buzz peaks in the first month."""
    mc = data.mention_counts()
    assert mc["GME"].idxmax() == pd.Timestamp("2021-01-31")


def test_mention_table_bbby_collapse():
    """BBBY mentions go to zero after the 2023 bankruptcy."""
    mc = data.mention_counts()
    assert mc.loc[pd.Timestamp("2023-12-31"), "BBBY"] == 0.0
    assert mc.loc[pd.Timestamp("2022-08-31"), "BBBY"] > 1000  # the Aug-2023 pump's predecessor spike


def test_mention_table_deterministic():
    a = data.mention_counts()
    b = data.mention_counts()
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_shapes():
    price_df, mention_df, truth = data.synthetic_panel(n_firms=14, n_months=60, premium=0.1, seed=1)
    assert price_df.shape == (60, 14)
    assert mention_df.shape == (60, 14)
    assert truth["n_firms"] == 14 and truth["n_months"] == 60


def test_synthetic_panel_prices_positive():
    price_df, _, _ = data.synthetic_panel(n_firms=14, n_months=120, premium=0.1, seed=2)
    assert (price_df > 0).all().all()


def test_synthetic_panel_mentions_positive():
    _, mention_df, _ = data.synthetic_panel(n_firms=14, n_months=120, premium=0.0, seed=3)
    assert (mention_df >= 0).all().all()


def test_synthetic_panel_deterministic():
    a_p, a_m, _ = data.synthetic_panel(n_firms=14, n_months=60, premium=0.1, seed=42)
    b_p, b_m, _ = data.synthetic_panel(n_firms=14, n_months=60, premium=0.1, seed=42)
    pd.testing.assert_frame_equal(a_p, b_p)
    pd.testing.assert_frame_equal(a_m, b_m)


def test_synthetic_panel_month_end_index():
    price_df, _, _ = data.synthetic_panel(n_firms=14, n_months=36, premium=0.0, seed=5)
    for ts in price_df.index:
        assert ts == ts + pd.offsets.MonthEnd(0)


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    mc = data.mention_counts()
    fp = data.fingerprint(mc)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _, _ = data.synthetic_panel(n_firms=14, n_months=60, premium=0.1, seed=7)
    b, _, _ = data.synthetic_panel(n_firms=14, n_months=60, premium=0.4, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Cache guard
# ---------------------------------------------------------------------------
def test_cache_guard_raises_when_absent(tmp_path):
    missing = str(tmp_path / "nope.parquet")
    with pytest.raises(FileNotFoundError):
        data.fetch_meme_monthly(fetch=False, cache_path=missing)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="meme_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_panel_shape():
    prices = data.fetch_meme_monthly(fetch=False, cache_path=CACHE_PATH)
    assert prices.shape[0] > 30, "Expected >30 monthly rows"
    assert prices.shape[1] >= 10, "Expected >=10 tickers"


@requires_cache
def test_real_panel_month_end_index():
    prices = data.fetch_meme_monthly(fetch=False, cache_path=CACHE_PATH)
    for ts in prices.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)

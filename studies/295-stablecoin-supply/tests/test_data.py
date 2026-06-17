"""Tests for the data layer (curated supply series, synthetic panel, cache guard)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stablecoin_supply import data  # noqa: E402

CACHE_PATH = data.BTC_CACHE


# ---------------------------------------------------------------------------
# Curated supply series
# ---------------------------------------------------------------------------
def test_supply_series_month_end():
    s = data.supply_series()
    assert isinstance(s, pd.Series)
    assert len(s) > 80
    for ts in s.index[:5]:
        assert ts == ts + pd.offsets.MonthEnd(0)


def test_supply_series_monotone_index():
    s = data.supply_series()
    assert s.index.is_monotonic_increasing


def test_supply_series_positive():
    s = data.supply_series()
    assert (s > 0).all()


def test_supply_series_captures_ust_collapse():
    """May-2022 UST/Luna collapse: supply must contract from its 2022 peak."""
    s = data.supply_series()
    peak = s["2022-03-31"]
    post = s["2022-12-31"]
    assert post < peak, "Stablecoin supply should contract after the 2022 peak"


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_shape():
    df, truth = data.synthetic_panel(n_months=120, premium=0.02, seed=1)
    assert df.shape == (120, 2)
    assert set(df.columns) == {"supply_bn", "btc_ret"}
    assert truth["n_months"] == 120


def test_synthetic_panel_supply_positive():
    df, _ = data.synthetic_panel(n_months=120, premium=0.02, seed=2)
    assert (df["supply_bn"] > 0).all()


def test_synthetic_panel_deterministic():
    a, _ = data.synthetic_panel(n_months=120, premium=0.02, seed=42)
    b, _ = data.synthetic_panel(n_months=120, premium=0.02, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_panel_month_end_index():
    df, _ = data.synthetic_panel(n_months=60, premium=0.0, seed=5)
    for ts in df.index:
        assert ts == ts + pd.offsets.MonthEnd(0)


def test_synthetic_panel_null_premium_recorded():
    df, truth = data.synthetic_panel(n_months=60, premium=0.0, seed=10)
    assert truth["premium"] == 0.0


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    df, _ = data.synthetic_panel(n_months=60, premium=0.01, seed=7)
    fp = data.fingerprint(df)
    assert isinstance(fp, str)
    assert len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_panel(n_months=60, premium=0.01, seed=7)
    b, _ = data.synthetic_panel(n_months=60, premium=0.05, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# build_real_panel using a synthetic stand-in BTC series (no network)
# ---------------------------------------------------------------------------
def test_build_real_panel_offline_join():
    """build_real_panel joins the curated supply with any BTC price Series."""
    supply = data.supply_series()
    # Fabricate a deterministic BTC price path over the supply window (no network)
    rng = np.random.default_rng(295)
    px = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.1, len(supply)))),
        index=supply.index,
        name="close",
    )
    panel = data.build_real_panel(px)
    assert set(panel.columns) == {"supply_bn", "btc_ret"}
    assert len(panel) > 50
    assert panel["supply_bn"].notna().all()
    assert panel["btc_ret"].notna().all()


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence
# ---------------------------------------------------------------------------
requires_cache = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH),
    reason="btc_monthly.parquet cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_btc_cache_loads():
    px = data.fetch_btc_monthly(fetch=False, cache_path=CACHE_PATH)
    assert isinstance(px, pd.Series)
    assert len(px) > 50
    assert (px > 0).all()


@requires_cache
def test_real_panel_shape():
    px = data.fetch_btc_monthly(fetch=False, cache_path=CACHE_PATH)
    panel = data.build_real_panel(px)
    assert len(panel) > 50
    assert set(panel.columns) == {"supply_bn", "btc_ret"}

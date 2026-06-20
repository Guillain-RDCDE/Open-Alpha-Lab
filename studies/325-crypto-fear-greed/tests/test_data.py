"""Tests for the data layer: synthetic BTC tape, the price-derived proxy gauge,
the cache-first real loader, and the fingerprint helper."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_fear_greed import data  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic BTC tape
# ---------------------------------------------------------------------------
def test_synthetic_btc_shape():
    df, truth = data.synthetic_btc(n_days=500, reversion=0.0, seed=1)
    assert df.shape == (500, 1)
    assert list(df.columns) == ["close"]
    assert truth["n_days"] == 500


def test_synthetic_btc_prices_positive():
    df, _ = data.synthetic_btc(n_days=400, reversion=0.3, seed=2)
    assert (df["close"] > 0).all()


def test_synthetic_btc_deterministic():
    a, _ = data.synthetic_btc(n_days=300, reversion=0.2, seed=42)
    b, _ = data.synthetic_btc(n_days=300, reversion=0.2, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_btc_null_recorded():
    _, truth = data.synthetic_btc(n_days=200, reversion=0.0, seed=10)
    assert truth["reversion"] == 0.0


def test_synthetic_btc_span_under_safe_limit():
    """A daily tape of 2500 days is far under the ~290yr ns-timestamp overflow limit."""
    df, _ = data.synthetic_btc(n_days=2500, reversion=0.0, seed=325)
    span_years = (df.index[-1] - df.index[0]).days / 365.25
    assert span_years < 50


# ---------------------------------------------------------------------------
# Proxy gauge
# ---------------------------------------------------------------------------
def test_proxy_gauge_range():
    df, _ = data.synthetic_btc(n_days=1500, reversion=0.0, seed=3)
    g = data.proxy_gauge(df["close"])
    g = g.dropna()
    assert (g >= 0).all() and (g <= 100).all(), "gauge must be in [0, 100]"
    assert len(g) > 500


def test_proxy_gauge_no_lookahead():
    """The gauge at row t must not change when *future* prices are appended."""
    df, _ = data.synthetic_btc(n_days=1200, reversion=0.0, seed=4)
    full = data.proxy_gauge(df["close"])
    cut = 900
    partial = data.proxy_gauge(df["close"].iloc[:cut])
    common = partial.dropna().index.intersection(full.dropna().index)
    common = common[common <= df.index[cut - 1]]
    assert len(common) > 100
    np.testing.assert_allclose(
        full.loc[common].to_numpy(), partial.loc[common].to_numpy(), atol=1e-9
    )


def test_proxy_gauge_deterministic():
    df, _ = data.synthetic_btc(n_days=800, reversion=0.0, seed=5)
    a = data.proxy_gauge(df["close"])
    b = data.proxy_gauge(df["close"])
    pd.testing.assert_series_equal(a, b)


def test_panel_from_close_columns():
    df, _ = data.synthetic_btc(n_days=1200, reversion=0.0, seed=6)
    panel = data.panel_from_close(df["close"])
    assert set(panel.columns) == {"close", "fng"}
    assert (panel["fng"] >= 0).all() and (panel["fng"] <= 100).all()
    assert panel["close"].notna().all()


# ---------------------------------------------------------------------------
# Real loader -- graceful, never raises offline
# ---------------------------------------------------------------------------
def test_load_real_offline_returns_none_or_series(tmp_path):
    """With an empty cache dir and fetch=False, load_real returns None (no crash)."""
    out = data.load_real(fetch=False, cache_dir=str(tmp_path))
    # If the shared repo cache happens to exist it returns a Series; otherwise None.
    assert out is None or isinstance(out, pd.Series)


def test_real_panel_offline_returns_none_or_frame(tmp_path):
    out = data.real_panel(fetch=False, cache_dir=str(tmp_path)) if False else data.real_panel(fetch=False)
    assert out is None or isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length():
    df, _ = data.synthetic_btc(n_days=300, reversion=0.0, seed=7)
    fp = data.fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_btc(n_days=300, reversion=0.0, seed=7)
    b, _ = data.synthetic_btc(n_days=300, reversion=0.5, seed=7)
    assert data.fingerprint(a) != data.fingerprint(b)


# ---------------------------------------------------------------------------
# Real-data tests: guarded by cache presence (skip offline CI)
# ---------------------------------------------------------------------------
_LOCAL_BTC = data._local_cache_path()
requires_cache = pytest.mark.skipif(
    not (os.path.exists(data.BTC_CACHE) or os.path.exists(_LOCAL_BTC)),
    reason="BTC cache absent (offline CI); covered by synthetic tests",
)


@requires_cache
def test_real_panel_shape():
    panel = data.real_panel(fetch=False)
    assert panel is not None
    assert panel.shape[0] > 1000
    assert (panel["fng"] >= 0).all() and (panel["fng"] <= 100).all()

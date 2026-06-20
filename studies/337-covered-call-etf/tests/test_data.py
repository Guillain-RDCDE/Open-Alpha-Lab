"""The synthetic buy-write tape is well-formed and deterministic; the real loaders are
cache-safe. Synthetic tests never skip; the one real-cache test is gated for offline CI."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from covered_call_etf import data  # noqa: E402


def test_synthetic_shape_and_columns(capped):
    panel, truth = capped
    assert len(panel) == truth["n_years"] * 12
    assert list(panel.columns) == ["index_tr", "bw_tr", "bw_price", "bw_dist"]
    assert panel.index.is_monotonic_increasing


def test_synthetic_total_is_price_plus_dist(capped):
    """The defining identity: buy-write total return = capped price + distribution."""
    panel, _ = capped
    assert np.allclose(panel["bw_tr"], panel["bw_price"] + panel["bw_dist"])


def test_synthetic_distribution_is_the_planted_premium(capped):
    panel, truth = capped
    assert np.allclose(panel["bw_dist"], truth["premium"])
    assert truth["ann_dist_yield"] > 0.05  # a fat marketed "yield"


def test_cap_shaves_only_up_moves(capped):
    """Capping is one-sided: down months pass through, up months are clipped at the strike."""
    panel, truth = capped
    g_index = np.log1p(panel["index_tr"])
    g_price = np.log1p(panel["bw_price"])
    up = g_index > truth["cap"]
    down = g_index <= 0
    # capped up-months are pinned at the strike; down-months untouched
    assert np.allclose(g_price[up], truth["cap"])
    assert np.allclose(g_price[down], g_index[down])


def test_null_world_is_the_index(null):
    """With no cap and no premium the buy-write is identical to the index."""
    panel, _ = null
    assert np.allclose(panel["bw_tr"], panel["index_tr"])
    assert np.allclose(panel["bw_price"], panel["index_tr"])
    assert (panel["bw_dist"] == 0.0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_buywrite(cap=0.02, premium=0.006, seed=5)
    b, _ = data.synthetic_buywrite(cap=0.02, premium=0.006, seed=5)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_buywrite(cap=0.02, premium=0.006, seed=6)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_no_timestamp_overflow_on_long_span():
    """The decorative index uses period_range, so even a long span never overflows ns."""
    panel, _ = data.synthetic_buywrite(n_years=100, seed=1)
    assert len(panel) == 1200
    assert panel.index.is_monotonic_increasing


def test_fingerprint_stable_and_content_sensitive(capped):
    panel, _ = capped
    assert data.fingerprint(panel) == data.fingerprint(panel)
    other, _ = data.synthetic_buywrite(cap=0.02, premium=0.006, seed=99)
    assert data.fingerprint(panel) != data.fingerprint(other)


def test_fetch_panel_cache_only_returns_empty_without_cache(tmp_path):
    """No cache + no network → an empty frame, so callers fall back gracefully (no crash)."""
    out = data.fetch_panel(cache_dir=str(tmp_path), fetch=False)
    assert out.empty


# --- the only test that reads the shared real cache: gated for offline CI ---
_PANEL = data._panel_path(data.DEFAULT_CACHE)


@pytest.mark.skipif(not os.path.exists(_PANEL), reason="offline CI / no real cache")
def test_real_panel_has_expected_tickers():
    panel = data.fetch_panel()
    assert not panel.empty
    assert "SPY" in panel.columns
    assert any(t in panel.columns for t in ("JEPI", "QYLD"))
    assert isinstance(panel.index, pd.DatetimeIndex)

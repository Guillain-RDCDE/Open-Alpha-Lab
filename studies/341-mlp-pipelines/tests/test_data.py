"""The synthetic MLP tape is well-formed and deterministic; the real loaders are cache-safe.
Synthetic tests never skip; the one real-cache test is gated for offline CI."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp_pipelines import data  # noqa: E402


def test_synthetic_shape_and_columns(trap):
    panel, truth = trap
    assert len(panel) == truth["n_years"] * 12
    assert list(panel.columns) == ["energy_tr", "mlp_tr", "mlp_price", "mlp_dist"]
    assert panel.index.is_monotonic_increasing


def test_synthetic_total_is_price_plus_dist(trap):
    """The defining identity: MLP total return = price (NAV) + distribution."""
    panel, _ = trap
    assert np.allclose(panel["mlp_tr"], panel["mlp_price"] + panel["mlp_dist"])


def test_synthetic_distribution_is_the_planted_payout(trap):
    panel, truth = trap
    assert np.allclose(panel["mlp_dist"], truth["dist"])
    assert truth["ann_dist_yield"] > 0.05  # a fat marketed "yield"


def test_null_world_is_flat_and_no_payout(null):
    """With beta 0, no distribution, no drift the fund is flat and market-neutral."""
    panel, _ = null
    assert np.allclose(panel["mlp_price"], 0.0)
    assert np.allclose(panel["mlp_tr"], 0.0)
    assert (panel["mlp_dist"] == 0.0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_mlp(beta=1.1, dist=0.006, seed=5)
    b, _ = data.synthetic_mlp(beta=1.1, dist=0.006, seed=5)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_mlp(beta=1.1, dist=0.006, seed=6)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_no_timestamp_overflow_on_long_span():
    """The decorative index uses period_range, so even a long span never overflows ns."""
    panel, _ = data.synthetic_mlp(n_years=100, seed=1)
    assert len(panel) == 1200
    assert panel.index.is_monotonic_increasing


def test_fingerprint_stable_and_content_sensitive(trap):
    panel, _ = trap
    assert data.fingerprint(panel) == data.fingerprint(panel)
    other, _ = data.synthetic_mlp(beta=1.1, dist=0.006, seed=99)
    assert data.fingerprint(panel) != data.fingerprint(other)


def test_fetch_panel_cache_only_returns_empty_without_cache(tmp_path):
    """No cache + no network → an empty frame, so callers fall back gracefully (no crash)."""
    out = data.fetch_panel(cache_dir=str(tmp_path), fetch=False)
    assert out.empty


def test_fetch_price_and_dist_cache_only_returns_empty_without_cache(tmp_path):
    out = data.fetch_price_and_dist(cache_dir=str(tmp_path), fetch=False)
    assert out.empty


# --- the only test that reads the shared real cache: gated for offline CI ---
_PANEL = data._panel_path(data.DEFAULT_CACHE)


@pytest.mark.skipif(not os.path.exists(_PANEL), reason="offline CI / no real cache")
def test_real_panel_has_expected_tickers():
    panel = data.fetch_panel()
    assert not panel.empty
    assert "SPY" in panel.columns
    assert any(t in panel.columns for t in ("AMLP", "XLE"))
    assert isinstance(panel.index, pd.DatetimeIndex)

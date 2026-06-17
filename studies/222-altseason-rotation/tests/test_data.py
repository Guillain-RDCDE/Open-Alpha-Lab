"""Tests for data.py -- synthetic tape, dominance computation, cache-gating."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altseason_rotation import data  # noqa: E402

CACHE_PATH = data._cache_path(data.DEFAULT_CACHE)


# ---- synthetic tape -------------------------------------------------------
def test_synthetic_shape(null_panel):
    panel, truth = null_panel
    assert "BTC-USD" in panel.columns
    for t in data.ALT_TICKERS:
        assert t in panel.columns
    assert len(panel) == truth["n_days"]
    assert (panel > 0).all().all()


def test_synthetic_deterministic():
    p1, _ = data.synthetic_daily(n_years=3, dom_signal=0.0, seed=222)
    p2, _ = data.synthetic_daily(n_years=3, dom_signal=0.0, seed=222)
    assert (p1.values == p2.values).all()


def test_synthetic_signal_differs_from_null():
    null, _ = data.synthetic_daily(n_years=4, dom_signal=0.0, seed=222)
    sig, _ = data.synthetic_daily(n_years=4, dom_signal=2.0, seed=222)
    # BTC should be identical (no signal injected into BTC)
    assert np.allclose(null["BTC-USD"].values, sig["BTC-USD"].values)
    # Alts should differ (signal is injected into alt returns)
    assert not np.allclose(null["ETH-USD"].values, sig["ETH-USD"].values)


# ---- dominance computation ------------------------------------------------
def test_dominance_bounded(null_panel):
    panel, _ = null_panel
    dom = data.compute_dominance(panel)
    assert (dom > 0).all() and (dom < 1).all()


def test_dominance_sum(null_panel):
    """BTC share is between 0 and 1 and sums to less than 1."""
    panel, _ = null_panel
    dom = data.compute_dominance(panel)
    assert dom.min() > 0
    assert dom.max() < 1.0


# ---- fingerprint ----------------------------------------------------------
def test_fingerprint_consistent(null_panel):
    panel, _ = null_panel
    fp1 = data.fingerprint(panel)
    fp2 = data.fingerprint(panel)
    assert fp1 == fp2
    assert len(fp1) == 12


# ---- cache gating ---------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_fetch_panel_cache_reads_correctly():
    panel = data.fetch_panel(fetch=False)
    assert "BTC-USD" in panel.columns
    assert len(panel) > 100
    assert panel.index.tz is None

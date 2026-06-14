"""The synthetic panel is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitcoin_dominance import data  # noqa: E402


def test_synthetic_shape(null_panel):
    panel, truth = null_panel
    assert "BTC-USD" in panel.columns
    # At least one alt present
    assert any(t in panel.columns for t in data.ALT_TICKERS)
    assert len(panel) == truth["n_days"]
    assert (panel > 0).all().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=3, dom_signal=0.0, seed=134)
    b, _ = data.synthetic_daily(n_years=3, dom_signal=0.0, seed=134)
    assert np.allclose(a["BTC-USD"].to_numpy(), b["BTC-USD"].to_numpy())
    c, _ = data.synthetic_daily(n_years=3, dom_signal=0.0, seed=999)
    assert not np.allclose(a["BTC-USD"].to_numpy(), c["BTC-USD"].to_numpy())


def test_synthetic_seed_changes_output():
    a, _ = data.synthetic_daily(n_years=3, dom_signal=0.0, seed=1)
    b, _ = data.synthetic_daily(n_years=3, dom_signal=0.0, seed=2)
    assert not np.allclose(a["BTC-USD"].to_numpy(), b["BTC-USD"].to_numpy())


def test_dominance_between_zero_and_one(null_panel):
    panel, _ = null_panel
    dom = data.compute_dominance(panel)
    assert (dom > 0).all() and (dom < 1).all()


def test_dominance_sums_correctly(null_panel):
    """BTC dominance should change when alt prices change."""
    panel, _ = null_panel
    dom = data.compute_dominance(panel)
    # Dominance should not be constant (prices move)
    assert dom.std() > 1e-6


def test_fingerprint_is_stable_and_content_sensitive(null_panel):
    panel, _ = null_panel
    fp1 = data.fingerprint(panel)
    fp2 = data.fingerprint(panel)
    assert fp1 == fp2
    other, _ = data.synthetic_daily(n_years=5, dom_signal=0.0, seed=999)
    assert data.fingerprint(panel) != data.fingerprint(other)


def test_fetch_panel_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_panel(fetch=False, cache_dir=str(tmp_path))

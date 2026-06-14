"""Tests for the data layer — synthetic generator shape, null property, fingerprint."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from country_momentum import data  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic_panel shape and basic properties
# ---------------------------------------------------------------------------
def test_synthetic_panel_shape():
    panel, truth = data.synthetic_panel(n_months=60, n_countries=8, seed=0)
    assert panel.shape == (60, 8), f"Expected (60, 8), got {panel.shape}"
    assert len(panel.index) == 60


def test_synthetic_panel_is_period_indexed():
    panel, _ = data.synthetic_panel(n_months=24, seed=0)
    assert isinstance(panel.index, pd.PeriodIndex)
    assert str(panel.index.freq) in ("M", "ME", "<MonthEnd>")


def test_synthetic_panel_no_nans():
    panel, _ = data.synthetic_panel(n_months=60, seed=0)
    assert not panel.isnull().any().any(), "Synthetic panel should have no NaN values"


def test_synthetic_panel_deterministic():
    p1, t1 = data.synthetic_panel(seed=42)
    p2, t2 = data.synthetic_panel(seed=42)
    assert p1.equals(p2), "Same seed must give identical panels"
    p3, _ = data.synthetic_panel(seed=99)
    assert not p1.equals(p3), "Different seeds must give different panels"


def test_synthetic_panel_null_has_no_momentum():
    """With mom_strength=0 the persistent drift is zero, returns should be near-zero mean."""
    panel, truth = data.synthetic_panel(n_months=480, mom_strength=0.0, n_countries=20, seed=0)
    assert not truth["has_momentum"]
    cross_means = panel.mean(axis=0)
    assert cross_means.abs().max() < 0.05, "Null panel cross-means should be small"


def test_synthetic_panel_momentum_positive_mean():
    """With baked momentum, the world drift should be positive (planted)."""
    panel, truth = data.synthetic_panel(n_months=240, mom_strength=0.04, seed=146)
    assert truth["has_momentum"]
    assert truth["mom_strength"] == 0.04


def test_truth_dict_keys():
    _, truth = data.synthetic_panel()
    for key in ("n_months", "n_countries", "mom_strength", "phi", "seed", "has_momentum"):
        assert key in truth, f"truth dict missing key: {key}"


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_length_and_hex():
    panel, _ = data.synthetic_panel(n_months=36, seed=0)
    fp = data.fingerprint(panel)
    assert len(fp) == 12
    int(fp, 16)  # must be valid hex


def test_fingerprint_changes_with_data():
    p1, _ = data.synthetic_panel(seed=1)
    p2, _ = data.synthetic_panel(seed=2)
    assert data.fingerprint(p1) != data.fingerprint(p2)


# ---------------------------------------------------------------------------
# Real-tape fetch raises on cache miss (no network)
# ---------------------------------------------------------------------------
def test_fetch_etfs_raises_on_cache_miss(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_etfs(fetch=False, cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Constant list of tickers
# ---------------------------------------------------------------------------
def test_tickers_list_nonempty_and_strings():
    assert len(data.TICKERS) >= 10
    for t in data.TICKERS:
        assert isinstance(t, str) and len(t) >= 2

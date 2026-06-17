"""Tests for the data layer — synthetic generator shape, null property, fingerprint."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sector_rotation import data  # noqa: E402

STUDY_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
CACHE_PATH = os.path.join(STUDY_CACHE, "sector_rotation_monthly.parquet")


# ---------------------------------------------------------------------------
# synthetic_panel shape and basic properties
# ---------------------------------------------------------------------------
def test_synthetic_panel_shape():
    panel, truth = data.synthetic_panel(n_months=60, n_sectors=8, seed=0)
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
    p1, _ = data.synthetic_panel(seed=42)
    p2, _ = data.synthetic_panel(seed=42)
    assert p1.equals(p2), "Same seed must give identical panels"
    p3, _ = data.synthetic_panel(seed=99)
    assert not p1.equals(p3), "Different seeds must give different panels"


def test_synthetic_panel_null_has_no_momentum():
    """With mom_strength=0 the persistent drift is zero, cross-means should be small."""
    panel, truth = data.synthetic_panel(n_months=480, mom_strength=0.0, n_sectors=9, seed=0)
    assert not truth["has_momentum"]
    cross_means = panel.mean(axis=0)
    assert cross_means.abs().max() < 0.05, "Null panel cross-means should be small"


def test_synthetic_panel_momentum_flag():
    """With baked momentum, truth['has_momentum'] should be True."""
    panel, truth = data.synthetic_panel(n_months=240, mom_strength=0.04, seed=225)
    assert truth["has_momentum"]
    assert truth["mom_strength"] == 0.04


def test_truth_dict_keys():
    _, truth = data.synthetic_panel()
    for key in ("n_months", "n_sectors", "mom_strength", "phi", "seed", "has_momentum"):
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
# Real-tape fetch raises on cache miss (no network in CI)
# ---------------------------------------------------------------------------
def test_fetch_etfs_raises_on_cache_miss(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_etfs(fetch=False, cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Constant list of tickers
# ---------------------------------------------------------------------------
def test_tickers_list_nonempty_and_strings():
    assert len(data.TICKERS) >= 9
    for t in data.TICKERS:
        assert isinstance(t, str) and len(t) >= 2


def test_tickers_includes_known_sectors():
    """The 9 original SPDR sectors must always be present."""
    must_have = {"XLK", "XLV", "XLF", "XLY", "XLI", "XLP", "XLE", "XLU", "XLB"}
    assert must_have.issubset(set(data.TICKERS))


# ---------------------------------------------------------------------------
# Real-tape smoke test (skip if cache absent — CI / offline)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="real-tape cache absent offline/CI")
def test_real_tape_shape_and_index():
    panel, spy = data.fetch_etfs(fetch=False, cache_dir=STUDY_CACHE)
    assert isinstance(panel.index, pd.PeriodIndex)
    assert panel.shape[1] >= 9  # at least the original 9 sectors
    assert len(spy) == len(panel)
    assert not panel.isnull().all().any(), "No column should be all-NaN"

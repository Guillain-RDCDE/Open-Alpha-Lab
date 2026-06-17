"""The synthetic panel is well-formed, deterministic, and tunable."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ohlson_o_score import data  # noqa: E402


def test_synthetic_shape(has_premium):
    o, fwd, truth = has_premium
    assert o.shape == (truth["n_years"], truth["n_firms"])
    assert fwd.shape == o.shape


def test_synthetic_index_columns(has_premium):
    o, fwd, truth = has_premium
    assert o.index.name == "year"
    assert fwd.index.name == "year"
    assert list(o.index) == list(fwd.index)
    assert list(o.columns) == list(fwd.columns)


def test_synthetic_o_range(has_premium):
    """O-scores are finite; no infinite values in a realistic synthetic panel."""
    o, fwd, truth = has_premium
    assert np.isfinite(o.values).all()
    assert np.isfinite(fwd.values).all()


def test_synthetic_is_deterministic():
    o1, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=42)
    o2, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=42)
    assert np.allclose(o1.values, o2.values)
    o3, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=99)
    assert not np.allclose(o1.values, o3.values)


def test_synthetic_null_knob(no_premium):
    """With o_premium=0 the truth dict correctly flags no premium."""
    _, _, truth = no_premium
    assert truth["o_premium"] == 0.0
    assert not truth["has_premium"]


def test_synthetic_premium_knob(has_premium):
    """With o_premium>0 the truth dict flags a premium."""
    _, _, truth = has_premium
    assert truth["has_premium"]


def test_fingerprint_stable_and_content_sensitive(has_premium):
    o, _, _ = has_premium
    fp1 = data.fingerprint(o)
    fp2 = data.fingerprint(o)
    assert fp1 == fp2  # deterministic
    o2, _, _ = data.synthetic_panel(n_firms=200, n_years=20, seed=999)
    assert fp1 != data.fingerprint(o2)


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    """fetch_panel with a non-existent cache directory returns empty frames."""
    o, fwd = data.fetch_panel(cache_dir=str(tmp_path))
    assert o.empty
    assert fwd.empty

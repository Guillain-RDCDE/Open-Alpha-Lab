"""The synthetic panel is well-formed, deterministic, and carries the planted premium."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_flow_yield import data  # noqa: E402


def test_synthetic_shape(signal_panel):
    ocfy, ey, fwd, truth = signal_panel
    assert ocfy.shape == ey.shape == fwd.shape
    assert ocfy.shape[0] == 18  # n_years
    assert ocfy.shape[1] == 200  # n_firms


def test_synthetic_is_deterministic():
    a, _, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=5)
    b, _, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=5)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _, _, _ = data.synthetic_panel(n_firms=50, n_years=10, seed=6)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_synthetic_year_index(signal_panel):
    ocfy, _, _, _ = signal_panel
    assert ocfy.index.name == "year"
    assert ocfy.index[0] == 2000
    assert ocfy.index[-1] == 2017


def test_synthetic_ocfy_range(signal_panel):
    """OCF yields in the synthetic panel should be mostly positive (modeled as ~2-8%)."""
    ocfy, _, _, _ = signal_panel
    flat = ocfy.to_numpy().ravel()
    # Planted: 0.05 + 0.03*z — z is ~N(0,1) so most values ~2-8%.
    assert (flat > 0).mean() > 0.80, "Most synthetic OCF yields should be positive"


def test_truth_attributes(signal_panel, null_panel):
    _, _, _, truth_signal = signal_panel
    _, _, _, truth_null = null_panel
    assert truth_signal.has_ocfy_edge
    assert not truth_null.has_ocfy_edge
    assert not truth_null.has_ey_edge


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    """Without cached data, fetch_panel returns empty DataFrames rather than crashing."""
    ocfy, ey, fwd = data.fetch_panel(cache_dir=str(tmp_path), study_cache=str(tmp_path))
    assert ocfy.empty
    assert ey.empty
    assert fwd.empty


def test_fingerprint_is_stable_and_content_sensitive(signal_panel):
    ocfy, _, _, _ = signal_panel
    assert data.fingerprint(ocfy) == data.fingerprint(ocfy)
    other, _, _, _ = data.synthetic_panel(n_firms=200, n_years=18, seed=999)
    assert data.fingerprint(ocfy) != data.fingerprint(other)


def test_winsorise():
    """Values outside [lo, hi] become NaN; values inside are preserved."""
    import pandas as pd

    df = pd.DataFrame({"a": [0.0, 0.5, 1.5, -0.1], "b": [0.1, 0.9, 0.3, 0.0]})
    out = data._winsorise(df, lo=0.0, hi=1.0)
    assert np.isnan(out.loc[2, "a"])   # 1.5 > 1.0 → NaN
    assert np.isnan(out.loc[3, "a"])   # -0.1 < 0.0 → NaN
    assert out.loc[1, "a"] == 0.5      # inside → preserved

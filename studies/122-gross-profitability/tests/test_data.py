"""The synthetic panel is well-formed, deterministic, and carries the planted signal."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gross_profitability import data  # noqa: E402


def test_synthetic_shapes_match(null_panel):
    signal, fwd_ret, truth = null_panel
    assert signal.shape == fwd_ret.shape
    assert signal.index.name == "year"
    assert fwd_ret.index.name == "year"


def test_synthetic_gpa_is_positive(null_panel):
    """GP/A is simulated as a log-normal — should always be positive."""
    signal, _, _ = null_panel
    assert (signal.to_numpy(dtype=float) > 0).all()


def test_synthetic_is_deterministic():
    a, b_a, _ = data.synthetic_panel(n_firms=100, n_years=10, gpa_premium=0.05, seed=7)
    c, d_c, _ = data.synthetic_panel(n_firms=100, n_years=10, gpa_premium=0.05, seed=7)
    assert np.allclose(a.to_numpy(), c.to_numpy())
    e, _, _ = data.synthetic_panel(n_firms=100, n_years=10, gpa_premium=0.05, seed=8)
    assert not np.allclose(a.to_numpy(), e.to_numpy())


def test_seed_changes_output():
    a, _, _ = data.synthetic_panel(seed=1)
    b, _, _ = data.synthetic_panel(seed=2)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_null_truth_flag(null_panel):
    _, _, truth = null_panel
    assert not truth.has_premium
    assert truth.gpa_premium == 0.0


def test_live_truth_flag(live_panel):
    _, _, truth = live_panel
    assert truth.has_premium


def test_fetch_panel_returns_empty_without_cache(tmp_path):
    """Absent cache → graceful empty DataFrames, not a crash."""
    sig, fwd = data.fetch_panel(cache_dir=str(tmp_path))
    assert sig.empty
    assert fwd.empty


def test_fingerprint_stable_and_sensitive(null_panel):
    signal, _, _ = null_panel
    fp = data.fingerprint(signal)
    assert fp == data.fingerprint(signal)
    other, _, _ = data.synthetic_panel(n_firms=200, n_years=20, gpa_premium=0.0, seed=99)
    assert fp != data.fingerprint(other)

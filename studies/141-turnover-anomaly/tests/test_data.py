"""The synthetic panel is well-formed, deterministic, and cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from turnover_anomaly import data  # noqa: E402


def test_synthetic_panel_shape(null_panel):
    sig, fwd, truth = null_panel
    assert sig.shape == (truth["n_years"], truth["n_firms"])
    assert fwd.shape == (truth["n_years"], truth["n_firms"])
    assert sig.index.name == "year"
    assert fwd.index.name == "year"


def test_synthetic_panel_signal_is_positive(null_panel):
    """Turnover is a positive quantity — all synthetic values must be > 0."""
    sig, _, _ = null_panel
    assert (sig.to_numpy() > 0).all()


def test_synthetic_panel_is_deterministic():
    a_sig, a_fwd, _ = data.synthetic_panel(n_firms=50, n_years=8, premium=-0.05, seed=7)
    b_sig, b_fwd, _ = data.synthetic_panel(n_firms=50, n_years=8, premium=-0.05, seed=7)
    assert np.allclose(a_sig.to_numpy(), b_sig.to_numpy())
    assert np.allclose(a_fwd.to_numpy(), b_fwd.to_numpy())


def test_synthetic_panel_seed_changes_output():
    a_sig, _, _ = data.synthetic_panel(n_firms=50, n_years=8, premium=-0.05, seed=7)
    c_sig, _, _ = data.synthetic_panel(n_firms=50, n_years=8, premium=-0.05, seed=99)
    assert not np.allclose(a_sig.to_numpy(), c_sig.to_numpy())


def test_synthetic_null_and_signal_differ_in_returns():
    """With premium = -0.10 the fwd returns should differ from premium = 0.0."""
    _, fwd0, _ = data.synthetic_panel(n_firms=80, n_years=10, premium=0.0, seed=141)
    _, fwd1, _ = data.synthetic_panel(n_firms=80, n_years=10, premium=-0.10, seed=141)
    # They share the same noise draw so means should be similar, but arrays differ
    assert not np.allclose(fwd0.to_numpy(), fwd1.to_numpy())


def test_synthetic_truth_dict_contents(signal_panel):
    _, _, truth = signal_panel
    assert "premium" in truth
    assert "n_firms" in truth
    assert "n_years" in truth
    assert "has_effect" in truth
    assert truth["has_effect"] is True


def test_fetch_panel_offline_returns_empty(tmp_path):
    """Without any cache file, fetch_panel returns two empty DataFrames (no crash)."""
    sig, fwd = data.fetch_panel(cache_dir=str(tmp_path), study_cache=str(tmp_path), fetch=False)
    assert sig.empty
    assert fwd.empty


def test_fingerprint_stable_and_content_sensitive(null_panel):
    sig, _, _ = null_panel
    assert data.fingerprint(sig) == data.fingerprint(sig)
    other_sig, _, _ = data.synthetic_panel(n_firms=100, n_years=12, premium=0.0, seed=999)
    assert data.fingerprint(sig) != data.fingerprint(other_sig)

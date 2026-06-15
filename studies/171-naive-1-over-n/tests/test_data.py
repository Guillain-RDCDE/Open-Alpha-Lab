"""The synthetic tape is well-formed, deterministic, and the planted signal is present."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from naive_1_over_n import data  # noqa: E402


def test_synthetic_shape(null_returns):
    df, truth = null_returns
    assert df.shape == (truth["n_periods"], truth["n_assets"])
    assert list(df.index) == sorted(df.index)


def test_synthetic_deterministic():
    a, _ = data.synthetic_returns(n_periods=100, seed=10)
    b, _ = data.synthetic_returns(n_periods=100, seed=10)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_returns(n_periods=100, seed=11)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_synthetic_null_has_equal_means(null_returns):
    """With signal_strength=0, true mu is all zeros — sample means should be near-zero."""
    df, truth = null_returns
    assert truth["signal_strength"] == 0.0
    assert all(m == 0.0 for m in truth["true_mu"])
    # Sample means are random, but should be well within 3*vol/sqrt(n)
    expected_se = truth["vol"] / np.sqrt(truth["n_periods"])
    for col in df.columns:
        assert abs(df[col].mean()) < 6 * expected_se


def test_synthetic_signal_creates_return_gradient(signal_returns):
    """With signal_strength > 0, later assets should have higher true expected returns."""
    df, truth = signal_returns
    assert truth["signal_strength"] > 0.0
    true_mu = truth["true_mu"]
    # True mu is monotone increasing
    for i in range(len(true_mu) - 1):
        assert true_mu[i + 1] >= true_mu[i]
    # Last asset has higher true mu than first
    assert true_mu[-1] > true_mu[0]


def test_fingerprint_stable_and_sensitive(null_returns):
    df, _ = null_returns
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_returns(n_periods=504, signal_strength=0.0, seed=999)
    assert data.fingerprint(df) != data.fingerprint(other)


def test_load_real_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(fetch=False, cache_dir=str(tmp_path))

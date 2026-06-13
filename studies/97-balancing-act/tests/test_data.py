"""Data-layer invariants for Study 97 (Balancing-Act)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from balancing_act import data  # noqa: E402


def test_shape_and_columns(diversifying):
    frame, truth = diversifying
    assert list(frame.columns) == ["STK", "BND"]
    assert len(frame) == truth["n_days"] == 6000
    assert (frame > 0).all().all()


def test_determinism():
    a, _ = data.synthetic_two_asset(n_days=1000, corr=-0.3, seed=7)
    b, _ = data.synthetic_two_asset(n_days=1000, corr=-0.3, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_planted_correlation_negative(diversifying):
    _, truth = diversifying
    # The realised correlation tracks the planted one (sampling noise aside).
    assert truth["realised_corr"] < -0.2


def test_planted_correlation_positive(redundant):
    _, truth = redundant
    assert truth["realised_corr"] > 0.6


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_two_asset(n_days=500, corr=-0.3, seed=1)
    b, _ = data.synthetic_two_asset(n_days=500, corr=-0.3, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fingerprint_changes_with_columns():
    a, _ = data.synthetic_two_asset(n_days=500, corr=-0.3, seed=1)
    b = a.rename(columns={"STK": "SPY"})
    assert data.fingerprint(a) != data.fingerprint(b)

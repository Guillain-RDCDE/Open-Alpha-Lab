"""Data-layer invariants for Study 339 (Convertible-Bonds)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from convertible_bonds import data  # noqa: E402


def test_shape_and_columns(convex):
    frame, truth = convex
    assert list(frame.columns) == ["IDX", "BND", "CVT"]
    assert len(frame) == truth["n_days"] == 4000
    assert (frame > 0).all().all()


def test_determinism():
    a, _ = data.synthetic_convex(n_days=1000, gamma=0.5, seed=7)
    b, _ = data.synthetic_convex(n_days=1000, gamma=0.5, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_gamma_recorded(convex, linear):
    assert convex[1]["gamma"] == 0.6
    assert linear[1]["gamma"] == 0.0


def test_index_span_is_bounded():
    # No timestamp-overflow risk: 4000 bdays is ~16 years, well under the ns limit.
    frame, _ = data.synthetic_convex(n_days=4000, seed=1)
    span_years = (frame.index[-1] - frame.index[0]).days / 365.25
    assert span_years < 30


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_convex(n_days=500, gamma=0.5, seed=1)
    b, _ = data.synthetic_convex(n_days=500, gamma=0.5, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fingerprint_changes_with_columns():
    a, _ = data.synthetic_convex(n_days=500, gamma=0.5, seed=1)
    b = a.rename(columns={"CVT": "CWB"})
    assert data.fingerprint(a) != data.fingerprint(b)

"""Data-layer invariants for Study 91 (Death-Cross)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from death_cross import data  # noqa: E402


def test_shape_and_columns(null_walk):
    frame, truth = null_walk
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"] == 6000
    assert (frame["close"] > 0).all()


def test_determinism():
    a, _ = data.synthetic_daily(n_days=1000, regime=1.0, seed=7)
    b, _ = data.synthetic_daily(n_days=1000, regime=1.0, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_null_has_no_bear(null_walk):
    _, truth = null_walk
    assert truth["frac_bear"] == 0.0


def test_planted_has_a_bear_minority(planted_regimes):
    _, truth = planted_regimes
    # The bull is the home state — bears are a minority of days but not negligible.
    assert 0.05 < truth["frac_bear"] < 0.45


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_daily(n_days=500, regime=0.0, seed=1)
    b, _ = data.synthetic_daily(n_days=500, regime=0.0, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)

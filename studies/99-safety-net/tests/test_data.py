"""Data-layer invariants for Study 99 (Safety-Net)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from safety_net import data  # noqa: E402


def test_shape_and_columns(trending):
    frame, truth = trending
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"] == 6000
    assert (frame["close"] > 0).all()


def test_meanrev_shape(meanrev):
    frame, truth = meanrev
    assert list(frame.columns) == ["close"]
    assert (frame["close"] > 0).all()
    assert truth["tape"] == "meanrev"


def test_determinism():
    a, _ = data.synthetic_trending(n_days=1000, seed=7)
    b, _ = data.synthetic_trending(n_days=1000, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_meanrev_determinism():
    a, _ = data.synthetic_meanrev(n_days=1000, seed=7)
    b, _ = data.synthetic_meanrev(n_days=1000, seed=7)
    assert data.fingerprint(a) == data.fingerprint(b)


def test_trending_has_a_bear_minority(trending):
    _, truth = trending
    # The bull is the home state — bears are a minority of days but not negligible.
    assert 0.05 < truth["frac_bear"] < 0.45


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_trending(n_days=500, seed=1)
    b, _ = data.synthetic_trending(n_days=500, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)

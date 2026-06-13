"""Data-layer invariants for Study 93 (Round-Numbers)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from round_numbers import data  # noqa: E402


def test_shape_and_columns(null_walk):
    frame, truth = null_walk
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"]
    assert (frame["close"] > 0).all()


def test_determinism():
    a, _ = data.synthetic_daily(n_days=2000, magnet=0.8, seed=7)
    b, _ = data.synthetic_daily(n_days=2000, magnet=0.8, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_magnet_changes_the_tape():
    a, _ = data.synthetic_daily(n_days=2000, magnet=0.0, seed=7)
    b, _ = data.synthetic_daily(n_days=2000, magnet=0.8, seed=7)
    # Same noise draws, different barrier behaviour → different paths.
    assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())


def test_both_tapes_visit_the_band(null_walk, planted_magnet):
    # Both tapes must spend a non-trivial share of days near a round level, or the fade
    # never triggers and the spine test is vacuous.
    for frame, truth in (null_walk, planted_magnet):
        assert 0.1 < truth["frac_in_band"] < 0.95


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_daily(n_days=500, magnet=0.0, seed=1)
    b, _ = data.synthetic_daily(n_days=500, magnet=0.0, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fingerprint_accepts_list_of_arrays():
    a, _ = data.synthetic_daily(n_days=300, magnet=0.0, seed=1)
    b, _ = data.synthetic_daily(n_days=300, magnet=0.0, seed=2)
    fp = data.fingerprint([a["close"].to_numpy(), b["close"].to_numpy()])
    assert isinstance(fp, str) and len(fp) == 12

"""Data-layer invariants for Study 100 (Melting-Ice)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from melting_ice import data  # noqa: E402


def test_shape_and_columns(flat_highvol):
    frame, truth = flat_highvol
    assert list(frame.columns) == ["close"]
    assert len(frame) == truth["n_days"] == 4000
    assert (frame["close"] > 0).all()


def test_determinism():
    a, _ = data.synthetic_path(path="uptrend_lowvol", seed=7)
    b, _ = data.synthetic_path(path="uptrend_lowvol", seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_default_seed_is_path_specific():
    # Default flat seed is chosen for a near-flat *realised* path; both paths reproducible.
    a, _ = data.synthetic_path(path="flat_highvol")
    b, _ = data.synthetic_path(path="flat_highvol")
    assert data.fingerprint(a) == data.fingerprint(b)


def test_flat_tape_is_actually_flat(flat_highvol):
    """The flat control's realised drift is ~0 — a genuinely trend-less tape, by design."""
    frame, _ = flat_highvol
    c = frame["close"]
    realized_logcagr = float(np.log(c.iloc[-1] / c.iloc[0]) / (len(c) / 252.0))
    assert abs(realized_logcagr) < 0.05  # within +/-5%/yr of flat


def test_uptrend_tape_trends_up(uptrend_lowvol):
    frame, truth = uptrend_lowvol
    c = frame["close"]
    assert c.iloc[-1] > c.iloc[0]
    assert truth["expect"] == "compound"


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_path(path="flat_highvol", seed=1)
    b, _ = data.synthetic_path(path="flat_highvol", seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_pairs_metadata_is_well_formed():
    for name, spec in data.PAIRS.items():
        assert spec["k"] == 3.0
        assert spec["underlying"] in {"QQQ", "SPY"}
        assert spec["start"][:4] in {"2009", "2010"}

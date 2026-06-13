"""Data-layer invariants for Study 94 (Level-Pegging)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from level_pegging import data  # noqa: E402


def test_shape_and_columns(broadening_panel):
    frame, truth = broadening_panel
    assert list(frame.columns) == ["ew", "cw"]
    assert len(frame) == truth["n_days"] == 4000
    assert (frame["ew"] > 0).all() and (frame["cw"] > 0).all()


def test_determinism():
    a, _ = data.synthetic_panel(mode="megacap", seed=7)
    b, _ = data.synthetic_panel(mode="megacap", seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    assert data.fingerprint(a) == data.fingerprint(b)


def test_broadening_plants_ew_win(broadening_panel):
    _, truth = broadening_panel
    assert truth["ew_wins"] is True


def test_megacap_plants_ew_loss(megacap_panel):
    _, truth = megacap_panel
    assert truth["ew_wins"] is False


def test_fingerprint_changes_with_data():
    a, _ = data.synthetic_panel(mode="broadening", seed=1)
    b, _ = data.synthetic_panel(mode="broadening", seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_unknown_mode_raises():
    import pytest

    with pytest.raises(ValueError):
        data.synthetic_panel(mode="nonsense")

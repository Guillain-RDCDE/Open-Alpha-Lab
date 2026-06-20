"""Shared fixtures — deterministic synthetic panels where the dead names can be toggled.

The whole study is a *measurement of a bias*, so the fixtures hand a panel in which we
control exactly one thing: how many firms delist. ``null_panel`` has NO deaths (FULL and
SURVIVORS are identical — the control); ``biased_panel`` has a realistic 30% death rate
(deleting the dead manufactures an edge). Tests never touch the network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from survivorship_bias import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No deaths — the FULL and SURVIVORS views are identical (the zero-bias control)."""
    return data.synthetic_panel(n_periods=240, n_assets=200, death_rate=0.0,
                                signal_strength=0.0, seed=345)


@pytest.fixture
def biased_panel():
    """A realistic 30% death rate with NO true signal — deleting the dead manufactures one."""
    return data.synthetic_panel(n_periods=240, n_assets=200, death_rate=0.30,
                                signal_strength=0.0, seed=345)

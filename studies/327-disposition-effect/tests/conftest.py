"""Shared fixtures — deterministic synthetic firm × month panels with a *known* amount of
capital-gains-overhang premium, so tests never touch the network and the only thing the
disposition sort can harvest is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from disposition_effect import data  # noqa: E402


@pytest.fixture
def planted():
    """A panel with a real overhang premium and no momentum loading — the rule must win."""
    return data.synthetic_panel(overhang_premium=0.004, momentum_premium=0.0, seed=327)


@pytest.fixture
def null():
    """A panel with NO overhang premium — the disposition sort must be a coin flip."""
    return data.synthetic_panel(overhang_premium=0.0, momentum_premium=0.0, seed=327)


@pytest.fixture
def momentum_only():
    """No overhang premium, only a momentum loading — the confound: raw overhang should
    spuriously load, the momentum-orthogonalised residual should not."""
    return data.synthetic_panel(overhang_premium=0.0, momentum_premium=0.02, seed=327)

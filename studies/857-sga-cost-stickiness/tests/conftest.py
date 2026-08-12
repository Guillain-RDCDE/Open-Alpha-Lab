"""Shared fixtures — deterministic synthetic SG&A / sales panels with a *known* planted link
between cost discipline and forward returns, so tests never touch the network. The panels
manufacture quarterly SG&A + revenue levels obeying a per-firm true β₂, run them through the
real stickiness estimator, and either plant a forward-return edge (``planted``) or leave the
forward path pure noise (``null``)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sga_stickiness import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No link (forward returns = noise). The tercile long-short must not manufacture signal."""
    return data.synthetic_panel(n_names=24, n_quarters=52, edge=0.0, seed=857)


@pytest.fixture
def planted_panel():
    """A strong planted link: lean (high-discipline) names drift up, sticky ones drift down."""
    return data.synthetic_panel(n_names=24, n_quarters=52, edge=0.35, seed=857)

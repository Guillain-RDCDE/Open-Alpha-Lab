"""Shared fixtures — deterministic synthetic ROIC panels with a *known* planted premium, so
tests never touch the network. The only thing the tercile long-short can harvest (ROIC-rank
predictability of forward returns) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from roic_premium import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No premium (ROIC = noise). The tercile long-short must not manufacture significance."""
    return data.synthetic_panel(n_names=30, n_quarters=44, edge=0.0, seed=859)


@pytest.fixture
def planted_panel():
    """A strong planted premium: high-ROIC names drift up, low-ROIC names drift down."""
    return data.synthetic_panel(n_names=30, n_quarters=44, edge=0.15, seed=859)

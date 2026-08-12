"""Shared fixtures — deterministic synthetic DSO-buildup panels with a *known* planted (negative)
lead, so tests never touch the network. The only thing the tercile long-short can harvest
(low-DSO-buildup names out-returning high-buildup names) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dso_signal import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No lead (DSO change = noise). The tercile long-short must not manufacture significance."""
    return data.synthetic_panel(n_names=30, n_quarters=40, edge=0.0, seed=853)


@pytest.fixture
def planted_panel():
    """A strong planted red flag: high-DSO-buildup names drift down, low-buildup names drift up."""
    return data.synthetic_panel(n_names=30, n_quarters=40, edge=0.15, seed=853)

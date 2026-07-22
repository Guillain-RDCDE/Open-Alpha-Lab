"""Shared fixtures — deterministic synthetic deferred-revenue panels with a *known* planted
lead, so tests never touch the network. The only thing the tercile long-short can harvest
(deferred-revenue-growth-rank predictability of forward returns) is baked in or deliberately
absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from deferred_revenue import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No lead (signal = noise). The tercile long-short must not manufacture significance."""
    return data.synthetic_panel(n_names=30, n_quarters=40, edge=0.0, seed=798)


@pytest.fixture
def planted_panel():
    """A strong planted lead: high deferred-growth names drift up, low ones drift down."""
    return data.synthetic_panel(n_names=30, n_quarters=40, edge=0.15, seed=798)

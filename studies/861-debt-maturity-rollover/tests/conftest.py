"""Shared fixtures — deterministic synthetic short-term-debt-share panels with a *known* planted
penalty, so tests never touch the network. The only thing the tercile long-short can harvest
(high-share names under-earning) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from debt_maturity import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No penalty (share = noise). The claim long-short must not manufacture significance."""
    return data.synthetic_panel(n_names=30, n_quarters=40, edge=0.0, seed=861)


@pytest.fixture
def planted_panel():
    """A strong planted penalty: high-share (rollover-risk) names drift down, low-share up."""
    return data.synthetic_panel(n_names=30, n_quarters=40, edge=0.15, seed=861)

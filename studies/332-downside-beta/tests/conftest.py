"""Shared fixtures — deterministic synthetic firm × day return panels with a *known*
downside-beta premium (or none), so tests never touch the network and the only thing the
high-minus-low β⁻ sort can harvest (a priced downside loading) is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from downside_beta import data  # noqa: E402


@pytest.fixture
def null():
    """A tape with NO downside-risk premium (downside_premium=0) — the sort must be a
    fair die: the mechanical down-day drag is neutralised, nothing is paid for β⁻."""
    rdf, mkt, truth = data.synthetic_panel(downside_premium=0.0, seed=332)
    return rdf, mkt, truth


@pytest.fixture
def priced():
    """A tape with a real, plantable downside-risk premium — the sort should earn it and
    clear the inference bar (the positive control proving the harness can detect it)."""
    rdf, mkt, truth = data.synthetic_panel(downside_premium=0.0025, seed=332)
    return rdf, mkt, truth

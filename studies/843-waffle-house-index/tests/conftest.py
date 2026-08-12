"""Shared fixtures — deterministic synthetic worlds with a *known* market-adjusted
disaster drift (insurers down / rebuilders up), so tests never touch the network and
the only thing the event study can detect is baked in (``edge>0``) or deliberately
absent (``edge=0``)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from waffle_index import data  # noqa: E402


@pytest.fixture(scope="session")
def null_world():
    """A pure-null world (edge=0): every name is just market + idiosyncratic noise."""
    closes, events = data.synthetic_world(edge=0.0, seed=843)
    return closes, events


@pytest.fixture(scope="session")
def edge_world():
    """A strong planted insurer-down / rebuilder-up drift (edge=+0.15%/day)."""
    closes, events = data.synthetic_world(edge=0.0015, seed=843)
    return closes, events

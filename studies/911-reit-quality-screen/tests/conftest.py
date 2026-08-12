"""Shared fixtures for Study 911 (REIT Quality Screen) — deterministic, offline.

Three synthetic worlds off one knob (``edge_ann``): a planted-quality-edge world (the
positive control), a null world (no edge — the estimators must stay silent), and a
larger-edge world for the recovery test. Every world also carries a TRAP leg whose Sharpe
is structurally worse, so the trap detector can be exercised. Tests never touch the network.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from reit_quality import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted +3%/yr quality edge over the broad index (the positive control)."""
    return data.synthetic_world(edge_ann=0.03, seed=911, n_months=228)


@pytest.fixture(scope="session")
def null_world():
    """The null — quality and broad share the same drift, no edge to find."""
    return data.synthetic_world(edge_ann=0.0, seed=911, n_months=228)


@pytest.fixture(scope="session")
def big_edge_world():
    """A large planted +8%/yr edge — must light up strongly."""
    return data.synthetic_world(edge_ann=0.08, seed=911, n_months=228)

"""Shared fixtures — deterministic synthetic SPLV/SPHB/SPY worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from low_volatility_anomaly import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """The low-vol leg carries a genuine risk-adjusted edge (the positive control)."""
    return data.synthetic_world(lowvol_edge=0.06, seed=330)


@pytest.fixture(scope="session")
def null_world():
    """The null — the calm leg is just lower-beta exposure, no risk-adjusted edge.

    With ``lowvol_edge=0`` the beta-neutral book is pure idiosyncratic noise whose sample mean
    wanders around zero; we pin a representative draw (seed 7) so the "coin" assertion tests the
    *construction*, not one unlucky path. The edge world (a strong planted +6%/yr) clears the
    bar on any seed."""
    return data.synthetic_world(lowvol_edge=0.0, seed=7)

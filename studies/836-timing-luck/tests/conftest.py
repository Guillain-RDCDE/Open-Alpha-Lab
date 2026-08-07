"""Shared fixtures — deterministic synthetic return panels (no network, no real data).

A null world (``mom_edge=0``: cross-sectional momentum has zero genuine edge, so any
Sharpe dispersion across rebalance offsets is pure luck) and a positive-control world
(``mom_edge=1``: a genuine planted momentum premium the tranched book must harvest).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from timing_luck import data  # noqa: E402


@pytest.fixture(scope="session")
def null_world():
    """No momentum edge: the offset-to-offset Sharpe dispersion is pure luck."""
    ret, _ = data.synthetic_panel(mom_edge=0.0, seed=836, n_assets=30, n_days=2600)
    return ret


@pytest.fixture(scope="session")
def edge_world():
    """A genuinely planted momentum premium: the tranched book must light up."""
    ret, _ = data.synthetic_panel(mom_edge=1.0, seed=836, n_assets=30, n_days=2600)
    return ret

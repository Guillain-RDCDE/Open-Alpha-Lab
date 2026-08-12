"""Shared fixtures — deterministic synthetic sector panels (no network, no real data)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sn_lowvol import data  # noqa: E402


@pytest.fixture(scope="session")
def secmap():
    """The deterministic synthetic sector assignment (round-robin, 8 sectors)."""
    return pd.Series(data.synthetic_sectors(n_assets=40, n_sectors=8))


@pytest.fixture(scope="session")
def edge_world():
    """A planted STOCK-LEVEL low-vol effect: within a sector, calmer names out-earn."""
    return data.synthetic_panel(edge=0.1, seed=903, n_assets=40, n_days=1500, n_sectors=8)


@pytest.fixture(scope="session")
def null_world():
    """The null — within-sector vol carries no forward information (no sector premium)."""
    return data.synthetic_panel(
        edge=0.0, seed=903, n_assets=40, n_days=1500, n_sectors=8, sector_prem_ann=0.0
    )


@pytest.fixture(scope="session")
def confound_world():
    """Only a defensive-SECTOR premium (edge=0): a raw low-vol sort reaps it, a
    sector-neutral sort must not."""
    return data.synthetic_panel(
        edge=0.0, seed=903, n_assets=40, n_days=1500, n_sectors=8, sector_prem_ann=0.08
    )

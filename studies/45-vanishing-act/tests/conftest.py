"""Shared fixtures — deterministic synthetic size worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vanishing_act import data


@pytest.fixture(scope="session")
def premium_world():
    """A stable, non-decaying size premium — SMB should be clearly positive."""
    return data.synthetic_smb(premium=0.05, decay=False, seed=45)


@pytest.fixture(scope="session")
def decay_world():
    """A premium that ramps from positive to negative — the Banz story."""
    return data.synthetic_smb(premium=0.06, decay=True, seed=45)


@pytest.fixture(scope="session")
def null_world():
    """No premium ever."""
    return data.synthetic_smb(premium=0.0, decay=False, seed=45)

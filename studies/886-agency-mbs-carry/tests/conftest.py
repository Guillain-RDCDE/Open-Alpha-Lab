"""Shared fixtures — deterministic synthetic excess-return frames (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mbs_carry import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted +2%/yr duration-neutral MBS carry over a shared rate factor."""
    return data.synthetic_world(carry_annual=0.02, seed=886)


@pytest.fixture(scope="session")
def null_world():
    """The null — MBS and Treasury share a rate factor but there is NO carry to find."""
    return data.synthetic_world(carry_annual=0.0, seed=886)

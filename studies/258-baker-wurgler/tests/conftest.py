"""Shared fixtures -- deterministic synthetic monthly market tapes with a *known*
amount of Baker-Wurgler contrarian sentiment effect, so tests never touch the
network and the only thing the regression/regime sort can harvest is baked in or
deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from baker_wurgler import data  # noqa: E402


@pytest.fixture
def null_tape():
    """No contrarian effect (sentiment carries no forward info)."""
    df, truth = data.synthetic_market(n_months=360, contrarian_bps=0.0, seed=258)
    return df, truth


@pytest.fixture
def live_tape():
    """A planted contrarian effect: high prior sentiment depresses next-month return."""
    df, truth = data.synthetic_market(n_months=360, contrarian_bps=60.0, seed=258)
    return df, truth

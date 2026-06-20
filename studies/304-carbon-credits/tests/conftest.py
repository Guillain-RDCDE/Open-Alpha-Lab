"""Shared fixtures — deterministic synthetic carbon tapes with a *known* drift and
persistence, so tests never touch the network and the only two things the study can harvest
(a rewarded hold, or trend persistence a timing overlay can exploit) are baked in or
deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from carbon_credits import data  # noqa: E402


@pytest.fixture
def null_tape():
    """No drift, no persistence — holding earns nothing, timing is a coin."""
    frame, truth = data.synthetic_carbon(n_days=3000, mu=0.0, momentum=0.0, seed=304)
    return frame, truth


@pytest.fixture
def drift_tape():
    """Strong positive drift, no persistence — a genuinely rewarded buy-and-hold."""
    frame, truth = data.synthetic_carbon(n_days=3000, mu=0.20, momentum=0.0, seed=304)
    return frame, truth


@pytest.fixture
def trend_tape():
    """Mild drift but strong persistence — a tape where a timing overlay can add value."""
    frame, truth = data.synthetic_carbon(
        n_days=4000, mu=0.05, momentum=0.40, daily_vol=0.025, seed=304
    )
    return frame, truth

"""Shared fixtures — deterministic synthetic (asset, market, event) tapes.

No network, no real data. The synthetic world plants a one-day abnormal jump on
scheduled pseudo-event dates; ``edge=0`` is the null.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from nflx_crackdown import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_world():
    """A planted one-day abnormal jump on every pseudo-event (the positive control)."""
    return data.synthetic_world(edge=0.03, seed=851)


@pytest.fixture(scope="session")
def null_world():
    """The null — event sessions statistically identical to every other session."""
    return data.synthetic_world(edge=0.0, seed=851)

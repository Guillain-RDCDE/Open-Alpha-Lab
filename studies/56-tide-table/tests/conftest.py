"""Shared fixtures — deterministic synthetic CAPE worlds (no network, no real data)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tide_table import data


@pytest.fixture(scope="session")
def predictive_world():
    """CAPE genuinely forecasts the next decade's returns (negatively)."""
    return data.synthetic_world(predicts=0.9, seed=56)


@pytest.fixture(scope="session")
def null_world():
    """The null — CAPE tells you nothing."""
    return data.synthetic_world(predicts=0.0, seed=56)

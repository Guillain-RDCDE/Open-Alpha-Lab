"""Shared fixtures — paths and a small, fast quote-distance grid for the tests."""

import os
import sys

import numpy as np
import pytest

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))


@pytest.fixture(scope="session")
def deltas():
    """A wide grid (out into the tail) so the exponential-vs-power-law contrast is visible."""
    return np.linspace(0.25, 40.0, 80)

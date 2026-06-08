"""Shared fixtures — the offline synthetic universe, built once per session."""

import os
import sys

import pytest

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from true_strength import data


@pytest.fixture(scope="session")
def synthetic():
    """(frames, truth) — a toy market with trend/cycle names among random-walk noise."""
    return data.synthetic_universe(seed=0)

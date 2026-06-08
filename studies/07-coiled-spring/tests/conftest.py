"""Shared fixtures — the offline synthetic universe, built once per session."""

import os
import sys

import pytest

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from coiled_spring import data


@pytest.fixture(scope="session")
def synthetic():
    """(frames, true_springs) — a toy market with planted springboards among noise walks."""
    return data.synthetic_universe(seed=0)

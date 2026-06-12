"""Shared fixtures — deterministic synthetic short-vol worlds (no network, no real data)."""
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from free_fall import data


@pytest.fixture(scope="session")
def crash_world():
    """A short-vol carry with rare catastrophic crashes (the real situation)."""
    return data.synthetic_shortvol(crash_prob=0.003, seed=63)


@pytest.fixture(scope="session")
def null_world():
    """The null — carry with no crash tail."""
    return data.synthetic_shortvol(crash_prob=0.0, seed=63)

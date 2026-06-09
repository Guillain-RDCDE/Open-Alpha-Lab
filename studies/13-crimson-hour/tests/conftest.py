"""Shared fixtures — a deterministic synthetic tape with a *known* momentum and a
deliberately uninformative IB-rejection flag, so tests never touch the network and the
quantities the decomposition hunts are baked in."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crimson_hour import data


@pytest.fixture
def synth():
    """520 sessions, afternoon momentum baked at 0.12, IB-rejection carrying no extra info."""
    feat, truth = data.synthetic_sessions(n_sessions=520, momentum=0.12, seed=13)
    return feat, truth


@pytest.fixture
def feat(synth):
    return synth[0]


@pytest.fixture
def truth(synth):
    return synth[1]

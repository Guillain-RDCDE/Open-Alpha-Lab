"""Shared fixtures — deterministic synthetic panels with a *known* genuine gamma effect and a
VIX-driven confound, so tests never touch the network and the quantities the decomposition hunts
are baked in."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gamma_gospel import data


@pytest.fixture
def real_effect():
    """750 sessions: a genuine gamma effect (beta_de=0.06, beta_vol=0.002) ON TOP of the VIX confound."""
    return data.synthetic_panel(n_sessions=750, beta_vol=0.0020, beta_de=0.060, seed=14)


@pytest.fixture
def panel(real_effect):
    return real_effect[0]


@pytest.fixture
def truth(real_effect):
    return real_effect[1]


@pytest.fixture
def mirage():
    """750 sessions with NO genuine gamma effect (beta = 0): the raw gap is pure VIX confound."""
    return data.synthetic_panel(n_sessions=750, beta_vol=0.0, beta_de=0.0, seed=14)

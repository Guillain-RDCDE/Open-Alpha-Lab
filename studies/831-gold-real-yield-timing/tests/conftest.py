"""Shared fixtures — deterministic synthetic gold / real-yield tapes (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gold_real_yield import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_world():
    """A planted timing edge: forward gold loads on the lagged real-yield-fall rank."""
    dfd, truth = data.synthetic_daily(n_days=3000, edge=0.04, link_beta=8.0, seed=831)
    return dfd, truth


@pytest.fixture(scope="session")
def null_world():
    """The null — the inverse *link* is present but the real-yield *trend* predicts nothing forward."""
    dfd, truth = data.synthetic_daily(n_days=3000, edge=0.0, link_beta=8.0, seed=831)
    return dfd, truth

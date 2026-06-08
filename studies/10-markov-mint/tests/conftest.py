"""Shared fixtures — small synthetic baskets, built once per session."""

import os
import sys

import pytest

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STUDY)
sys.path.insert(0, os.path.abspath(os.path.join(_STUDY, "..", "..")))

from markov_mint import data


@pytest.fixture(scope="session")
def efficient():
    """A small efficient (martingale) null basket — the price is the fair posterior."""
    return data.efficient_markets(n_markets=400, seed=0)


@pytest.fixture(scope="session")
def biased():
    """A small basket with a planted favorite-longshot wedge — a real edge of known size."""
    return data.biased_markets(n_markets=400, seed=1)

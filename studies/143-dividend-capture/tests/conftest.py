"""Shared fixtures — deterministic synthetic dividend-event frames.

Tests never touch the network; the efficiency knob lets us verify the strategy
finds an edge only when we plant one (partial efficiency) and correctly returns
~zero when markets are fully efficient.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dividend_capture import data  # noqa: E402


@pytest.fixture
def efficient():
    """Fully efficient market: efficiency=0.0 means price drops by exactly the dividend.

    Convention: efficiency is the fraction of the dividend the market does NOT take back.
    efficiency=0.0 → price drops by 100% of div (fully efficient, EMH null).
    efficiency=0.5 → price drops by 50% of div (partially inefficient, capture earns ~half).
    """
    events, truth = data.synthetic_events(n_events=200, efficiency=0.0, seed=143)
    return events, truth


@pytest.fixture
def inefficient():
    """Partially efficient: price drops by only 50% of the dividend — capture earns ~half."""
    events, truth = data.synthetic_events(n_events=200, efficiency=0.5, seed=143)
    return events, truth

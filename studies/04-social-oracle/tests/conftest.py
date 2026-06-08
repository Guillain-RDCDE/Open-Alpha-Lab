"""Shared pytest fixtures. Adds the study root to the path and builds a small,
deterministic synthetic universe (panel + mention feed) so tests never touch the
network and the pump-and-fade signature is present for the machinery to find."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from social_oracle import data, mentions


@pytest.fixture
def universe():
    """A 20-name, 300-session toy panel with ~120 mentions carrying a pump-and-fade."""
    panel, feed = data.synthetic_panel(n_tickers=20, n_days=300, n_mentions=120, seed=7)
    return panel, feed


@pytest.fixture
def panel(universe):
    return universe[0]


@pytest.fixture
def feed(universe):
    return universe[1]


@pytest.fixture
def events(universe):
    panel, feed = universe
    ev, _ = mentions.to_events(feed, panel)
    return ev

"""Shared fixtures — deterministic synthetic daily tapes whose *character* is known, so
tests never touch the network and the only thing a trend rule can possibly harvest
(direction-persistence) is baked in (``trend``), absent (``random``), or fake-but-
regime-specific (``hype``)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from uranium_revival import data  # noqa: E402


@pytest.fixture
def trending():
    """Persistent bull/bear runs (regime='trend') — the trend rule SHOULD beat B&H here."""
    bars, truth = data.synthetic_daily(n_days=3000, regime="trend", seed=303)
    return bars, truth


@pytest.fixture
def coin():
    """A random walk (regime='random') — the trend rule should NOT reliably beat B&H."""
    bars, truth = data.synthetic_daily(n_days=3000, regime="random", seed=303)
    return bars, truth


@pytest.fixture
def hype():
    """A boom-bust bubble (regime='hype') — the 'uranium rocket' caricature."""
    bars, truth = data.synthetic_daily(n_days=2500, regime="hype", seed=303)
    return bars, truth

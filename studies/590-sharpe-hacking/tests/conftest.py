"""Shared fixtures — deterministic synthetic tapes (a zero-alpha null the demo games, and a tape
with a genuinely planted edge = positive control), so tests never touch the network and the only
thing an honest metric can reward (real edge) is either baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sharpe_hacking import data  # noqa: E402


@pytest.fixture
def null_tape():
    """Zero-alpha tape (honest Sharpe ~0.2): any Sharpe *gain* from a transform is an artefact."""
    ret, _ = data.synthetic_series(n_days=3000, alpha=0.0, seed=590)
    return ret


@pytest.fixture
def edge_tape():
    """A genuinely planted edge — the HONEST Sharpe must rise with it (the control)."""
    ret, _ = data.synthetic_series(n_days=3000, alpha=0.20, seed=590)
    return ret

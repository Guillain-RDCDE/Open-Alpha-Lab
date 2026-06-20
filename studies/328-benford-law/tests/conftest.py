"""Shared fixtures — deterministic synthetic price tapes with a *known* leading-digit law.

The Benford positive control (a wide-range geometric walk) should conform; the
anti-control (a range-bound, mean-reverting price) should not. Both are offline and
seeded, so the synthetic core never touches the network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from benford_law import data  # noqa: E402


@pytest.fixture
def benford_tape():
    """A wide-range geometric walk — should obey Benford's first-digit law."""
    return data.synthetic_benford(n_days=6000, seed=328)


@pytest.fixture
def nonbenford_tape():
    """A range-bound mean-reverting price — should fail Benford (digits cluster)."""
    return data.synthetic_nonbenford(n_days=6000, seed=328)

"""Shared fixtures for Study 1019 — deterministic, offline, no network.

The synthetic world is a subordinated process: daily returns are N(0, sigma^2 * m_t), and the
``signal_strength`` knob sets how much of ``log m_t`` runs through the volume clock. At 1.0
Clark (1973) is true by construction; at 0.0 the returns are exactly as fat-tailed but run on a
clock that volume knows nothing about.

Both worlds come from the same generator with one knob moved, which is the whole point: a
detector that fires on the planted world and *also* fires on the null is not detecting anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from volclock import data  # noqa: E402

N_DAYS = 4000


@pytest.fixture(scope="session")
def planted():
    """Variance runs entirely through the volume clock (signal_strength=1)."""
    return data.synthetic_tape(n_days=N_DAYS, signal_strength=1.0, seed=1019)


@pytest.fixture(scope="session")
def null_tape():
    """The matched null: same fat tails, clock unrelated to variance (signal_strength=0)."""
    return data.synthetic_tape(n_days=N_DAYS, signal_strength=0.0, seed=1019)

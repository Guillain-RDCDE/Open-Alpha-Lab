"""Shared fixtures for Study 1016 — deterministic, offline, no network.

The synthetic world is a GJR-GARCH index whose leverage term is the knob. Both fixtures draw the
**same** shocks from the same seed; only ``signal_strength`` moves, and the persistence the
leverage term carries is handed to the symmetric ARCH term in the null, so the two worlds have
identical volatility clustering. Every "elevator" statistic must therefore fire on the first and
stay quiet on the second — a detector that passes on both is not testing anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stairs import data  # noqa: E402

N_DAYS = 8000


@pytest.fixture(scope="session")
def planted():
    """Leverage planted at full strength (signal_strength=1)."""
    return data.synthetic_returns(n_days=N_DAYS, signal_strength=1.0, seed=1016)


@pytest.fixture(scope="session")
def null_world():
    """The matched, sign-symmetric null (signal_strength=0)."""
    return data.synthetic_returns(n_days=N_DAYS, signal_strength=0.0, seed=1016)

"""Shared fixtures — deterministic synthetic breadth panels.

All fixtures are offline and deterministic; no network calls anywhere in the test-suite.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hindenburg_omen import data  # noqa: E402


@pytest.fixture(scope="session")
def null_breadth():
    """A panel where the Hindenburg signal has no predictive power (crash_signal_bps=0)."""
    df, truth = data.synthetic_breadth(n_stocks=100, n_days=504, crash_signal_bps=0.0, seed=167)
    return df, truth


@pytest.fixture(scope="session")
def signal_breadth():
    """A panel where Hindenburg days are followed by a planted crash (crash_signal_bps=50)."""
    df, truth = data.synthetic_breadth(n_stocks=100, n_days=504, crash_signal_bps=50.0, seed=167)
    return df, truth

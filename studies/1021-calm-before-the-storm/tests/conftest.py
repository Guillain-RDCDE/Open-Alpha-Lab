"""Shared fixtures for Study 1021 — deterministic, offline, no network.

Two synthetic worlds from one generator with one knob moved. Both have GARCH volatility
that clusters and mean-reverts, and both have crashes at about the same rate. In the
**planted** world (``signal_strength=1``) a crash is more likely to start after a long calm;
in the **null** world (``signal_strength=0``) crashes are blind to calm, so mean reversion is
the only link between quiet and storm. A detector that fires in both worlds is detecting
mean reversion, not Minsky — which is precisely the confusion this study exists to avoid.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from calm import data  # noqa: E402

N_MONTHS = 2400


@pytest.fixture(scope="session")
def planted():
    """200 years of monthly returns with the planted calm → crash feedback."""
    return data.synthetic_monthly(n_months=N_MONTHS, signal_strength=1.0, seed=1021)


@pytest.fixture(scope="session")
def null_world():
    """The matched null: same GARCH, same crash rate, crashes blind to calm."""
    return data.synthetic_monthly(n_months=N_MONTHS, signal_strength=0.0, seed=1021)

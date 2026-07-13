"""Shared fixtures — deterministic synthetic daily tapes with a *known* pre/post-OpEx
drift, so tests never touch the network and the planted effect is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from charm_decay import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape with no planted charm effect — drift identical on/off the OpEx windows."""
    return data.synthetic_daily(n_years=25, pre_drift_bps=0.0, post_drift_bps=0.0, seed=768)


@pytest.fixture
def drift_tape():
    """A tape with a planted +8 bps pre-OpEx drift and −4 bps post-OpEx give-back."""
    return data.synthetic_daily(n_years=25, pre_drift_bps=8.0, post_drift_bps=-4.0, seed=768)

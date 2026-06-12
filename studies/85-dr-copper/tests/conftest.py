"""Shared fixtures — deterministic synthetic daily frames with a *known* amount of
cross-series predictability, so tests never touch the network and the regression
engine can be validated against known ground truth."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dr_copper import data  # noqa: E402


@pytest.fixture
def null_tape():
    """No predictive link (pred_r=0) — regressions should not be significant here."""
    daily, truth = data.synthetic_daily(n_years=20, pred_r=0.0, seed=85)
    return daily, truth


@pytest.fixture
def signal_tape():
    """Strong planted predictive link (pred_r=0.5) — regressions should be significant."""
    daily, truth = data.synthetic_daily(n_years=20, pred_r=0.5, seed=85)
    return daily, truth

"""Shared fixtures — deterministic synthetic panels with a *known* amount of
recovery-momentum, so tests never touch the network. The only thing a recovery-speed
rule can harvest (recovery speed predicting forward returns) is planted or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from recovery_speed import data  # noqa: E402


@pytest.fixture
def null_panel():
    """recovery_edge=0 — recovery speed predicts NOTHING forward. The rule is a coin."""
    return data.synthetic_panel(n_names=80, n_days=1500, recovery_edge=0.0, seed=333)


@pytest.fixture
def control_panel():
    """recovery_edge>0 — fast recoverers keep leading. The rule must bank it."""
    return data.synthetic_panel(n_names=80, n_days=1500, recovery_edge=0.0015, seed=333)

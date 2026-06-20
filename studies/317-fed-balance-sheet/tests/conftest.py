"""Shared fixtures — deterministic synthetic daily tapes with a *known* QE/QT edge, so
tests never touch the network and the only thing a "don't fight the Fed" rule can harvest
(a difference in drift between QE and QT days) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fed_balance_sheet import data  # noqa: E402


@pytest.fixture
def null():
    """A tape where regime carries NO information (qe_edge=0) — rule must be a fair coin."""
    frame, truth = data.synthetic_daily(n_days=6000, qe_edge=0.0, seed=317)
    return frame, truth


@pytest.fixture
def control():
    """A tape with a strong planted QE>QT drift (qe_edge=0.001) — rule should bank it."""
    frame, truth = data.synthetic_daily(n_days=6000, qe_edge=0.001, seed=317)
    return frame, truth

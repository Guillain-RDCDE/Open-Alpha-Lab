"""Shared fixtures — deterministic synthetic annual panels with a *known* dividend-
initiation premium (or no premium), so tests never touch the network and the only thing
the initiator screen can harvest (a planted return advantage) is baked in or deliberately
absent."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dividend_initiation import data  # noqa: E402


@pytest.fixture(scope="session")
def real_world():
    """Panel where initiators genuinely outperform — the harness MUST detect the edge."""
    return data.synthetic_panel(initiation_premium=0.10, seed=240)


@pytest.fixture(scope="session")
def null_world():
    """Panel where initiators carry no premium — the harness should find nothing (null)."""
    return data.synthetic_panel(initiation_premium=0.0, seed=240)

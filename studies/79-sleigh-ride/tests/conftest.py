"""Shared fixtures — deterministic synthetic daily tapes with a *known* amount of
Santa Claus Rally drift, so tests never touch the network and the only thing the
window detector can harvest (a planted seasonal drift) is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sleigh_ride import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A martingale tape (santa_bps 0) — the window detector must find no drift here."""
    bars, truth = data.synthetic_daily(n_years=50, santa_bps=0.0, seed=79)
    return bars, truth


@pytest.fixture
def planted_tape():
    """A tape with a real Santa drift (santa_bps 80) — the detector should find it here."""
    bars, truth = data.synthetic_daily(n_years=50, santa_bps=80.0, seed=79)
    return bars, truth

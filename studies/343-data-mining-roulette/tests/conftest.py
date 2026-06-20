"""Shared fixtures — deterministic synthetic tapes for the p-hacking machine.

Two tapes, no network: a pure NULL (planted_edge=0 → nothing to find but luck) and a
POSITIVE CONTROL (a planted edge on the signal feature the harness must rediscover)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from data_mining_roulette import data  # noqa: E402


@pytest.fixture
def null_tape():
    """No edge anywhere — every feature is noise; only luck can 'win'."""
    return data.synthetic_tape(n_days=2520, planted_edge=0.0, vol_cluster=0.0, seed=343)


@pytest.fixture
def control_tape():
    """A strong planted edge on f0_signal — a faithful miner must rediscover it."""
    return data.synthetic_tape(n_days=2520, planted_edge=0.10, vol_cluster=0.0, seed=343)

"""Shared fixtures — deterministic synthetic split worlds with a KNOWN planted (or
deliberately absent) post-split abnormal drift, so tests never touch the network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stock_split_modern import data  # noqa: E402


@pytest.fixture
def null_world():
    """Raw stock + market paths and split dates with NO planted drift (the null)."""
    return data.synthetic_world(planted_bps=0.0, seed=802)


@pytest.fixture
def planted_world():
    """Same, with a strongly PLANTED +8 bps/day post-split abnormal drift."""
    return data.synthetic_world(planted_bps=8.0, seed=802)

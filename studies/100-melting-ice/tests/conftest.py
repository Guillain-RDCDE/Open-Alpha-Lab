"""Shared fixtures for Study 100 (Melting-Ice) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from melting_ice import data  # noqa: E402


@pytest.fixture
def flat_highvol():
    """A zero-drift, high-vol tape — a 3x sleeve must DECAY below naive-3x here."""
    return data.synthetic_path(path="flat_highvol")


@pytest.fixture
def uptrend_lowvol():
    """A steady low-vol uptrend — a 3x sleeve must COMPOUND ahead of naive-3x here."""
    return data.synthetic_path(path="uptrend_lowvol")

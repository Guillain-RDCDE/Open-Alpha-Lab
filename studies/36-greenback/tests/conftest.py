"""Shared fixtures — the deterministic synthetic FX panel (carry + trend + crash) and the carry null."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from greenback import data

SEED = 36


@pytest.fixture
def fx():
    """Carry premium + trend + crash baked in."""
    return data.synthetic_fx(carry_strength=0.9, trend_strength=0.35, seed=SEED)


@pytest.fixture
def null():
    """Full-UIRP carry null (no premium, no crash) — but trend stays so momentum still has something."""
    return data.synthetic_fx(carry_strength=0.0, trend_strength=0.35, seed=SEED)


@pytest.fixture
def fx_xr(fx):
    return fx[0]


@pytest.fixture
def fx_rd(fx):
    return fx[1]


@pytest.fixture
def null_xr(null):
    return null[0]


@pytest.fixture
def null_rd(null):
    return null[1]

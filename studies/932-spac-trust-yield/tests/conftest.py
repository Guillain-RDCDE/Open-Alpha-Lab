"""Shared fixtures — deterministic synthetic panels for Study 932 (Trust Yield).

Two worlds, both offline and deterministic (fixed seed 932; no network):

- ``put_binds`` — the trust redemption right is real (``signal_strength=1``): quotes sit
  at a discount that decays to zero and the terminal payoff is the accrued trust, so
  buying below trust must pay.
- ``put_is_fiction`` — the null (``signal_strength=0``): quotes are an unanchored
  martingale walk and the terminal payoff is simply the last quote, so "below trust"
  carries no information and the rule must earn nothing.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from trust_yield import data  # noqa: E402


@pytest.fixture
def put_binds():
    """A world where the trust put is real — the rule must recover the planted discount."""
    return data.synthetic_panel(signal_strength=1.0, seed=932)


@pytest.fixture
def put_is_fiction():
    """The null — an unanchored walk paying its last quote; the rule must stay quiet."""
    return data.synthetic_panel(signal_strength=0.0, seed=932)

"""Shared fixtures — deterministic synthetic spin-off worlds for Study 930.

Two fixtures, both offline and deterministic (fixed seed 930; no network, no cache):

- ``planted`` — a world where the parent earns a real announcement-to-distribution run-up
  and the child earns a real drift over its first 21 regular-way sessions
  (``signal_strength=1``). The estimator must recover both.
- ``null_world`` — the same world with both effects switched off
  (``signal_strength=0``). The estimator must stay quiet.

The synthetic worlds use the *same schema* as the real loader, so ``strategy.event_panel``
runs on them unchanged — the machinery proof exercises the production code path.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from when_issued import data  # noqa: E402


@pytest.fixture
def planted():
    """A world with a genuine parent run-up and a genuine child regular-way drift."""
    return data.synthetic_daily(signal_strength=1.0, seed=930)


@pytest.fixture
def null_world():
    """The null: parent and child are pure beta plus idiosyncratic noise."""
    return data.synthetic_daily(signal_strength=0.0, seed=930)

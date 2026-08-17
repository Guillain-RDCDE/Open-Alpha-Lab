"""Shared fixtures — deterministic synthetic ladders for Study 951 (The Crossover Rung).

Two fixtures, both offline and deterministic (fixed seed 951; no network):

- ``paid_rung`` — a three-rung ladder in which the crossover rung really is paid a premium
  *on top of* its duration and equity-beta loadings (``signal_strength=1``). The two-factor
  head-to-head estimator must recover it.
- ``flat_ladder`` — the same world with zero alpha everywhere (``signal_strength=0``). The
  raw excess returns still differ across rungs (they load differently on drifting factors),
  so the estimator has every opportunity to manufacture a premium; it must not.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crossover_credit import data  # noqa: E402


@pytest.fixture
def paid_rung():
    """A ladder where the crossover rung carries a genuine, recoverable premium."""
    prices, truth = data.synthetic_panel(signal_strength=1.0, seed=951)
    return prices, truth


@pytest.fixture
def flat_ladder():
    """The null ladder — every rung earns exactly its factor loadings and no more."""
    prices, truth = data.synthetic_panel(signal_strength=0.0, seed=951)
    return prices, truth

"""Shared fixtures — deterministic synthetic panels for Study 959 (Crypto Fee War).

Two fixtures, both offline and deterministic (fixed seed 959; no network, no cache):

- ``fee_ladder`` — the planted world (``signal_strength=1``): each wrapper is shaved by
  its own published fee, from 19 bp to 150 bp, on top of a shared 50%-vol asset and an
  observed benchmark carrying the 24/7-versus-16:00 clock stub. The estimators must
  recover the planted ladder.
- ``flat_fees`` — the null (``signal_strength=0``): every wrapper charges the cohort
  average while the published fee sheet still shows the same dispersion. The fee-rank
  test faces an informative-looking input with nothing behind it and must stay quiet.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fee_war import data  # noqa: E402


@pytest.fixture
def fee_ladder():
    """A planted fee ladder the tracking-difference estimators should recover."""
    return data.synthetic_panel(signal_strength=1.0, seed=959)


@pytest.fixture
def flat_fees():
    """The null: one fee for everybody, a dispersed fee sheet, nothing to find."""
    return data.synthetic_panel(signal_strength=0.0, seed=959)

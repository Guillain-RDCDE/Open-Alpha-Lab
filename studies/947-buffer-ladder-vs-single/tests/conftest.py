"""Shared fixtures — deterministic synthetic panels for Study 947 (The Buffer Ladder).

Three fixtures, all offline and deterministic (fixed seed 947; no network, no cache):

- ``planted_ladder`` — a panel where the laddered wrapper genuinely earns a laddering
  premium (``signal_strength=1``) on top of the equal-weight vintage basket, net of a
  planted extra fee layer. The detector must recover it.
- ``null_ladder`` — the wrapper is *exactly* the basket net of the planted fee
  (``signal_strength=0``): a detector that reports anything more is biased.
- ``clean_null`` — no premium and no fee: the measured gap must sit on zero.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from buffer_ladder import data  # noqa: E402


@pytest.fixture
def planted_ladder():
    """A wrapper with a genuine, recoverable laddering premium (net of a planted fee)."""
    return data.synthetic_panel(signal_strength=1.0, extra_fee_ann=0.002, seed=947)


@pytest.fixture
def null_ladder():
    """No laddering premium — the wrapper is the basket minus the planted fee layer."""
    return data.synthetic_panel(signal_strength=0.0, extra_fee_ann=0.002, seed=947)


@pytest.fixture
def clean_null():
    """No premium and no fee — the measured gap must be indistinguishable from zero."""
    return data.synthetic_panel(signal_strength=0.0, extra_fee_ann=0.0, seed=947)

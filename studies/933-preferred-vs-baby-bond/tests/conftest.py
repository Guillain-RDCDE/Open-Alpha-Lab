"""Shared fixtures — deterministic synthetic panels for Study 933 (Same Issuer, Two Ladders).

Two fixtures, both offline and deterministic (fixed seed 933; no network, no cache):

- ``planted`` — an eight-issuer panel in which the junior (preferred) rung of every balance
  sheet earns a real **+6.0%/yr carry premium** over the senior (baby-bond) rung, on top of
  its higher loading on the issuer's credit factor. The pairwise estimator must recover it.
- ``null_panel`` — the same world with the premium set to exactly zero
  (``signal_strength=0``): the two rungs differ only in *beta*, not in *paid carry*, so the
  pairwise difference has mean zero by construction and the estimator must stay quiet.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from two_ladders import data  # noqa: E402


@pytest.fixture(scope="module")
def planted():
    """Panel with a genuine planted junior carry premium (the effect the study looks for)."""
    return data.synthetic_panel(signal_strength=1.0, seed=933)


@pytest.fixture(scope="module")
def null_panel():
    """Panel with the premium set to exactly zero — beta differs, paid carry does not."""
    return data.synthetic_panel(signal_strength=0.0, seed=933)

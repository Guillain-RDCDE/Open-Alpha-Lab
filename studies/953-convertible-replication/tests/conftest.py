"""Shared fixtures — deterministic synthetic panels for Study 953 (Replicating the Convert).

Two fixtures, both offline and deterministic (fixed seed 953; no network, no cache):

- ``planted`` — a fund built as a static equity/growth/credit mix **plus** a genuine 2%/yr
  alpha and a genuine delta ratchet (``signal_strength=1``): the positive control the
  harness must recover on both axes.
- ``null`` — the same fund with both planted effects switched off (``signal_strength=0``):
  a pure static mix plus idiosyncratic noise, on which the harness must find no alpha and
  no convexity.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from convert_repl import data  # noqa: E402


@pytest.fixture
def planted():
    """A fund with a real alpha and a real delta ratchet on top of a static mix."""
    return data.synthetic_panel(signal_strength=1.0, seed=953)


@pytest.fixture
def null():
    """A fund that is *exactly* a static mix plus noise — the null."""
    return data.synthetic_panel(signal_strength=0.0, seed=953)

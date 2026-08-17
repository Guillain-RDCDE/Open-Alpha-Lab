"""Shared fixtures — deterministic synthetic panels for Study 918 (Creation Halt).

Both fixtures are offline and deterministic (fixed seed 918; no network, no cache):

- ``planted`` — six (fund, uncapped twin) pairs, each with a *planted* premium that
  accretes at 12 bps/day while issuance is suspended and fades over the fifteen
  sessions after it resumes (``signal_strength=1``).
- ``null`` — the same six pairs with the planted premium switched off
  (``signal_strength=0``): the fund and the twin differ only by tracking noise, so the
  event dates carry no information at all.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from creation_halt import data  # noqa: E402


@pytest.fixture
def planted():
    """Six pairs, each with a genuine creation-halt premium the estimator must find."""
    return data.synthetic_panel(n_events=6, signal_strength=1.0, seed=918)


@pytest.fixture
def null():
    """Six pairs with no planted premium — the estimator must stay quiet."""
    return data.synthetic_panel(n_events=6, signal_strength=0.0, seed=918)

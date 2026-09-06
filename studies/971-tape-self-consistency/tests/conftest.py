"""Shared fixtures for Study 971 — deterministic, offline, no network.

The synthetic fixtures here are a *known-good* tape and a deliberately corrupted
one: the corrupted version has a missing session, a mis-applied split and a dropped dividend
planted in it at known places, so every check in the audit can be shown to fire when the fault
is present and stay quiet when it is not.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tape_audit import data  # noqa: E402

N_YEARS = 10


@pytest.fixture
def clean_tape():
    """A known-good tape: every consistency check must pass on it, exactly."""
    return data.synthetic_tape(n_years=N_YEARS, seed=971)


@pytest.fixture
def broken_tape(clean_tape):
    """The same tape with a dropped session, an unapplied split and a missing dividend."""
    frames, _ = clean_tape
    return data.corrupt_tape(frames, seed=971)


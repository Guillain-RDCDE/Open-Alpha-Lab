"""Shared fixtures — deterministic synthetic panels for Study 952 (After-Tax Equivalent).

Both fixtures are offline and deterministic (fixed seed 952; no network, no cache):

- ``planted`` — a world with a genuine 150 bp pre-tax coupon-yield gap between the taxable
  and muni legs plus a small duration mismatch (``signal_strength=1``). The theoretical
  break-even effective rate is 33.3%, and the solver must recover it from the tape alone.
- ``twin_null`` — identical coupon yields and identical duration (``signal_strength=0``):
  the two legs are statistical twins, so the break-even must collapse to ~0 and the
  *pre-tax* difference must be indistinguishable from zero.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from after_tax import data  # noqa: E402


@pytest.fixture
def planted():
    """A planted 150 bp pre-tax yield gap — break-even must land near 33.3%."""
    return data.synthetic_panel(signal_strength=1.0, seed=952)


@pytest.fixture
def twin_null():
    """Statistical twins — break-even ~0 and no pre-tax difference to find."""
    return data.synthetic_panel(signal_strength=0.0, seed=952)

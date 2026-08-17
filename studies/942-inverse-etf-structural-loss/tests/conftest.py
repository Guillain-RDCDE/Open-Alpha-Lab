"""Shared fixtures — deterministic synthetic tapes for Study 942 (The Inverse Tax).

Every fixture is offline and deterministic (fixed seed 942; no network, no cache):

- ``taxed`` — a synthetic inverse fund carrying a *planted* 3%/yr structural tax
  (``signal_strength=1``): the claim under test is true by construction, and the harness
  must recover it with the right sign and roughly the right size.
- ``fair`` — the same fund with **no** tax (``signal_strength=0``): a costless replicate,
  so the measured gap must sit at zero (the null).
- ``rich`` — a fund with a real dividend advantage and an above-unit financing
  passthrough: the configuration the *real* tape turns out to have, used to check the
  accounting decomposition recovers planted parameters it was not told.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from inverse_tax import data  # noqa: E402


@pytest.fixture
def taxed():
    """An inverse fund that really does leak 3%/yr versus an honest direct short."""
    return data.synthetic_daily(signal_strength=1.0, seed=942)


@pytest.fixture
def fair():
    """A costless replicate — the null: no structural gap either way."""
    return data.synthetic_daily(signal_strength=0.0, seed=942)


@pytest.fixture
def rich():
    """A fund with a dividend advantage and a 1.5x financing passthrough."""
    return data.synthetic_daily(
        signal_strength=1.0, tax_ann_full=0.01, div_yield_ann=0.02,
        passthrough=1.5, rate_ann=0.04, seed=942,
    )

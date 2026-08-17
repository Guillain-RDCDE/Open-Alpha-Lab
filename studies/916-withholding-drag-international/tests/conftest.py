"""Shared fixtures — deterministic synthetic panels for Study 916 (Withholding Drag).

Both fixtures are offline and deterministic (fixed seed 916; no network, no cache):

- ``planted`` — the broad fund suffers a further 8 pp of withholding on top of the
  benchmark's 10 pp (``signal_strength=1``), so a real gap of
  ``gross_yield x 0.08`` bp/yr exists and the estimator must recover it.
- ``null`` — every fund suffers the *same* withholding (``signal_strength=0``), so the
  true gap is exactly zero and the estimator must stay quiet.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from withholding import data  # noqa: E402


@pytest.fixture
def planted():
    """A panel with a genuine, known withholding gap on the broad fund."""
    return data.synthetic_panel(signal_strength=1.0, seed=916)


@pytest.fixture
def null():
    """A panel where every fund is taxed identically — the null."""
    return data.synthetic_panel(signal_strength=0.0, seed=916)

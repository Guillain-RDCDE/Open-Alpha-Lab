"""Shared fixtures — deterministic synthetic panels for Study 961 (Which Gold).

Two fixtures, both offline and deterministic (fixed seed 961; no network):

- ``planted`` — five wrappers on one bullion price, each shaved by its own published fee
  (``signal_strength=1``): the fee ranking *is* the tracking-difference ranking, and the
  estimators must recover it.
- ``null_panel`` — the same five wrappers, but every one of them charging the cohort-average
  fee while the published sheet still shows dispersion (``signal_strength=0``): the rank
  test faces an informative-looking input with nothing behind it and must stay quiet.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from which_gold import data  # noqa: E402


@pytest.fixture
def planted():
    """A planted fee ladder the measurement stack must recover."""
    return data.synthetic_panel(signal_strength=1.0, seed=961)


@pytest.fixture
def null_panel():
    """Every wrapper charging the same fee — the null the rank test must not fire on."""
    return data.synthetic_panel(signal_strength=0.0, seed=961)

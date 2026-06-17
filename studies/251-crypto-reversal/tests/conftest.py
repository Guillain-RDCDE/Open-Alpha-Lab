"""Shared fixtures for Study 251 (Crypto-Reversal) tests.

Two synthetic panels: one with a genuine planted reversal (``planted``) and one null
(``null_panel``). Both are deterministic and offline — the real-data tests are guarded
separately and skip when the yfinance cache is absent.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_reversal import data  # noqa: E402


@pytest.fixture(scope="session")
def planted():
    """A synthetic panel with a strong planted ~21-day idiosyncratic reversal."""
    return data.synthetic_panel(n_tokens=20, n_years=6, signal_strength=1.0, seed=213)


@pytest.fixture(scope="session")
def null_panel():
    """A synthetic panel with no cross-sectional predictability (the null)."""
    return data.synthetic_panel(n_tokens=20, n_years=6, signal_strength=0.0, seed=213)

"""Shared fixtures for Study 296 (Oscars-Effect).

All fixtures are deterministic and offline — they use either the hardcoded
Oscar-ceremony table or the synthetic generator, never the network. The real
^GSPC tape is only read from the cache via the study's own loader and is not
required for the test suite (the synthetic tape stands in).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from oscars_effect import data  # noqa: E402


@pytest.fixture
def oscar_table():
    """The hardcoded Oscar-ceremony table (67th–97th, 1995–2025)."""
    return data.oscar_table()


@pytest.fixture
def synthetic_null():
    """A synthetic daily tape with NO planted post-Oscar bump (the null)."""
    ret, cer, truth = data.synthetic_daily(post_oscar_bp=0.0, seed=296)
    return ret, cer, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic daily tape with a strong planted post-Oscar bump."""
    ret, cer, truth = data.synthetic_daily(post_oscar_bp=300.0, seed=296)
    return ret, cer, truth

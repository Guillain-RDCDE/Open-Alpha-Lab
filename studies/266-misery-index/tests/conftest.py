"""Shared fixtures for Study 266 (Misery-Index).

All fixtures are deterministic and offline -- they use either the hardcoded
misery table or the synthetic generator, never the network or the ^GSPC parquet
(so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from misery_index import data  # noqa: E402


@pytest.fixture
def misery_tbl():
    """The hardcoded misery-index table (1948-2025)."""
    return data.misery_table()


@pytest.fixture
def synthetic_null():
    """A synthetic (misery, fwd_return) frame with NO planted link (the null)."""
    df, truth = data.synthetic_panel(n_years=77, beta=0.0, seed=266)
    df["misery_chg"] = df["misery"].diff()
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic frame with a strong planted misery->return link."""
    df, truth = data.synthetic_panel(n_years=400, beta=0.08, seed=266)
    df["misery_chg"] = df["misery"].diff()
    return df, truth

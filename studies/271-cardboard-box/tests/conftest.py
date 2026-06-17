"""Shared fixtures for Study 271 (Cardboard-Box).

All fixtures are deterministic and offline -- they use either the hardcoded
freight/box-production table or the synthetic generator, never the network or the
^GSPC parquet (so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cardboard_box import data  # noqa: E402


@pytest.fixture
def freight():
    """The hardcoded box/rail-freight growth table (1970-2024, 55 rows)."""
    return data.freight_table()


@pytest.fixture
def synthetic_null():
    """A synthetic forecast panel with NO planted beta (the null)."""
    df, truth = data.synthetic_panel(n_years=55, beta=0.0, seed=271)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic forecast panel with a strong planted forecasting beta."""
    df, truth = data.synthetic_panel(n_years=400, beta=4.0, seed=271)
    return df, truth


@pytest.fixture
def synthetic_gspc():
    """A synthetic ^GSPC-like annual return frame to stand in for the real cache."""
    df, _ = data.synthetic_panel(n_years=56, beta=0.0, seed=99)
    # Map to a 'year'/'gspc_return' frame covering 1970..2025.
    import pandas as pd
    years = list(range(1970, 1970 + len(df)))
    return pd.DataFrame({"year": years, "gspc_return": df["fwd_return"].to_numpy()})

"""Shared fixtures for Study 270 (Underwear-Index).

All fixtures are deterministic and offline — they use either the hardcoded
underwear table / NBER calendar or the synthetic generator, never the network
or the Shiller parquet (so CI passes without the real data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from underwear_index import data  # noqa: E402


@pytest.fixture
def uw_table():
    """The hardcoded underwear index table (1992–2024)."""
    return data.underwear_table()


@pytest.fixture
def synthetic_null():
    """A synthetic annual frame with NO planted dip→recession link (the null)."""
    df, truth = data.synthetic_annual(n_years=400, signal_bps=0.0, seed=270)
    return df, truth


@pytest.fixture
def synthetic_signal():
    """A synthetic annual frame with a strong planted dip→recession link."""
    df, truth = data.synthetic_annual(n_years=400, signal_bps=6_000.0, seed=270)
    return df, truth


@pytest.fixture
def merged_like():
    """A synthetic merged frame shaped like fetch_sp500_annual() output."""
    import numpy as np

    df, _ = data.synthetic_annual(n_years=80, signal_bps=0.0, seed=270)
    df = df.sort_values("year").reset_index(drop=True)
    df["next_recession"] = df["recession"].shift(-1).fillna(False).astype(bool)
    rng = np.random.default_rng(270)
    df["sp500_return"] = rng.normal(0.08, 0.17, len(df))
    df["next_sp500_return"] = df["sp500_return"].shift(-1)
    return df

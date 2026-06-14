"""Shared fixtures — deterministic synthetic monthly inflation/equity panels
with a *known* Fisher pass-through coefficient, so tests never touch the Shiller
parquet and the only thing the strategy recovers (inflation hedging) is baked
in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from inflation_hedge import data  # noqa: E402


@pytest.fixture
def null_world():
    """No Fisher pass-through (beta=0) — the inflation hedge is completely absent."""
    df, truth = data.synthetic_world(n_years=100, fisher_beta=0.0, seed=152)
    return df, truth


@pytest.fixture
def full_hedge_world():
    """Full Fisher pass-through (beta=1) — nominal returns rise one-for-one with
    inflation; real returns are (approximately) invariant across regimes."""
    df, truth = data.synthetic_world(n_years=100, fisher_beta=1.0, seed=152)
    return df, truth


@pytest.fixture
def shiller_df():
    """The real Shiller monthly panel (loaded from the staged parquet cache).

    Skipped automatically if the parquet is not present (CI environments without
    the full _cache/).
    """
    import os as _os
    cache = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "_cache")
    )
    parquet = _os.path.join(cache, "shiller_sp500.parquet")
    if not _os.path.exists(parquet):
        pytest.skip("shiller_sp500.parquet not present — skip real-data test")
    return data.load_shiller(cache_dir=cache)

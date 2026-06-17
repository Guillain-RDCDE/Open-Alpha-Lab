"""Shared fixtures for Study 275 (Whisky-Cask).

All fixtures are deterministic and offline — they use the hardcoded whisky index
or the synthetic generator, never the network or the ^GSPC cache (so CI passes
without the real-data cache).
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from whisky_cask import data  # noqa: E402


@pytest.fixture
def wh_index():
    """The hardcoded annual whisky-cask index (2008–2024)."""
    return data.whisky_index()


@pytest.fixture
def synthetic_null():
    """A synthetic alt-asset with NO smoothing (the honest null)."""
    df, truth = data.synthetic_index(n_years=16, smoothing=0.0, seed=275)
    return df, truth


@pytest.fixture
def synthetic_smoothed():
    """A synthetic alt-asset with heavy appraisal smoothing planted."""
    df, truth = data.synthetic_index(n_years=400, smoothing=0.8, seed=275)
    return df, truth


@pytest.fixture
def merged_synth():
    """A synthetic merged frame shaped like fetch_gspc_annual() output."""
    import numpy as np
    import pandas as pd

    df, _ = data.synthetic_index(n_years=16, smoothing=0.4, seed=275)
    rng = np.random.default_rng(275)
    sp = rng.normal(0.08, 0.15, len(df))
    out = pd.DataFrame({
        "year": df["year"].to_numpy(),
        "index_level": df["index_level"].to_numpy(),
        "whisky_return": df["reported_return"].to_numpy(),
        "sp500_return": sp,
    })
    out["both_up"] = (out["whisky_return"] > 0) & (out["sp500_return"] > 0)
    out["whisky_beats_sp"] = out["whisky_return"] > out["sp500_return"]
    return out

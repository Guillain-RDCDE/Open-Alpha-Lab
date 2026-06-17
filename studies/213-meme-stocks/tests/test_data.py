"""Data-layer invariants for Study 213 (Meme Stocks)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meme_stocks import data  # noqa: E402

CACHE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "_cache"))


def test_meme_tickers_list():
    """The basket must contain exactly the six WSB names."""
    assert set(data.MEME_TICKERS) == {"GME", "AMC", "BB", "BBBY", "KOSS", "NOK"}
    assert len(data.MEME_TICKERS) == 6


def test_bbby_in_delisted():
    """BBBY bankruptcy must be explicitly disclosed in the DELISTED mapping."""
    assert "BBBY" in data.DELISTED
    # Bankruptcy date is in 2023
    assert "2023" in data.DELISTED["BBBY"]


def test_synthetic_panel_shape_and_determinism():
    a = data.synthetic_panel(premium=0.20, seed=7)
    b = data.synthetic_panel(premium=0.20, seed=7)
    assert a["prices"].shape == b["prices"].shape
    assert np.allclose(a["prices"].to_numpy(), b["prices"].to_numpy())
    assert np.allclose(np.array(a["bench"]), np.array(b["bench"]))


def test_synthetic_panel_null_has_zero_premium():
    p = data.synthetic_panel(premium=0.0, seed=213)
    assert p["premium"] == 0.0


def test_synthetic_panel_planted_has_positive_premium():
    p = data.synthetic_panel(premium=0.25, seed=213)
    assert p["premium"] > 0


def test_fingerprint_changes_with_data():
    a = data.synthetic_panel(premium=0.10, seed=1)
    b = data.synthetic_panel(premium=0.10, seed=2)
    assert data.fingerprint(a) != data.fingerprint(b)


def test_fingerprint_stable_same_inputs():
    a = data.synthetic_panel(premium=0.10, seed=99)
    b = data.synthetic_panel(premium=0.10, seed=99)
    assert data.fingerprint(a) == data.fingerprint(b)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(CACHE_PATH, "GME_px.parquet")),
    reason="real-tape cache absent offline/CI",
)
def test_real_panel_loads_gme():
    panel = data.load_panel(cache_dir=CACHE_PATH)
    assert "GME" in panel["prices"].columns
    assert len(panel["prices"]) > 100

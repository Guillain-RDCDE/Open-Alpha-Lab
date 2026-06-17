"""Data-layer tests for Study 220 (Small Dogs of the Dow) — offline/synthetic only."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from small_dogs import data  # noqa: E402

# Path guards for real-tape tests (skipped on CI where caches are absent).
DOGS88_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "88-dogs-of-the-dow", "_cache")
)
LOCAL_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
CACHE_EXISTS = os.path.exists(DOGS88_CACHE) or os.path.exists(LOCAL_CACHE)
BENCH_PATH = os.path.join(DOGS88_CACHE, "DIA_bench.parquet")
BENCH_EXISTS = os.path.exists(BENCH_PATH) or os.path.exists(
    os.path.join(LOCAL_CACHE, "DIA_bench.parquet")
)


def test_synthetic_panel_shape():
    """Panel dimensions match the declared truth block."""
    p = data.synthetic_panel(n_names=30, n_years=20, planted=True)
    assert p["yields"].shape == (20, 30)
    assert p["prices_abs"].shape == (20, 30)
    assert p["fwd_returns"].shape == (20, 30)


def test_synthetic_panel_deterministic():
    """Same seed -> same numbers every time."""
    a = data.synthetic_panel(planted=True, seed=7)
    b = data.synthetic_panel(planted=True, seed=7)
    assert np.allclose(a["fwd_returns"].to_numpy(), b["fwd_returns"].to_numpy())


def test_synthetic_panel_planted_flag():
    """Planted excess is positive; null excess is zero."""
    assert data.synthetic_panel(planted=True)["truth"]["excess_small"] > 0
    assert data.synthetic_panel(planted=False)["truth"]["excess_small"] == 0.0


def test_synthetic_prices_are_positive():
    """All absolute share prices are strictly positive (required for ranking)."""
    p = data.synthetic_panel(planted=True, seed=42)
    assert (p["prices_abs"].to_numpy() > 0).all()


def test_fingerprint_is_12_chars():
    """Fingerprint digest is always 12 hexadecimal characters."""
    p = data.synthetic_panel(planted=True, seed=1)
    fp = data.fingerprint(p["yields"])
    assert isinstance(fp, str) and len(fp) == 12


def test_fingerprint_differs_across_seeds():
    a = data.synthetic_panel(planted=True, seed=1)
    b = data.synthetic_panel(planted=True, seed=2)
    assert data.fingerprint(a["yields"]) != data.fingerprint(b["yields"])


@pytest.mark.skipif(not BENCH_EXISTS, reason="real-tape cache absent offline/CI")
def test_load_panel_bench_shape():
    """The bench series loads and has at least 6,000 daily observations."""
    panel = data.load_panel()
    assert "bench" in panel
    assert len(panel["bench"]) >= 6000


@pytest.mark.skipif(not CACHE_EXISTS, reason="real-tape cache absent offline/CI")
def test_load_panel_prices_has_columns():
    """The real panel contains at least 20 ticker columns."""
    panel = data.load_panel()
    assert panel["prices"].shape[1] >= 20

"""Tests for the data layer — synthetic tape is well-formed and deterministic;
real-data tests are cache-gated and SKIP in offline CI."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from reits_diversifier import data  # noqa: E402
from tests.conftest import requires_cache  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic tape tests (always run — no cache dependency)
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns(null_world):
    prices, truth = null_world
    expected_days = int(truth["n_years"] * 252)
    assert len(prices) == expected_days
    assert list(prices.columns) == ["EQ", "REIT", "BOND"]


def test_synthetic_prices_positive(null_world):
    prices, _ = null_world
    assert (prices > 0).all().all()


def test_synthetic_prices_start_at_100(null_world):
    prices, _ = null_world
    # The first row of prices is 100 * exp(first day return) — check it's near 100
    assert (prices.iloc[0] > 50).all()
    assert (prices.iloc[0] < 200).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_portfolio(n_years=10, diversification=0.0, seed=42)
    b, _ = data.synthetic_portfolio(n_years=10, diversification=0.0, seed=42)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_portfolio(n_years=10, diversification=0.0, seed=99)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_diversification_knob_changes_output(null_world, diversified_world):
    """A non-zero diversification knob should produce a different price path."""
    p_null, _ = null_world
    p_div, _ = diversified_world
    # Paths differ (income boost changes the REIT trajectory)
    assert not np.allclose(p_null["REIT"].to_numpy(), p_div["REIT"].to_numpy())
    # But EQ and BOND paths with the same seed and same base noise structure — may differ
    # because the RNG is consumed differently when income_boost is drawn; test REIT only
    assert not np.allclose(p_null["REIT"].to_numpy(), p_div["REIT"].to_numpy())


def test_synthetic_truth_dict_contents(null_world):
    _, truth = null_world
    for key in ("diversification", "n_years", "n_days", "seed", "ann_vols", "eq_reit_corr"):
        assert key in truth
    assert truth["diversification"] == 0.0
    assert truth["n_years"] == 20


def test_fingerprint_stable_and_content_sensitive(null_world, diversified_world):
    p_null, _ = null_world
    p_div, _ = diversified_world
    assert data.fingerprint(p_null) == data.fingerprint(p_null)
    assert data.fingerprint(p_null) != data.fingerprint(p_div)


def test_load_real_raises_without_cache(tmp_path):
    """Without a cache, load_real must raise FileNotFoundError (not crash silently)."""
    with pytest.raises(FileNotFoundError):
        data.load_real(cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Real-data tests — cache-gated; SKIP in offline CI
# ---------------------------------------------------------------------------
@requires_cache
def test_real_prices_all_tickers_present():
    prices = data.load_real()
    for t in data.TICKERS:
        assert t in prices.columns, f"ticker {t!r} missing from real panel"


@requires_cache
def test_real_prices_no_nans_after_start():
    prices = data.load_real()
    assert not prices.isna().any().any(), "NaN found in real price panel"


@requires_cache
def test_real_prices_start_at_vnq_inception():
    prices = data.load_real()
    assert str(prices.index[0].date()) >= data.VNQ_INCEPTION


@requires_cache
def test_real_prices_positive():
    prices = data.load_real()
    assert (prices > 0).all().all()


@requires_cache
def test_real_fingerprint_deterministic():
    prices = data.load_real()
    assert data.fingerprint(prices) == data.fingerprint(prices)

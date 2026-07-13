"""Offline tests for the S2F reconstruction — the exact-issuance anchors and the SF ratio."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_to_flow import data  # noqa: E402


def test_supply_exact_at_halvings(sf_curve):
    """Circulating supply must hit the EXACT consensus integers at each halving date.

    reward*210_000 summed over epochs: 10.5M / 15.75M / 18.375M / 19.6875M.
    """
    exact = {
        "2012-11-28": 10_500_000.0,
        "2016-07-09": 15_750_000.0,
        "2020-05-11": 18_375_000.0,
        "2024-04-20": 19_687_500.0,
    }
    for d, want in exact.items():
        got = float(sf_curve.loc[pd.Timestamp(d), "supply"])
        assert abs(got - want) < 1.0, f"{d}: supply {got:,.0f} != {want:,.0f}"


def test_sf_doubles_across_halving(sf_curve):
    """SF should roughly double across a halving (flow halves, stock barely moves)."""
    before = float(sf_curve.loc[pd.Timestamp("2020-05-10"), "sf"])
    after = float(sf_curve.loc[pd.Timestamp("2020-05-12"), "sf"])
    assert 1.9 < after / before < 2.1, f"SF ratio across halving = {after/before:.2f}"


def test_flow_is_stepwise_and_halves(sf_curve):
    """Flow is the annualised issuance and must halve at the halving."""
    before = float(sf_curve.loc[pd.Timestamp("2020-05-10"), "flow"])
    after = float(sf_curve.loc[pd.Timestamp("2020-05-12"), "flow"])
    assert abs(before / after - 2.0) < 0.02
    assert abs(before - 12.5 * data.BLOCKS_PER_YEAR) < 1.0


def test_supply_flow_deterministic():
    a = data.supply_flow_daily()
    b = data.supply_flow_daily()
    pd.testing.assert_frame_equal(a, b)


def test_supply_monotone_nondecreasing(sf_curve):
    assert (sf_curve["supply"].diff().dropna() >= -1e-6).all()


def test_synthetic_world_shape_and_columns(null_world):
    assert set(["price", "sf", "true_resid"]).issubset(null_world.columns)
    assert len(null_world) == 3000
    assert (null_world["price"] > 0).all()


def test_synthetic_resid_stationary(null_world):
    """The planted valuation gap is a stationary AR(1): finite, mean-reverting, not exploding."""
    r = null_world["true_resid"]
    assert np.isfinite(r).all()
    assert abs(r.mean()) < 0.2
    assert 0.7 < r.autocorr(1) < 0.99

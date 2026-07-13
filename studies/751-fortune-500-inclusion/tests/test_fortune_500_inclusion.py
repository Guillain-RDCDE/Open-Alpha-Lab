"""Offline, deterministic tests for Study 751 — Fortune-500-Inclusion.

No network: everything runs on the deterministic synthetic control or on tiny in-memory
frames. The spine is the positive/negative control — the engine must recover a planted
added-bucket CAR edge and must NOT manufacture significance from ~a dozen events per bucket
when the true edge is zero.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fortune_500_inclusion import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Event table sanity
# --------------------------------------------------------------------------- #
def test_event_table_shape():
    assert len(data.FORTUNE_EVENTS) == 26
    n_added = sum(e["added"] for e in data.FORTUNE_EVENTS)
    assert n_added == 14 and (len(data.FORTUNE_EVENTS) - n_added) == 12


def test_fingerprint_deterministic():
    assert data.fingerprint(data.FORTUNE_EVENTS) == data.fingerprint(data.FORTUNE_EVENTS)
    assert len(data.fingerprint(data.FORTUNE_EVENTS)) == 12


# --------------------------------------------------------------------------- #
# welch_t
# --------------------------------------------------------------------------- #
def test_welch_t_zero_mean_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    assert abs(st.welch_t(x)) < 3.0


def test_welch_t_shifted_mean_large():
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 1, 500)
    assert st.welch_t(x) > 5.0


def test_welch_t_too_short_is_nan():
    assert np.isnan(st.welch_t(np.array([1.0])))


# --------------------------------------------------------------------------- #
# summarize_bucket / net_of_costs
# --------------------------------------------------------------------------- #
def test_summarize_bucket_keys_and_winrate_range():
    s = st.summarize_bucket(np.array([0.01, -0.02, 0.03, 0.0]))
    for k in ("n", "mean_pct", "win", "t"):
        assert k in s
    assert 0.0 <= s["win"] <= 1.0


def test_net_of_costs_reduces_and_can_flip():
    gross = st.net_of_costs(0.0002, cost_bps=0.0)["net_pct"]
    net = st.net_of_costs(0.0002, cost_bps=10.0)["net_pct"]
    assert net < gross
    assert net < 0.0  # a 2-bp gross edge is wiped out by a 10-bp round-trip


# --------------------------------------------------------------------------- #
# Positive / negative control — the spine
# --------------------------------------------------------------------------- #
def test_planted_edge_raises_added_mean():
    null = data.synthetic_events(car_bps=0.0, seed=751)
    plant = data.synthetic_events(car_bps=500.0, seed=751)
    assert plant["added_car"].mean() > null["added_car"].mean()


def test_null_control_not_significant():
    """With no planted edge, added-minus-dropped must stay below |t|=2."""
    null = data.synthetic_events(car_bps=0.0, seed=751)
    t = st.welch_t(null["added_car"], null["dropped_car"])
    assert abs(t) < 2.0


def test_large_edge_is_significant():
    """A +500 bps planted edge must light up added-minus-dropped past |t|=2."""
    plant = data.synthetic_events(car_bps=500.0, seed=751)
    t = st.welch_t(plant["added_car"], plant["dropped_car"])
    assert t > 2.0


# --------------------------------------------------------------------------- #
# event_car on a tiny synthetic frame (no network)
# --------------------------------------------------------------------------- #
def test_event_car_finite_on_clean_frame():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(1)
    spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), index=idx)
    stock = pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.02, 200))), index=idx)
    prices = pd.DataFrame({"XYZ": stock, "SPY": spy})
    car = st.event_car(prices, "XYZ", idx[160], window=(0, 2))
    assert np.isfinite(car)


def test_event_car_nan_when_no_history():
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    prices = pd.DataFrame({"XYZ": range(40), "SPY": range(40)}, index=idx).astype(float)
    # estimation window (120d) cannot fit -> NaN
    assert np.isnan(st.event_car(prices, "XYZ", idx[20], window=(0, 2)))


# --------------------------------------------------------------------------- #
# Real-tape guard (skipped offline / in CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real-tape cache absent offline/CI")
def test_real_tape_added_dropped_below_bar():
    """On the real tape the added-minus-dropped CAR t-stat is below 2 (a null signal)."""
    prices, events = data.load_real()
    panel = st.car_panel(prices, events)
    s = st.summarize(panel)
    assert abs(s["diff_t"]) < 2.0, (
        f"Real-tape added-dropped t = {s['diff_t']:.2f} unexpectedly >= 2.0; verdict may need update"
    )

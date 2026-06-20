"""The event-study engine's invariants and the study's two spines: (1) the harness detects
a planted sag and stays silent on a null, and (2) once you charge real two-leg costs the
believers' short stops paying — even on a generously planted sag."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lockup_expiry import strategy as st  # noqa: E402


# ---- HAC -------------------------------------------------------------------
def test_hac_zero_mean_is_insignificant():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    assert abs(st.hac_tstat(x)) < 2.0


def test_hac_nonzero_mean_is_significant():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1, 500)
    assert st.hac_tstat(x) > 2.0


# ---- AAR / CAR shape -------------------------------------------------------
def test_aar_columns_and_coverage(null_panel):
    panel, truth = null_panel
    a = st.aar(panel)
    assert list(a.columns) == ["tau", "aar", "n", "tstat"]
    assert len(a) == 2 * truth["window"] + 1
    assert (a["n"] == truth["n_events"]).all()


def test_car_keys_and_window(sag_panel):
    panel, _ = sag_panel
    s = st.car(panel, lo=-5, hi=5, n_boot=300)
    for k in ("window", "n", "mean_car_bps", "median_car_bps",
              "win_rate", "tstat", "ci_lo_bps", "ci_hi_bps"):
        assert k in s
    assert s["window"] == "[-5,+5]"


def test_per_event_car_sums_window(sag_panel):
    panel, _ = sag_panel
    car_i = st.per_event_car(panel, -2, 2)
    # one CAR per event, each the sum of 5 abnormal-return bars
    assert car_i.size == panel["event"].nunique()


# ---- Spine #1: detect a sag, stay silent on a null -------------------------
def test_detects_planted_sag(sag_panel):
    panel, _ = sag_panel
    s = st.car(panel, lo=-1, hi=1, n_boot=500)
    assert s["mean_car_bps"] < 0           # the sag is negative
    assert s["tstat"] <= -2.0              # and clears the bar
    assert s["ci_hi_bps"] < 0              # CI excludes zero on the negative side


def test_silent_on_null(null_panel):
    panel, _ = null_panel
    s = st.car(panel, lo=-1, hi=1, n_boot=500)
    assert abs(s["tstat"]) < 2.0           # noise: not distinguishable from zero
    assert not st.detects_sag(panel, lo=-1, hi=1)


def test_detects_sag_helper_true_on_sag_false_on_null(sag_panel, null_panel):
    assert st.detects_sag(sag_panel[0], lo=-1, hi=1) is True
    assert st.detects_sag(null_panel[0], lo=-1, hi=1) is False


# ---- Spine #2: costs kill the believers' short -----------------------------
def test_costs_kill_the_short(sag_panel):
    """Even on a generously planted sag, two legs of one-way costs flip the net edge.

    The short PnL is the negative window CAR. With a -300 bps one-shot sag over a 6-bar
    window the gross short pays; charging 4×cost on both legs eats it."""
    panel, _ = sag_panel
    gross = st.tradable_short(panel, lo=0, hi=5, cost_bps=0.0)
    netted = st.tradable_short(panel, lo=0, hi=5, cost_bps=30.0)
    assert gross["gross_bps"] > 0                       # the planted sag is shortable gross
    assert netted["net_bps"] < gross["net_bps"]        # costs lower the net
    assert netted["cost_bps_total"] == pytest.approx(4 * 30.0)  # two legs, round-trip


def test_short_pnl_is_negative_car(sag_panel):
    panel, _ = sag_panel
    s = st.tradable_short(panel, lo=0, hi=5, cost_bps=0.0)
    c = st.car(panel, lo=0, hi=5, n_boot=300)
    assert s["gross_bps"] == pytest.approx(-c["mean_car_bps"], rel=1e-6)


# ---- bootstrap -------------------------------------------------------------
def test_block_bootstrap_brackets_mean(sag_panel):
    panel, _ = sag_panel
    car_i = st.per_event_car(panel, -1, 1)
    lo, hi = st.block_bootstrap_ci(car_i, n_boot=500)
    assert lo <= car_i.mean() <= hi
    assert lo < hi

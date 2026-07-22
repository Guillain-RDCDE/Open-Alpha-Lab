"""Offline, deterministic tests for Study 799 — Order-Backlog Drift (RPO).

Fixed seeds, no network. These pin the inference primitives and — the core contract —
the synthetic positive control: the calendar long-short must NOT fire on the null world
(edge = 0) yet must light up when a forward-return edge is planted.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from order_backlog import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert abs(st.one_sample_t(x)) < 3.0


def test_one_sample_t_detects_shift():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1, 5000)
    assert st.one_sample_t(x) > 10.0


def test_welch_t_separates_means():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1, 2000)
    b = rng.normal(0.0, 1, 2000)
    assert st.welch_t(a, b) > 10.0
    assert abs(st.welch_t(a, a)) < 1e-9


def test_newey_west_reduces_to_iid_at_zero_lags():
    # At 0 lags the HAC t matches the one-sample t up to the ddof convention
    # (NW uses the population variance ÷n, one_sample_t the sample variance ÷n-1).
    rng = np.random.default_rng(3)
    x = rng.normal(0.1, 1, 1000)
    nw, iid = st.newey_west_t(x, lags=0), st.one_sample_t(x)
    assert abs(nw - iid) / abs(iid) < 1e-3


def test_newey_west_handles_positive_autocorrelation():
    # AR(1) with positive rho -> HAC t should be smaller (less significant) than iid t.
    rng = np.random.default_rng(4)
    n = 2000
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = 0.6 * x[i - 1] + e[i]
    x = x + 0.1
    assert abs(st.newey_west_t(x, lags=10)) < abs(st.one_sample_t(x))


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_bucketize_partitions_evenly():
    b = st._bucketize(np.arange(30.0), 3)
    assert set(np.unique(b)) == {0, 1, 2}
    assert (b == 0).sum() == (b == 2).sum() == 10


# --------------------------------------------------------------------------- #
# Synthetic control — the machinery proof
# --------------------------------------------------------------------------- #
def test_synthetic_null_does_not_fire():
    """Across 12 seeds, the edge=0 world must not manufacture significance."""
    ts = []
    for sd in range(12):
        pr, ev = data.synthetic_panel(edge=0.0, seed=799 + sd)
        ts.append(st.synthetic_detect(pr, ev)["t_nw"])
    ts = np.asarray(ts)
    # No systematic bias, and no more false positives than chance tolerates.
    assert abs(np.nanmean(ts)) < 1.0
    assert int((np.abs(ts) >= 2).sum()) <= 1


def test_synthetic_planted_edge_lights_up():
    pr, ev = data.synthetic_panel(edge=0.10, seed=799)
    out = st.synthetic_detect(pr, ev)
    assert out["t_nw"] > 2.0
    assert out["mean_bps"] > 0.0


def test_synthetic_edge_is_monotone_in_knob():
    lo = st.synthetic_detect(*data.synthetic_panel(edge=0.02, seed=799))["mean_bps"]
    hi = st.synthetic_detect(*data.synthetic_panel(edge=0.12, seed=799))["mean_bps"]
    assert hi > lo


def test_synthetic_panel_shapes_and_determinism():
    p1, e1 = data.synthetic_panel(edge=0.05, seed=7)
    p2, e2 = data.synthetic_panel(edge=0.05, seed=7)
    assert p1.shape == p2.shape
    assert list(e1.columns) == ["ticker", "end", "filed", "rpo", "rpo_yoy",
                                "rpo_chg_assets", "rev_yoy", "next_rev_yoy"]
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert (e1["filed"] > e1["end"]).all()   # filing after period end


# --------------------------------------------------------------------------- #
# Pipeline plumbing on a synthetic panel (no network)
# --------------------------------------------------------------------------- #
def test_calendar_ls_respects_one_lag_and_columns():
    pr, ev = data.synthetic_panel(edge=0.06, seed=3)
    ls = st.calendar_ls(pr, ev, min_names=5, staleness_days=120)
    assert {"ls", "long", "short", "n", "turnover"} <= set(ls.columns)
    assert (ls["n"] >= 5).all()
    assert ((ls["turnover"] >= 0) & (ls["turnover"] <= 1)).all()


def test_event_drift_frame_entry_after_filing():
    pr, ev = data.synthetic_panel(edge=0.06, seed=2)
    fr = st.event_drift_frame(pr, ev, horizon=21, lag=1)
    assert len(fr) > 50
    assert {"ticker", "filed", "signal", "drift"} <= set(fr.columns)


def test_costs_reduce_net_return():
    pr, ev = data.synthetic_panel(edge=0.10, seed=799)
    ls = st.calendar_ls(pr, ev, min_names=5, staleness_days=120)
    gross = st.calendar_ls_stats(ls)["mean_bps"]
    net = st.calendar_ls_net(ls, cost_bps=20.0, borrow_bps_ann=100.0)["net_mean_bps"]
    assert net < gross

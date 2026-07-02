"""Offline, deterministic tests for Study 558 (Failures-To-Deliver)."""
from __future__ import annotations

import numpy as np
import pytest

from failures_to_deliver import data, strategy as st


def test_synthetic_panel_deterministic():
    p1, t1 = data.synthetic_panel(seed=558)
    p2, t2 = data.synthetic_panel(seed=558)
    assert data.fingerprint(p1) == data.fingerprint(p2)
    assert not t1.has_effect  # default alpha = 0 is the null
    assert set(p1.columns) == {"name", "day", "ftd", "ret"}
    assert (p1["ftd"] > 0).all()


def test_no_lookahead_in_spike_flags():
    # spike flags at day t must use only data up to t-1: shifting the tail must not change earlier flags
    _, _ = data.synthetic_panel(seed=558)
    x = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.2, 400)) + 9)
    f_full = data.spike_flags(x, spike_z=2.0)
    f_trunc = data.spike_flags(x[:300], spike_z=2.0)
    # flags on the first 300 days are identical whether or not the future exists
    assert np.array_equal(f_full[:300], f_trunc)


def test_null_world_no_signal():
    # at the null the market-adjusted abnormal t is small (no post-spike pop)
    p, _ = data.synthetic_panel(squeeze_alpha=0.0, seed=558)
    es = st.event_study(p)
    assert es["n_events"] > 50
    assert abs(es["t"]) < 2.5  # this seed's null t is ~ -0.14


def test_null_seed_robust_flat():
    # averaged over seeds the null abnormal-t sits near 0 (house rule: no false signal)
    mt = st.synthetic_mean_t(0.0, n_seeds=20)
    assert abs(mt) < 1.5


def test_positive_control_detects_planted_squeeze():
    # planting a real post-spike pop drives the seed-robust abnormal-t well past +2
    weak = st.synthetic_mean_t(0.002, n_seeds=20)
    strong = st.synthetic_mean_t(0.006, n_seeds=20)
    assert weak > 1.0
    assert strong > weak > 0
    assert strong > 2.0


def test_placebo_centered_at_null():
    p, _ = data.synthetic_panel(squeeze_alpha=0.0, seed=558)
    pv = st.placebo_pvalue(p, n_perm=300, seed=558)
    assert 0.2 < pv < 0.8  # null spike labels carry no forward info


def test_placebo_small_when_planted():
    p, _ = data.synthetic_panel(squeeze_alpha=0.006, seed=558)
    pv = st.placebo_pvalue(p, n_perm=300, seed=558)
    assert pv < 0.05


def test_refractory_events_disjoint():
    days = np.array([10, 12, 13, 20, 21, 40])
    kept = st._refractory(days, horizon=5)
    assert kept == [10, 20, 40]  # each kept event > horizon apart


def test_net_costs_reduce_return():
    gross = 0.01
    net = st.net_event_return(gross, cost_bps=5.0, borrow_ann_bps=300.0, horizon_days=5)
    assert net < gross


def test_fetch_panel_is_empty_offline():
    assert data.fetch_panel().empty  # no free FTD tape to fetch


def test_robustness_sweep_shape():
    p, _ = data.synthetic_panel(seed=558)
    sw = st.robustness_sweep(p, horizons=(1, 5), thresholds=(2.5,))
    assert set(sw["horizon"]) == {1, 5}
    assert "t" in sw.columns

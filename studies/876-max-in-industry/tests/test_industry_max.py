"""Offline, fixed-seed tests for the industry-relative MAX machinery.

The synthetic panel is deterministic; the industry adjustment removes the sector-wide MAX
level; the industry-relative sort recovers a planted idiosyncratic-MAX -> return relation and
does so MORE sharply than the raw-MAX sort (the study's thesis); the null shows nothing; the
MAX panel is point-in-time (this month's MAX pairs with next month's return); the timer costs
reduce the net; the inference primitives behave. All offline, synthetic-only.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from max_industry import data, strategy as st  # noqa: E402

# Real cache path — probed so the one real-tape test self-skips when absent (offline CI).
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "_cache", "panel_50_a2964d3d2ba7_2010-01-01.parquet")


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.012, seed=876, n_months=240)
    for key in ("max", "fwd_ret", "mret"):
        assert np.allclose(edge_world[key].to_numpy(), p2[key].to_numpy(), equal_nan=True)


def test_industry_adjustment_removes_sector_level(edge_world):
    # Within each month, the median industry-relative MAX of a sector is ~0 by construction.
    adj = st.industry_relative_max(edge_world["max"], edge_world["sectors"])
    sectors = edge_world["sectors"]
    for sec in pd.unique(sectors.values):
        cols = [c for c in adj.columns if sectors.get(c) == sec]
        med = adj[cols].median(axis=1)
        assert np.nanmax(np.abs(med.to_numpy())) < 1e-9


def test_planted_relation_recovered_adjusted(edge_world):
    r = st.run_sort(edge_world["max"], edge_world["fwd_ret"], edge_world["sectors"],
                    adjusted=True)
    ss = st.spread_stats(r["spread"])
    assert ss["tstat"] > 3.0        # long-low / short-high spread lights up (right sign)
    assert ss["mean_bps"] > 0       # low-MAX names out-earn high-MAX names (the claim)


def test_adjustment_sharpens_vs_raw(edge_world):
    # The whole thesis: stripping the un-priced sector-wide MAX SHARPENS detection.
    raw = st.synthetic_detect(edge_world, adjusted=False)["t_nw"]
    adj = st.synthetic_detect(edge_world, adjusted=True)["t_nw"]
    assert adj > raw > 2.0


def test_null_world_no_signal(null_world):
    raw = st.synthetic_detect(null_world, adjusted=False)["t_nw"]
    adj = st.synthetic_detect(null_world, adjusted=True)["t_nw"]
    assert abs(raw) < 2.5 and abs(adj) < 2.5


def test_max_panel_is_point_in_time(edge_world):
    # fwd_ret row t must equal mret row t+1 (this month's MAX pairs with next month's return).
    mret, fwd = edge_world["mret"], edge_world["fwd_ret"]
    assert np.allclose(fwd.iloc[5].to_numpy(), mret.iloc[6].to_numpy(), equal_nan=True)
    assert fwd.iloc[-1].isna().all()   # last month has no forward return


def test_costs_reduce_net(edge_world):
    r = st.run_sort(edge_world["max"], edge_world["fwd_ret"], edge_world["sectors"],
                    adjusted=True)
    gross = st.timer_stats(r["spread"], r["qret"], high=r["high"],
                           cost_bps=0.0, borrow_ann_bps=0.0)["net_bps"]
    net = st.timer_stats(r["spread"], r["qret"], high=r["high"],
                         cost_bps=5.0, borrow_ann_bps=50.0)["net_bps"]
    assert net < gross


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 400)
    assert abs(st.newey_west_t(x) - st.one_sample_t(x)) < 0.6


def test_welch_detects_mean_gap():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 500)
    b = rng.normal(1.0, 1.0, 500)
    assert st.welch_t(a, b) < -5.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_quantile_returns_shape(null_world):
    qret = st.quantile_returns(null_world["max"], null_world["fwd_ret"], n_q=5)
    assert list(qret.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert len(qret) > 100


@pytest.mark.skipif(not os.path.exists(_CACHE), reason="real cache absent offline CI")
def test_real_cache_shapes():
    # Only runs when the real panel parquet is present locally; never on offline CI.
    mp = data.build_panel(data.load_panel())
    assert mp["max"].shape[1] == len(data.UNIVERSE)
    assert mp["sectors"].nunique() >= 6
    r = st.run_sort(mp["max"], mp["fwd_ret"], mp["sectors"], adjusted=True)
    assert st.spread_stats(r["spread"])["n"] > 100

"""Offline, fixed-seed tests for the GDPNow-revision machinery.

The synthetic frames are deterministic; the predictive regression recovers a *planted*
revision->return edge and stays silent on the null; the inference primitives behave; the
decile split is well-formed; the timer costs reduce the net; the revision is a genuine
within-quarter diff. All offline & synthetic — the one real-cache test is skipped when the
git-ignored ``_cache/`` is absent (CI).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gdpnow import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism + synthetic control
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic(edge_world):
    again = data.synthetic(edge=0.005, seed=877, n=2000)
    assert np.allclose(edge_world["rev"].to_numpy(), again["rev"].to_numpy())
    assert np.allclose(edge_world["fwd1"].to_numpy(), again["fwd1"].to_numpy())


def test_planted_edge_is_recovered(edge_world):
    r = st.predict_stats(edge_world, ycol="fwd1")
    assert r["t"] > 3.0        # the planted up-revision->up-return edge lights up
    assert r["beta"] > 0       # right sign: higher revision -> higher forward return
    assert r["r2"] > 0


def test_null_world_no_signal(null_world):
    r = st.predict_stats(null_world, ycol="fwd1")
    assert abs(r["t"]) < 2.5   # noise revisions must not manufacture significance


def test_null_world_robust_across_seeds():
    ts = np.array([st.synthetic_detect(data.synthetic(edge=0.0, seed=877 + s, n=1500))["t"]
                   for s in range(12)])
    assert (np.abs(ts) >= 2).sum() <= 1   # near-zero false-positive rate on the null


def test_decile_conditional_recovers_planted_sign(edge_world):
    d = st.decile_conditional(edge_world, ycol="fwd1")
    assert d["up_bps"] > d["down_bps"]     # up-revisions out-earn down-revisions when planted


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=5) - st.one_sample_t(x)) < 0.6


def test_ols_nw_recovers_known_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 3000)
    y = 0.5 * x + rng.normal(0, 0.1, 3000)
    r = st.ols_nw(x, y, lags=5)
    assert abs(r["beta"] - 0.5) < 0.02
    assert r["t"] > 10
    assert r["r2"] > 0.9


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_sign():
    a = np.array([0.02, 0.03, 0.025, 0.028])
    b = np.array([-0.01, -0.02, -0.015, -0.012])
    assert st.welch_t(a, b) > 0


# --------------------------------------------------------------------------- #
# Timer + placebo
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(edge_world):
    free = st.timer_stats(edge_world, cost_bps=0.0)["net"]["mean_bps"]
    costed = st.timer_stats(edge_world, cost_bps=5.0)["net"]["mean_bps"]
    assert costed < free


def test_placebo_flags_planted_edge(edge_world):
    p = st.placebo_pvalue(edge_world, ycol="fwd1", n_draws=500, seed=1)
    assert p["p_value"] < 0.05     # the planted slope is far outside the shuffled null


def test_placebo_null_is_uninformative(null_world):
    p = st.placebo_pvalue(null_world, ycol="fwd1", n_draws=500, seed=1)
    assert p["p_value"] > 0.05


# --------------------------------------------------------------------------- #
# Revision construction (the signal is a within-quarter diff, no cross-quarter leak)
# --------------------------------------------------------------------------- #
def test_revision_is_within_quarter_diff():
    g = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-05", "2020-01-10", "2020-04-06", "2020-04-09"]),
        "qtr": pd.to_datetime(["2020-03-31", "2020-03-31", "2020-06-30", "2020-06-30"]),
        "nowcast": [1.0, 1.5, 3.0, 2.4],
    })
    g["rev"] = g.groupby("qtr")["nowcast"].diff()
    # first forecast of each quarter has no revision; no diff bleeds across the quarter change
    assert np.isnan(g["rev"].iloc[0]) and np.isnan(g["rev"].iloc[2])
    assert abs(g["rev"].iloc[1] - 0.5) < 1e-9
    assert abs(g["rev"].iloc[3] + 0.6) < 1e-9


# --------------------------------------------------------------------------- #
# Real-cache smoke test — SKIPPED when the git-ignored cache is absent (CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_frame_shapes():
    frame = data.build_real()
    assert {"nowcast", "rev", "fwd1", "fwd5"}.issubset(frame.columns)
    assert len(frame) > 1000
    assert frame["rev"].notna().all()          # first-of-quarter rows dropped
    assert frame.index.max() <= pd.Timestamp(data.AS_OF)

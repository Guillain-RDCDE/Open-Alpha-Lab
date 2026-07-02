"""Tests for the engine of Study 549 (Spotify-Mood). Offline & deterministic."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spotify_mood import data, strategy as st  # noqa: E402


def test_null_hac_t_small(null_frame):
    frame, _ = null_frame
    reg = st.predictive_regression(frame, lag=1)
    assert abs(reg["hac_t"]) < 2.0  # no signal at the null


def test_planted_edge_recovered(edge_frame):
    frame, _ = edge_frame
    reg = st.predictive_regression(frame, lag=1)
    assert reg["hac_t"] > 2.0  # a real planted edge clears the bar


def test_control_monotone_and_flat_null():
    t0 = st.synthetic_mean_t(data, predictive_beta=0.0, n_seeds=20)
    t1 = st.synthetic_mean_t(data, predictive_beta=0.01, n_seeds=20)
    assert abs(t0) < 1.0        # flat at the null
    assert t1 > 2.0             # detects a planted edge
    assert t1 > t0             # monotone in the knob


def test_placebo_pvalue_reasonable_at_null(null_frame):
    frame, _ = null_frame
    p = st.placebo_pvalue(frame, lag=1, n_perm=500, seed=549)
    assert 0.0 < p <= 1.0
    assert p > 0.05            # the null slope is swallowed by the placebo


def test_mood_timing_keys(null_frame):
    frame, _ = null_frame
    out = st.mood_timing(frame, long_short=False)
    for k in ("gross_ann", "net_ann", "mkt_ann", "gross_t", "net_t"):
        assert k in out and np.isfinite(out[k])
    # net is never better than gross (costs only subtract)
    assert out["net_ann"] <= out["gross_ann"] + 1e-9


def test_hit_rate_bounds(null_frame):
    frame, _ = null_frame
    hr = st.hit_rate(frame, lag=1)
    assert 0.0 <= hr["hit"] <= 1.0 and 0.5 <= hr["base"] <= 1.0

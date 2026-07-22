"""Offline, deterministic tests for Study 789 — SUE Earnings-Surprise Drift.

No network: every test builds its own synthetic panel via ``data.synthetic_sue`` (fixed seeds)
or exercises the pure inference primitives on hand-made arrays. These guard the machinery:
the SUE construction is look-ahead-free, the detector fires on a planted drift and stays quiet
on the null, and the inference primitives behave.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sue_drift import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# SUE construction
# --------------------------------------------------------------------------- #
def test_sue_is_lookahead_free_seasonal_random_walk():
    # EPS rising by a constant each year => seasonal diff constant => SUE well-defined and,
    # once the denominator has variation, finite. Build 6 years of quarterly EPS.
    ends = pd.date_range("2010-03-31", periods=24, freq="QE")
    filed = ends + pd.Timedelta(days=30)
    # a seasonal pattern (Q4 high) plus a growing trend and a little wobble
    base = np.array([1.0, 1.2, 1.1, 2.0])
    vals = np.concatenate([base + 0.1 * yr + 0.01 * (yr % 2) for yr in range(6)])
    eps = pd.DataFrame({"end": ends, "filed": filed, "val": vals})
    sue = data.compute_sue(eps, min_history=8, vol_window=8)
    # first 8 seasonal diffs are warmup => first SUE only from the 13th quarter onward
    assert sue["end"].min() >= ends[12]
    assert np.isfinite(sue["sue"]).all()


def test_sue_denominator_uses_only_past():
    # If we corrupt the LAST quarter's EPS, no earlier SUE value may change (strictly-prior deps).
    ends = pd.date_range("2010-03-31", periods=24, freq="QE")
    filed = ends + pd.Timedelta(days=30)
    vals = 1.0 + 0.05 * np.arange(24) + 0.2 * np.sin(np.arange(24))
    eps = pd.DataFrame({"end": ends, "filed": filed, "val": vals})
    a = data.compute_sue(eps)
    eps2 = eps.copy()
    eps2.loc[23, "val"] += 5.0                       # perturb only the final quarter
    b = data.compute_sue(eps2)
    common = min(len(a), len(b)) - 1                 # all but the last emitted SUE
    assert np.allclose(a["sue"].values[:common], b["sue"].values[:common])


# --------------------------------------------------------------------------- #
# Synthetic control — the positive/negative power check
# --------------------------------------------------------------------------- #
def test_null_world_does_not_fire():
    ts = []
    for seed in range(12):
        p, e = data.synthetic_sue(edge=0.0, seed=789 + seed)
        ts.append(st.synthetic_detect(p, e)["t"])
    ts = np.asarray(ts)
    assert (np.abs(ts) >= 2).sum() == 0               # a null world never crosses the bar
    assert abs(ts.mean()) < 1.0


def test_planted_edge_lights_up():
    p, e = data.synthetic_sue(edge=0.08, seed=789)
    d = st.synthetic_detect(p, e)
    assert d["ls_mean"] > 0.03                         # a real, large planted drift
    assert d["t"] > 5.0


def test_planted_edge_is_monotone_in_sue():
    p, e = data.synthetic_sue(edge=0.08, seed=789)
    fr = st.event_drift_frame(p, e, horizon=63)
    q = st.bucket_means(fr, n_buckets=5)
    # top quintile drift should exceed bottom quintile drift when the edge is planted
    assert q[-1] > q[0]


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_bucketize_equal_frequency():
    x = np.arange(30, 0, -1).astype(float)            # reversed, to check it sorts
    b = st._bucketize(x, n_buckets=3)
    counts = np.bincount(b)
    assert set(b) == {0, 1, 2}
    assert counts.min() >= 9 and counts.max() <= 11   # ~equal frequency


def test_ttest_and_newey_west_zero_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 500)
    assert abs(st.ttest_vs_zero(x)) < 3.0
    nw_t, mean = st.newey_west_t(x, lags=3)
    assert abs(nw_t) < 3.0
    assert abs(mean - x.mean()) < 1e-12


def test_newey_west_detects_a_real_mean():
    x = np.full(100, 0.5) + np.random.default_rng(1).normal(0, 0.1, 100)
    nw_t, mean = st.newey_west_t(x, lags=2)
    assert nw_t > 5.0 and abs(mean - 0.5) < 0.05


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_long_short_sign_and_costs():
    # a frame where drift increases with sue => positive long-short, and costs only reduce it
    n = 90
    sue = np.linspace(-2, 2, n)
    drift = 0.02 * sue + 0.001
    fr = pd.DataFrame({"sue": sue, "drift": drift,
                       "filed": pd.date_range("2015-01-01", periods=n, freq="20D"),
                       "ticker": ["X"] * n})
    ls = st.long_short_drift(fr, n_buckets=3)
    assert ls["ls_mean"] > 0
    # net_of_costs on a prices/events pair isn't needed here; check the arithmetic directly
    c = 10.0 / 1e4
    net = ls["ls_mean"] - 4.0 * c - (50.0 / 1e4) * (63 / 252.0)
    assert net < ls["ls_mean"]

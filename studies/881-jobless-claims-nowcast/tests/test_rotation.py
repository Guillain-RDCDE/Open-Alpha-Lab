"""Offline, fixed-seed tests for the jobless-claims sector-rotation machinery.

The synthetic frame is deterministic; the claims change has the right sign; the
predictive regression recovers a *planted* negative (claim-shaped) slope; the null shows
nothing; the alignment is point-in-time (one lag, no look-ahead); costs reduce the net;
the inference primitives behave. The single real-cache test is skipped when the cache is
absent (offline CI). Everything else is synthetic-only.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from claims_nowcast import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world is deterministic + well-formed
# --------------------------------------------------------------------------- #
def test_world_deterministic(planted_frame):
    p2 = data.synthetic_frame(edge=0.5, seed=881, n_months=360)
    assert np.allclose(planted_frame.to_numpy(), p2.to_numpy())


def test_synthetic_schema(planted_frame):
    for col in ["claims"] + data.SECTORS:
        assert col in planted_frame.columns
    assert (planted_frame[data.SECTORS] > 0).all().all()          # positive prices
    assert planted_frame["claims"].min() > 0


def test_claims_change_sign():
    # a strictly rising claims level -> positive change; falling -> negative
    idx = pd.period_range("2000-01", periods=6, freq="M").to_timestamp("M")
    fr = pd.DataFrame({"claims": [300, 330, 300, 270, 300, 330.0]}, index=idx)
    dcl = st.claims_change(fr, k=1)
    assert dcl.iloc[1] > 0 and dcl.iloc[2] < 0 and dcl.iloc[3] < 0


# --------------------------------------------------------------------------- #
# The planted (claim-shaped) relation is recovered; the null is silent
# --------------------------------------------------------------------------- #
def test_planted_relation_recovered(planted_frame):
    r = st.predictive_regression(planted_frame)
    assert r["t_nw"] < -3.0        # rising claims -> cyclicals under-earn (claim holds)
    assert r["slope"] < 0
    assert r["corr"] < 0


def test_null_world_no_signal(null_frame):
    r = st.predictive_regression(null_frame)
    assert abs(r["t_nw"]) < 2.5


def test_null_across_seeds_rarely_fires():
    ts = np.array([st.synthetic_detect(data.synthetic_frame(edge=0.0, seed=881 + s,
                                                            n_months=360))["t_nw"]
                   for s in range(12)])
    assert (np.abs(ts) >= 2).sum() <= 1     # a well-behaved null fires ~5% of the time


def test_placebo_detects_planted(planted_frame):
    pl = st.placebo_pvalue(planted_frame, n_draws=500)
    assert pl["p_value"] < 0.05
    assert abs(pl["placebo_mean"]) < abs(pl["obs_slope"])


# --------------------------------------------------------------------------- #
# No look-ahead: the forward spread strictly follows the signal
# --------------------------------------------------------------------------- #
def test_alignment_is_point_in_time(planted_frame):
    x, y, idx = st.build_xy(planted_frame, k=1, lag=1)
    spread = st.cyc_def_spread(planted_frame)
    # y at position i must equal the spread realised one month AFTER idx[i]
    pos = spread.index.get_indexer(idx)
    assert np.allclose(y, spread.to_numpy()[pos + 1], equal_nan=True)


# --------------------------------------------------------------------------- #
# The timer: costs reduce the net; COVID sensitivity structure
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(planted_frame):
    gross = st.rotation_timer(planted_frame, cost_bps=0.0, borrow_bps_yr=0.0)["net_ann_pct"]
    net = st.rotation_timer(planted_frame, cost_bps=10.0, borrow_bps_yr=50.0)["net_ann_pct"]
    assert net < gross


def test_covid_sensitivity_keys(planted_frame):
    cs = st.covid_sensitivity(planted_frame)
    for key in ("full", "ex_covid", "winsor"):
        assert "slope" in cs[key] and "t_nw" in cs[key]


# --------------------------------------------------------------------------- #
# Inference primitives behave
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_t_symmetry():
    a = np.array([0.01, 0.02, -0.01, 0.03, 0.0])
    b = np.array([-0.02, 0.0, -0.03, 0.01, -0.01])
    assert np.isclose(st.welch_t(a, b), -st.welch_t(b, a))


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_hac_ols_recovers_known_slope():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 1000)
    y = 2.0 + (-0.7) * x + rng.normal(0, 0.5, 1000)
    out = st._hac_ols(x, y, lags=6)
    assert abs(out["slope"] - (-0.7)) < 0.05
    assert out["t_nw"] < -5


# --------------------------------------------------------------------------- #
# Real cache — skipped offline (git-ignored _cache absent on CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.ETF_CACHE),
                    reason="real cache absent offline CI")
def test_real_frame_shape():
    frame = data.load_real()
    assert {"claims"}.issubset(frame.columns)
    for t in data.SECTORS:
        assert t in frame.columns
    assert len(frame) > 200
    assert frame.index.max() <= pd.Timestamp(data.AS_OF)

"""Offline, fixed-seed tests for the aggregate-short-interest machinery.

The synthetic frame is deterministic; the detrended index is zero-mean/unit-var; the
predictive regression recovers a planted negative slope (RRZ direction); the null shows
nothing; the forward return is point-in-time (a documented publication lag, no
look-ahead); the placebo null centres at zero; the timer costs never increase net; the
inference primitives behave. All offline (synthetic only). One real-cache test is skipped
when ``_cache/`` is absent (as on CI).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest  # noqa: E402

from agg_short import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_frame(edge=0.015, seed=880, n_periods=200)
    assert np.allclose(edge_world["index"]["si_index"].to_numpy(),
                       p2["index"]["si_index"].to_numpy())
    assert np.allclose(edge_world["spy"].to_numpy(), p2["spy"].to_numpy())


def test_detrend_is_zero_mean_unit_var(edge_world):
    sii = st.detrend_log(edge_world["index"]["si_index"])
    assert abs(float(sii.mean())) < 1e-9
    assert abs(float(sii.std(ddof=0)) - 1.0) < 1e-9


def test_detrend_removes_linear_trend():
    # a pure exponential-trend index -> detrended residual is ~0 (unit-var scaling of noise-free 0)
    idx = pd.Series(np.exp(0.01 * np.arange(50)),
                    index=pd.bdate_range("2020-01-01", periods=50))
    sii = st.detrend_log(idx)
    assert float(np.abs(sii.to_numpy()).max()) < 1e-6


def test_planted_relation_recovered(edge_world):
    fr = st.build_frame(edge_world, horizon=1, lag=1, min_names=1)
    reg = st.predictive_regression(fr, nw_lags=6)
    assert reg["beta"] < 0                 # RRZ direction: high SI -> low forward return
    assert reg["t_nw"] < -2.0              # the planted slope lights up
    assert 0.0 <= reg["r2"] <= 1.0


def test_null_world_no_signal(null_world):
    fr = st.build_frame(null_world, horizon=1, lag=1, min_names=1)
    reg = st.predictive_regression(fr, nw_lags=6)
    assert abs(reg["t_nw"]) < 2.5


def test_forward_return_publication_lag():
    # Build a frame where SPY log-price is a clean ramp so the forward return over
    # [t+lag, t+lag+horizon] is exactly horizon * step. Confirms the lag alignment.
    n_days = 300
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    step = 0.001
    price = pd.Series(np.exp(step * np.arange(n_days)), index=idx, name="SPY")
    settle_pos = [(t + 1) * 10 - 1 for t in range(29)]
    settle = idx[settle_pos]
    index = pd.DataFrame({"si_index": np.linspace(2.0, 3.0, 29), "n": 50}, index=settle)
    real = {"index": index, "spy": price}
    fr = st.build_frame(real, horizon=1, lag=1, min_names=1)
    # each settlement is 10 business days apart -> one-period log return ~ 10*step
    assert np.allclose(fr["fwd"].to_numpy(), 10 * step, atol=1e-6)


def test_placebo_centers_at_zero(edge_world):
    fr = st.build_frame(edge_world, horizon=1, lag=1, min_names=1)
    pl = st.placebo_pvalue(fr, n_draws=2000, seed=1)
    assert abs(pl["placebo_mean"]) < 5e-4          # permuted slope centres near 0
    assert pl["p_value"] < 0.10                    # observed (planted) is in the left tail


def test_costs_never_increase_net(edge_world):
    fr = st.build_frame(edge_world, horizon=1, lag=1, min_names=1)
    gross = st.timer_stats(fr, cost_bps=0.0)["overlay_net_bps"]
    net = st.timer_stats(fr, cost_bps=5.0)["overlay_net_bps"]
    assert net <= gross + 1e-9


def test_era_split_partitions(edge_world):
    fr = st.build_frame(edge_world, horizon=1, lag=1, min_names=1)
    cut = fr.index[len(fr) // 2]
    es = st.era_split(fr, str(cut.date()))
    assert es["early"]["n"] + es["late"]["n"] == len(fr.dropna(subset=["sii", "fwd"]))


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_sign():
    a = np.array([0.02, 0.03, 0.01, 0.04, 0.02])
    b = np.array([-0.01, 0.0, -0.02, 0.01, -0.01])
    assert st.welch_t(a, b) > 0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


# --------------------------------------------------------------------------- #
# Real-cache test — SKIPPED when _cache/ is absent (offline CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_cache_regression_runs():
    real = data.load_real()
    assert len(real["index"]) > 100
    fr = st.build_frame(real, horizon=1, lag=1)
    reg = st.predictive_regression(fr, nw_lags=6)
    assert reg["n"] > 100
    assert np.isfinite(reg["beta"]) and np.isfinite(reg["t_nw"])
    assert 0.0 <= reg["r2"] <= 1.0

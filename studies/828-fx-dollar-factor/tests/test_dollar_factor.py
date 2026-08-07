"""Offline, fixed-seed tests for the FX dollar-factor machinery.

The synthetic panel is deterministic; DOL recovers a planted common dollar premium and a
planted timing relation; the null shows nothing; the quote-inversion normalises to
USD-per-foreign; the timing regression is point-in-time (no look-ahead); costs reduce the
net; the inference primitives behave. All offline. The one test that touches the real
cache is skipped when the parquet is absent.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dollar_factor import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism + construction
# --------------------------------------------------------------------------- #
def test_world_deterministic(null_world):
    p2 = data.synthetic_panel(edge=0.0, timing=0.0, seed=828, n_months=480)
    for c in null_world.columns:
        assert np.allclose(null_world[c].to_numpy(), p2[c].to_numpy())


def test_dol_is_equal_weight_mean():
    # a hand-built 3-currency panel: DOL spot must equal the mean of the per-ccy returns
    idx = pd.bdate_range("2020-01-31", periods=4, freq="ME")
    panel = pd.DataFrame(
        {"A": [1.0, 1.10, 1.10, 1.10], "B": [1.0, 1.00, 1.20, 1.20],
         "C": [1.0, 1.00, 1.00, 1.30]}, index=idx)
    dol = st.dol_series(panel, carry=None)
    ret = panel.pct_change().dropna(how="all")
    assert np.allclose(dol["spot"].to_numpy(), ret.mean(axis=1).to_numpy())


def test_excess_leg_adds_average_carry():
    panel = data.synthetic_panel(edge=0.0, timing=0.0, seed=1, n_months=120)
    panel.columns = data.CURRENCIES  # relabel so the carry map applies
    dol = st.dol_series(panel, carry=data.CARRY_PROXY)
    afd_ann = np.mean([data.CARRY_PROXY[c] for c in data.CURRENCIES]) / 100.0
    diff = (dol["excess"] - dol["spot"]).dropna()
    assert np.allclose(diff.to_numpy(), afd_ann / 12.0)


# --------------------------------------------------------------------------- #
# Planted-effect recovery + null silence
# --------------------------------------------------------------------------- #
def test_premium_recovered(premium_world):
    dol = st.dol_series(premium_world, carry=None)
    h = st.dol_stats(dol, "spot")
    assert h["t_nw"] > 3.0
    assert h["mean_bps"] > 0


def test_null_premium_silent(null_world):
    dol = st.dol_series(null_world, carry=None)
    h = st.dol_stats(dol, "spot")
    assert abs(h["t_nw"]) < 2.5


def test_timing_recovered(timing_world):
    dol = st.dol_series(timing_world, carry=None)
    reg = st.timing_regression(dol, "spot", 12)
    assert reg["t_beta"] > 3.0
    assert reg["beta"] > 0


def test_null_timing_silent(null_world):
    dol = st.dol_series(null_world, carry=None)
    reg = st.timing_regression(dol, "spot", 12)
    assert abs(reg["t_beta"]) < 2.5


def test_placebo_small_on_planted(timing_world):
    dol = st.dol_series(timing_world, carry=None)
    pl = st.block_shuffle_placebo(dol, "spot", 12, n_perm=300)
    assert pl["p_value"] < 0.10


def test_placebo_large_on_null(null_world):
    dol = st.dol_series(null_world, carry=None)
    pl = st.block_shuffle_placebo(dol, "spot", 12, n_perm=300)
    assert pl["p_value"] > 0.10


# --------------------------------------------------------------------------- #
# No look-ahead + costs
# --------------------------------------------------------------------------- #
def test_timing_is_point_in_time(null_world):
    # the regression pairs signal_t with DOL_{t+1}; the signal must use only past returns
    dol = st.dol_series(null_world, carry=None)
    sig = st.timing_signal(dol, "spot", 12)
    # signal at row t is the trailing 12m product ending at t -> equals product of last 12 rets
    r = dol["spot"].to_numpy()
    manual = np.prod(1.0 + r[0:12]) - 1.0
    assert np.isclose(sig.iloc[11], manual, atol=1e-12)
    assert np.isnan(sig.iloc[10])  # not enough history before month 12


def test_costs_reduce_net(premium_world):
    dol = st.dol_series(premium_world, carry=None)
    free = st.timer_stats(dol, "spot", 12, cost_bps=0.0)["static_net_ann_pct"]
    costed = st.timer_stats(dol, "spot", 12, cost_bps=50.0)["static_net_ann_pct"]
    assert costed < free


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_hac_slope_recovers_known_beta():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    y = 0.5 * x + rng.normal(0, 1, 2000)
    res = st._hac_slope_t(x, y, lags=6)
    assert abs(res["beta"] - 0.5) < 0.1
    assert res["t_beta"] > 5


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_t_sign():
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    b = np.array([0.0, 0.1, -0.1, 0.05, -0.05])
    assert st.welch_t(a, b) > 0


# --------------------------------------------------------------------------- #
# Real cache (guarded — skipped offline)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.FX_CACHE),
                    reason="real cache absent offline CI")
def test_real_inversion_sane():
    panel = data.load_panel()
    # USD-per-foreign: JPY ~ 0.005-0.012, CAD/CHF ~ 0.6-1.2, EUR/GBP ~ 1.0-1.8
    assert panel["JPY"].median() < 0.05
    assert 0.4 < panel["CAD"].median() < 1.5
    assert 0.9 < panel["EUR"].median() < 2.0

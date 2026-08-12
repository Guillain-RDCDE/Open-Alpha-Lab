"""Offline, fixed-seed tests for the currency-crash-risk machinery.

The synthetic panel is deterministic; the carry basket is dollar-neutral; the sort
recovers a planted skew-carry relation (higher carry -> more negative skew) and a
negatively-skewed basket; the null shows neither; the label-shuffle placebo separates
planted from null; costs reduce the net; the realized-skew and inference primitives
behave. All offline. The one test that touches the real cache is skipped when the
parquet is absent.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fx_crash import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism + construction
# --------------------------------------------------------------------------- #
def test_world_deterministic(planted_world):
    p2 = data.synthetic_panel(edge=0.02, seed=867, n_weeks=1000)
    assert np.allclose(planted_world["spot_ret"].to_numpy(), p2["spot_ret"].to_numpy())


def test_bundle_shape(null_world):
    for key in ("spot", "spot_ret", "total_ret", "carry"):
        assert key in null_world
    assert list(null_world["total_ret"].columns) == list(null_world["spot_ret"].columns)


def test_basket_is_dollar_neutral():
    w = st.carry_weights({"A": 3.0, "B": 1.0, "C": -1.0, "D": -3.0}, ["A", "B", "C", "D"], k=1)
    assert abs(float(w.sum())) < 1e-12
    assert w["A"] > 0 and w["D"] < 0  # long the top carry, short the bottom


def test_total_leg_adds_carry(null_world):
    # total_ret = spot_ret + carry/100/52, per currency
    tr, sr = null_world["total_ret"], null_world["spot_ret"]
    c = null_world["carry"]
    for col in sr.columns:
        diff = (tr[col] - sr[col]).to_numpy()
        assert np.allclose(diff, c[col] / 100.0 / data.PERIODS_PER_YEAR)


# --------------------------------------------------------------------------- #
# Planted-effect recovery + null silence
# --------------------------------------------------------------------------- #
def test_planted_skew_carry_recovered(planted_world):
    reg = st.skew_carry_regression(planted_world)
    assert reg["slope"] < 0            # higher carry -> more negative skew
    assert reg["t_slope"] < -3.0
    assert reg["spearman"] < -0.6


def test_null_skew_carry_silent(null_world):
    reg = st.skew_carry_regression(null_world)
    assert abs(reg["t_slope"]) < 2.5


def test_planted_basket_negatively_skewed(planted_world):
    bs = st.basket_stats(planted_world, k=3)
    assert bs["skew"] < -0.4           # a clear crash tail


def test_null_basket_not_skewed(null_world):
    bs = st.basket_stats(null_world, k=3)
    assert abs(bs["skew"]) < 0.5


def test_high_leg_more_skewed_than_low(planted_world):
    ls = st.leg_skews(planted_world, k=3)
    assert ls["hi_skew"] < ls["lo_skew"]   # high-carry leg is the crash-prone one
    assert ls["diff"] < 0


def test_placebo_small_on_planted(planted_world):
    pl = st.label_shuffle_placebo(planted_world, k=3, n_perm=400)
    assert pl["p_value"] < 0.10


def test_placebo_large_on_null(null_world):
    pl = st.label_shuffle_placebo(null_world, k=3, n_perm=400)
    assert pl["p_value"] > 0.10


# --------------------------------------------------------------------------- #
# Costs + skew primitives
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(planted_world):
    free = st.timer_stats(planted_world, cost_bps=0.0, borrow_bps_ann=0.0)["net_ann_pct"]
    costed = st.timer_stats(planted_world, cost_bps=5.0, borrow_bps_ann=100.0)["net_ann_pct"]
    assert costed < free


def test_costs_do_not_change_skew(planted_world):
    # costs shift the mean, not the shape
    tm0 = st.timer_stats(planted_world, cost_bps=0.0, borrow_bps_ann=0.0)
    tm1 = st.timer_stats(planted_world, cost_bps=5.0, borrow_bps_ann=100.0)
    assert np.isclose(tm0["skew"], tm1["skew"])


def test_realized_skew_of_left_skewed_is_negative():
    x = -np.abs(np.random.default_rng(0).normal(0, 1, 5000)) ** 1.5
    assert st.realized_skew(x) < 0


def test_skew_nw_t_small_on_symmetric():
    x = np.random.default_rng(0).normal(0.0, 1.0, 4000)
    assert abs(st.skew_nw_t(x, lags=6)) < 2.5


def test_skew_nw_t_sign_matches_skew():
    x = -np.abs(np.random.default_rng(1).normal(0, 1, 4000)) ** 1.6
    assert st.realized_skew(x) < 0
    assert st.skew_nw_t(x, lags=6) < 0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_t_sign():
    a = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    b = np.array([0.0, 0.1, -0.1, 0.05, -0.05])
    assert st.welch_t(a, b) > 0


def test_era_split_covers_all_weeks(planted_world):
    # a DatetimeIndex is needed for the era cut; synthetic uses RangeIndex, so build a
    # small dated bundle to exercise era_stats without look-ahead leakage
    idx = pd.bdate_range("2005-01-07", periods=400, freq="W-FRI")
    tr = pd.DataFrame(planted_world["total_ret"].to_numpy()[:400],
                      index=idx, columns=planted_world["total_ret"].columns)
    sr = pd.DataFrame(planted_world["spot_ret"].to_numpy()[:400],
                      index=idx, columns=planted_world["spot_ret"].columns)
    bundle = {"total_ret": tr, "spot_ret": sr, "carry": planted_world["carry"]}
    era = st.era_stats(bundle, split="2010-01-01", k=3)
    assert era["early"]["n_weeks"] > 0 and era["late"]["n_weeks"] > 0
    assert era["early"]["n_weeks"] + era["late"]["n_weeks"] == len(tr)


# --------------------------------------------------------------------------- #
# Real cache (guarded — skipped offline)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.FX_CACHE),
                    reason="real cache absent offline CI")
def test_real_inversion_sane():
    panel = data.load_panel()
    # USD-per-foreign: JPY ~ 0.006-0.012, MXN ~ 0.05-0.10, EUR/GBP ~ 1.0-2.0
    assert panel["JPY"].median() < 0.05
    assert 0.02 < panel["MXN"].median() < 0.12
    assert 0.9 < panel["EUR"].median() < 2.0

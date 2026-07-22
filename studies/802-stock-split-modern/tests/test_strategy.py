"""Tests for strategy.py — inference primitives, the abnormal-CAR event panel, the
costed timer, and the positive control (null must not fire; a planted drift must)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_split_modern import data, strategy as st  # noqa: E402


# ---- inference primitives -------------------------------------------------- #
def test_one_sample_t_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 500)
    assert abs(st.one_sample_t(x)) < 3.0


def test_one_sample_t_detects_shift():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1.0, 500)
    assert st.one_sample_t(x) > 5.0


def test_welch_t_sign():
    a = np.array([2.0, 3.0, 2.5, 3.5, 2.8])
    b = np.array([0.0, -1.0, 0.5, -0.5, 0.2])
    assert st.welch_t(a, b) > 0


def test_hac_t_finite_and_matches_sign():
    rng = np.random.default_rng(2)
    x = rng.normal(0.3, 1.0, 200)
    t = st.hac_t(x)
    assert np.isfinite(t) and t > 0


def test_wilson_interval_bounds():
    lo, hi = st.wilson_interval(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0


def test_hac_t_too_few_obs_is_nan():
    assert np.isnan(st.hac_t([0.1, 0.2]))


# ---- event panel ----------------------------------------------------------- #
def test_build_event_panel_columns(null_world):
    prices, _, splits = null_world
    panel = st.build_event_panel(prices, splits, era_split="1900-01-01")
    for c in ("ticker", "event_date", "ratio", "car_ex", "car_1m", "car_3m",
              "car_6m", "car_12m", "raw_3m", "mkt_3m"):
        assert c in panel.columns
    assert not panel.empty


def test_abnormal_car_is_stock_minus_market(null_world):
    prices, _, splits = null_world
    panel = st.build_event_panel(prices, splits, era_split="1900-01-01").dropna(subset=["car_3m"])
    row = panel.iloc[0]
    assert abs(row["car_3m"] - (row["raw_3m"] - row["mkt_3m"])) < 1e-9


def test_missing_market_raises(null_world):
    prices, _, splits = null_world
    with_no_mkt = prices.drop(columns=[data.MARKET])
    try:
        st.build_event_panel(with_no_mkt, splits)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---- horizon summary + timer ----------------------------------------------- #
def test_horizon_summary_keys(null_world):
    prices, _, splits = null_world
    panel = st.build_event_panel(prices, splits, era_split="1900-01-01")
    s = st.horizon_summary(panel, "car_3m")
    for k in ("n", "mean_pct", "median_pct", "std_pct", "t_hac", "t_plain", "win"):
        assert k in s


def test_timer_costs_reduce_long_net(planted_world):
    prices, _, splits = planted_world
    panel = st.build_event_panel(prices, splits, era_split="1900-01-01")
    free = st.timer_stats(panel, "car_3m", cost_bps=0.0, cohort="all")
    costed = st.timer_stats(panel, "car_3m", cost_bps=50.0, cohort="all")
    assert costed["net_pct"] < free["net_pct"]


def test_short_pays_borrow(planted_world):
    prices, _, splits = planted_world
    panel = st.build_event_panel(prices, splits, era_split="1900-01-01")
    no_borrow = st.timer_stats(panel, "car_3m", cost_bps=0.0, cohort="all",
                               side="short", borrow_bps_ann=0.0)
    borrow = st.timer_stats(panel, "car_3m", cost_bps=0.0, cohort="all",
                            side="short", borrow_bps_ann=2000.0)
    assert borrow["net_pct"] < no_borrow["net_pct"]


# ---- the positive control — the spine -------------------------------------- #
def test_null_control_does_not_fire():
    """Across seeds, the HAC detector must not systematically clear |t| = 2 on the null."""
    ts = []
    for s in range(12):
        pr, _, sp = data.synthetic_world(planted_bps=0.0, seed=802 + s)
        ts.append(st.synthetic_detect(pr, sp)["t_hac"])
    ts = np.asarray(ts)
    assert (np.abs(ts) >= 2).sum() <= 1          # at most one seed by chance
    assert abs(np.mean(ts)) < 1.5                # centred near zero


def test_planted_drift_lights_up():
    pr, _, sp = data.synthetic_world(planted_bps=8.0, seed=802)
    det = st.synthetic_detect(pr, sp)
    assert det["t_hac"] > 2.5
    assert det["mean_pct"] > 0

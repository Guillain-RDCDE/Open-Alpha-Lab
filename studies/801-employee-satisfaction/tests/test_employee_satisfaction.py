"""Offline, deterministic tests for Study 801 — Employee Satisfaction.

No network: everything runs on the deterministic synthetic control or on tiny in-memory
frames. The spine is the positive/negative control — the market-model alpha detector must
recover a planted monthly alpha and must NOT manufacture significance from a null world.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from employee_satisfaction import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Basket table sanity
# --------------------------------------------------------------------------- #
def test_basket_shape_and_uniqueness():
    assert len(data.BASKET) == 17
    tick = [t for t, _ in data.BASKET]
    assert len(set(tick)) == 17            # no duplicate tickers
    assert data.MARKET not in tick         # market is not inside the basket
    assert set(tick).isdisjoint(data.CONTROL)  # basket disjoint from placebo pool


def test_all_tickers_cover_everything():
    assert data.MARKET in data.ALL_TICKERS
    for t, _ in data.BASKET:
        assert t in data.ALL_TICKERS
    for t in data.CONTROL:
        assert t in data.ALL_TICKERS


def test_fingerprint_deterministic():
    idx = pd.date_range("2016-01-31", periods=12, freq="ME")
    df = pd.DataFrame({"A": np.linspace(0, 0.1, 12), "SPY": np.linspace(0, 0.05, 12)}, index=idx)
    assert data.fingerprint(df, ["A", "SPY"]) == data.fingerprint(df, ["A", "SPY"])
    assert len(data.fingerprint(df, ["A", "SPY"])) == 12


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_alpha_recovers_intercept():
    rng = np.random.default_rng(0)
    x = rng.normal(0.007, 0.04, 400)
    y = 0.003 + 1.1 * x + rng.normal(0, 0.01, 400)   # true monthly alpha 30 bps
    mm = st.newey_west_alpha(y, x, lags=3)
    assert abs(mm["alpha"] - 0.003) < 0.002
    assert abs(mm["beta"] - 1.1) < 0.05
    assert mm["t_alpha_nw"] > 3.0


def test_newey_west_alpha_null_not_significant():
    rng = np.random.default_rng(1)
    x = rng.normal(0.007, 0.04, 400)
    y = 1.1 * x + rng.normal(0, 0.02, 400)           # no alpha
    mm = st.newey_west_alpha(y, x, lags=3)
    assert abs(mm["t_alpha_nw"]) < 2.0


def test_newey_west_alpha_too_short_is_nan():
    mm = st.newey_west_alpha(np.array([0.01, 0.02]), np.array([0.01, 0.0]), lags=3)
    assert np.isnan(mm["alpha"])


def test_one_sample_and_nw_mean_agree_on_sign():
    rng = np.random.default_rng(2)
    x = rng.normal(0.004, 0.03, 300)
    assert st.one_sample_t(x) > 0 and st.newey_west_mean_t(x) > 0


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(74, 126)
    assert lo < 74 / 126 < hi and 0.0 <= lo <= hi <= 1.0


def test_equal_weight_basket_is_row_mean():
    idx = pd.date_range("2016-01-31", periods=5, freq="ME")
    m = pd.DataFrame({"A": [0.1, 0.0, -0.1, 0.2, 0.0],
                      "B": [0.0, 0.0, 0.1, 0.0, 0.2]}, index=idx)
    b = st.equal_weight_basket(m, ["A", "B"])
    assert np.allclose(b.to_numpy(), m.mean(axis=1).to_numpy())


# --------------------------------------------------------------------------- #
# Positive / negative control — the spine
# --------------------------------------------------------------------------- #
def test_planted_alpha_raises_recovered_alpha():
    null = st.synthetic_detect(data.synthetic_world(alpha_bps=0.0, seed=801))
    plant = st.synthetic_detect(data.synthetic_world(alpha_bps=80.0, seed=801))
    assert plant["alpha_ann_pct"] > null["alpha_ann_pct"]


def test_null_world_rarely_fires_across_seeds():
    """With no planted alpha, the detector must stay unbiased across 20 seeds."""
    ts = np.array([st.synthetic_detect(data.synthetic_world(alpha_bps=0.0, seed=801 + s))["t_alpha_nw"]
                   for s in range(20)])
    assert abs(ts.mean()) < 1.0                 # centred on zero
    assert (np.abs(ts) >= 2).sum() <= 3         # ~5% false positives at a 2-sigma bar


def test_large_planted_alpha_is_significant():
    plant = st.synthetic_detect(data.synthetic_world(alpha_bps=80.0, seed=801))
    assert plant["t_alpha_nw"] > 2.0


# --------------------------------------------------------------------------- #
# Timer costs reduce (never increase) net alpha
# --------------------------------------------------------------------------- #
def test_costed_timer_net_below_gross():
    w = data.synthetic_world(alpha_bps=40.0, seed=801)
    idx = w["basket"].index
    n = len(idx)
    # tiny per-name frame consistent with the basket series (mean == basket)
    monthly = pd.DataFrame({f"N{i}": w["basket"].to_numpy() for i in range(3)}, index=idx)
    monthly[data.MARKET] = w["market"].to_numpy()
    tickers = [f"N{i}" for i in range(3)]
    basket = st.equal_weight_basket(monthly, tickers)
    tm = st.timer_stats(basket, w["market"], monthly, tickers, cost_bps=10.0)
    assert tm["net_alpha_ann_pct"] <= tm["gross_alpha_ann_pct"]
    assert tm["turnover_1way_pct"] >= 0.0


# --------------------------------------------------------------------------- #
# Real-tape guard (skipped offline / in CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real-tape cache absent offline/CI")
def test_real_tape_alpha_below_bar():
    """On the real tape the CAPM-alpha NW t is below 2 (a weak, uncertified signal)."""
    monthly, basket_present, _ = data.load_real()
    basket = st.equal_weight_basket(monthly, basket_present)
    h = st.headline_stats(basket, monthly[data.MARKET])
    assert abs(h["t_alpha_nw"]) < 2.0, (
        f"Real-tape CAPM alpha t = {h['t_alpha_nw']:.2f} unexpectedly >= 2.0; verdict may need update"
    )

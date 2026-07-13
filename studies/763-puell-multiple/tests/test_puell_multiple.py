"""Fully offline, deterministic tests for Study 763 — Puell-Multiple.

No network: everything runs on synthetic/hardcoded inputs. Asserts pure-function behaviour of
the reconstruction and the inference machinery (the positive control must find a planted link and
read ~zero on the null).

    pytest -q studies/763-puell-multiple/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from puell_multiple import data, strategy as st  # noqa: E402


def test_reward_step_function():
    """The halving schedule is a right-continuous step; the constant on halving day is the new one."""
    idx = pd.DatetimeIndex(["2016-07-08", "2016-07-09", "2020-05-11", "2024-04-20", "2025-01-01"])
    r = data.reward_at(idx)
    assert list(r.values) == [25.0, 12.5, 6.25, 3.125, 3.125]


def test_blocks_per_day_cancels_in_ratio():
    """The Puell Multiple must be invariant to the blocks/day constant (it cancels)."""
    idx = pd.date_range("2018-01-01", periods=900, freq="D")
    price = pd.Series(np.linspace(100.0, 300.0, len(idx)) + 10.0 * np.sin(np.arange(len(idx)) / 30),
                      index=idx, name="btc_usd")
    pm = data.puell_multiple(price)
    saved = data.BLOCKS_PER_DAY
    try:
        data.BLOCKS_PER_DAY = 999  # any other constant
        pm2 = data.puell_multiple(price)
    finally:
        data.BLOCKS_PER_DAY = saved
    pd.testing.assert_series_equal(pm.dropna(), pm2.dropna())


def test_puell_flat_price_is_one_within_epoch():
    """A constant price far from any halving gives Puell == 1 (issuance == its own average)."""
    idx = pd.date_range("2021-01-01", periods=800, freq="D")  # no halving in window
    price = pd.Series(50000.0, index=idx, name="btc_usd")
    pm = data.puell_multiple(price).dropna()
    assert np.allclose(pm.values, 1.0, atol=1e-9)


def test_halving_suppresses_multiple():
    """A flat price straddling a halving must push the multiple below 1 for ~a year after it."""
    idx = pd.date_range("2019-06-01", periods=760, freq="D")  # spans 2020-05-11 halving
    price = pd.Series(9000.0, index=idx, name="btc_usd")
    pm = data.puell_multiple(price)
    # a month after the halving the denominator still holds double-reward days -> ratio < 1
    after = pm.loc["2020-06-15"]
    assert after < 0.99


def test_band_state_thresholds():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    pm = pd.Series([0.4, 0.5, 1.2, 4.0, 5.0], index=idx, name="puell")
    s = st.band_state(pm, high=4.0, low=0.5)
    assert list(s.values) == [1.0, 1.0, 0.0, -1.0, -1.0]


def test_no_lookahead_one_day_lag():
    """A single overheated day must cost exactly the NEXT day's return, not the same day's."""
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    # flat then a jump; place a high-Puell flag the day BEFORE the jump
    price = pd.Series([100, 100, 100, 200, 200, 200], index=idx, dtype=float, name="btc_usd")
    pm = pd.Series([1, 1, 5, 1, 1, 1], index=idx, dtype=float, name="puell")  # >=4 on day index 2
    # timing_signal flat on day 2 -> exposure applies to day 3's +100% move, which is skipped
    tb = st.backtest_timing(pm, price, high=4.0, cost_bps=0.0)
    # buy-and-hold captures the +100%; timer (flat over the jump) does not -> timer trails
    assert tb["bh_total_pct"] > tb["strat_total_pct"] + 50.0


def test_synthetic_positive_control_finds_planted_link():
    """Planted contrarian beta -> strongly negative HAC t; null -> ~zero."""
    pu1, pr1 = data.synthetic_world(beta=0.5, seed=763)
    r1 = st.synthetic_detect(pu1, pr1, horizon=30)
    assert r1["t_puell"] < -5.0            # detector clearly finds the planted contrarian link

    null_t = []
    for s in range(10):
        pu0, pr0 = data.synthetic_world(beta=0.0, seed=763 + s)
        null_t.append(st.synthetic_detect(pu0, pr0, horizon=30)["t_puell"])
    assert abs(np.mean(null_t)) < 1.0      # reads ~zero on the null on average


def test_welch_t_needs_two_obs():
    assert np.isnan(st.welch_t(np.array([1.0]), np.array([1.0, 2.0])))
    assert np.isfinite(st.welch_t(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0])))

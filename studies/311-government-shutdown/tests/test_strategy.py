"""The event-study engine's invariants, and the study's two spines: (1) the engine
recovers a planted post-event bounce only when one is really there, and (2) over the
right (random-date) null it is well-calibrated — no signal where none was planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from government_shutdown import data  # noqa: E402
from government_shutdown import strategy as st  # noqa: E402


# ---- snapping & engine invariants -----------------------------------------
def test_snap_picks_first_session_on_or_after(planted):
    bars, _, _ = planted
    # an event on a known session maps to that session's position
    target = bars.index[100]
    locs = st.snap_to_sessions(bars, pd.DatetimeIndex([target]))
    assert locs.tolist() == [100]
    # an event on a weekend-ish gap snaps forward, never backward
    between = bars.index[100] + pd.Timedelta(hours=12)
    locs2 = st.snap_to_sessions(bars, pd.DatetimeIndex([between]))
    assert locs2[0] >= 100


def test_lag_shifts_entry_forward(planted):
    bars, _, _ = planted
    target = pd.DatetimeIndex([bars.index[100]])
    assert st.snap_to_sessions(bars, target, lag=0)[0] == 100
    assert st.snap_to_sessions(bars, target, lag=1)[0] == 101


def test_forward_returns_columns_and_horizon(planted):
    bars, ev, _ = planted
    led = st.event_forward_returns(bars, ev, horizon=20, cost_bps=5.0)
    assert list(led.columns) == [
        "event_date", "entry", "exit", "horizon", "ret_gross", "ret_net",
    ]
    assert (led["horizon"] == 20).all()
    # net = gross - round-trip cost
    assert np.allclose(led["ret_net"], led["ret_gross"] - 5.0e-4)


def test_cost_monotonically_lowers_mean(planted):
    bars, ev, _ = planted
    means = [
        st.event_forward_returns(bars, ev, horizon=20, cost_bps=c)["ret_net"].mean()
        for c in (0.0, 5.0, 10.0)
    ]
    assert means[0] > means[1] > means[2]


# ---- inference helpers -----------------------------------------------------
def test_hac_tstat_detects_a_clear_mean():
    rng = np.random.default_rng(0)
    x = 0.05 + rng.normal(0, 0.02, 200)  # strongly positive mean
    assert st.hac_tstat(x) > 5
    z = rng.normal(0, 0.02, 200)         # zero mean
    assert abs(st.hac_tstat(z)) < 3


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    x = 0.03 + rng.normal(0, 0.05, 300)
    lo, hi = st.block_bootstrap_ci(x, block=3, n_boot=2000)
    assert lo < x.mean() < hi
    assert lo > 0  # a clearly positive mean has a CI above zero


def test_permutation_pvalue_bounds():
    rng = np.random.default_rng(2)
    ev = 0.05 + rng.normal(0, 0.02, 30)
    nul = rng.normal(0, 0.02, 2000)
    obs, p = st.permutation_pvalue(ev, nul, n_perm=4000)
    assert obs > 0
    assert 0.0 <= p <= 1.0
    assert p < 0.05  # a real shift is flagged


# ---- the two spines --------------------------------------------------------
def test_engine_recovers_planted_bounce(planted):
    """Spine #1a: with a planted post-event effect the engine clears the bar AND beats
    the random-date null."""
    bars, ev, _ = planted
    led = st.event_forward_returns(bars, ev, horizon=20)
    s = st.summarize(led["ret_gross"].to_numpy())
    nul = data.synthetic_null_events(bars, n_events=400, window=20, seed=312)
    nret = st.all_horizon_returns(bars, st.snap_to_sessions(bars, nul), 20)
    obs, p = st.permutation_pvalue(led["ret_gross"].to_numpy(), nret, n_perm=10000)
    assert s["tstat"] > 2.0
    assert obs > 0 and p < 0.05


def test_engine_finds_nothing_in_null(null):
    """Spine #1b: with no planted effect the excess over random dates is not significant."""
    bars, ev, _ = null
    led = st.event_forward_returns(bars, ev, horizon=20)
    nul = data.synthetic_null_events(bars, n_events=400, window=20, seed=315)
    nret = st.all_horizon_returns(bars, st.snap_to_sessions(bars, nul), 20)
    obs, p = st.permutation_pvalue(led["ret_gross"].to_numpy(), nret, n_perm=10000)
    assert p > 0.05  # clean null seed: no spurious signal


def test_car_curve_rebased_to_event(planted):
    """Spine #2: the average path is rebased to 0 at the event bar (t=0)."""
    bars, ev, _ = planted
    car = st.car_curve(bars, ev, pre=10, post=40)
    assert car.loc[0, "mean"] == pytest.approx(0.0, abs=1e-9)
    assert car.index.min() == -10 and car.index.max() == 40
    # a planted positive bounce should lift the path by the end of the window
    assert car.loc[40, "mean"] > car.loc[0, "mean"]

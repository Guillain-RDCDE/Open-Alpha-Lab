"""The event-study engine's invariants, the short-trade costs/borrow, and the study's two
spines: (1) the engine recovers a planted downgrade dip only when one is really there, and
the null gives nothing; (2) shorting a drifting-up tape bleeds the drift plus borrow, so
'sell the downgrade' starts from a hole even before the panic shows up."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sovereign_downgrade import data  # noqa: E402
from sovereign_downgrade import strategy as st  # noqa: E402


# ---- snapping -------------------------------------------------------------
def test_snap_to_sessions_applies_lag(planted):
    bars, _, _ = planted
    import pandas as pd

    ev = pd.DatetimeIndex([bars.index[100]])
    assert st.snap_to_sessions(bars, ev, lag=0).tolist() == [100]
    assert st.snap_to_sessions(bars, ev, lag=1).tolist() == [101]


# ---- abnormal returns & CARs ----------------------------------------------
def test_abnormal_returns_demeaned(planted):
    bars, _, _ = planted
    abn = st.abnormal_returns(bars)
    # the abnormal returns are the raw returns minus their mean → ~zero mean (ex bar 0)
    assert abs(abn[1:].mean()) < 1e-9


def test_event_day_columns_and_sign(planted, null):
    bars, ev, _ = planted
    df = st.event_day_abnormal(bars, ev, lag=0)
    assert list(df.columns) == ["event_date", "ret", "abn"]
    assert len(df) > 0
    # on the planted tape the announcement-bar abnormal return is mostly negative
    assert (df["abn"] < 0).mean() > 0.7


def test_car_window_and_sign(planted):
    bars, ev, _ = planted
    df = st.car(bars, ev, window=(0, 5), lag=0)
    assert list(df.columns) == ["event_date", "car"]
    assert len(df) > 0
    # the planted dip drags the [0,+5] CAR negative for most events
    assert (df["car"] < 0).mean() > 0.6


def test_window_cars_matches_random_positions(planted):
    bars, ev, _ = planted
    locs = st.snap_to_sessions(bars, ev, lag=0)
    cars = st.window_cars(bars, locs, window=(0, 5))
    df = st.car(bars, ev, window=(0, 5), lag=0)
    assert np.allclose(np.sort(cars), np.sort(df["car"].to_numpy()))


# ---- the short trade: costs + borrow + drift drag (spine #2) --------------
def test_short_columns_and_cost_borrow():
    bars, ev, _ = data.synthetic_spy(event_effect=0.0, n_events=40, seed=317)
    rt = st.short_the_downgrade(bars, ev, hold=5, cost_bps=5.0, borrow_bps_per_day=2.0)
    assert list(rt.columns) == ["event_date", "pnl_gross", "pnl_net"]
    drag = (rt["pnl_gross"] - rt["pnl_net"]).to_numpy()
    expected = 2.0 * 1e-4 * 5 + 2.0 * 5.0 * 1e-4
    assert np.allclose(drag, expected)


def test_short_bleeds_drift_on_upward_tape():
    """Spine #2: with NO planted dip but a positive drift, a blind short loses gross — the
    drift it fights — and net it loses even more (borrow + costs)."""
    bars, ev, _ = data.synthetic_spy(event_effect=0.0, n_events=40,
                                     daily_drift=0.0004, daily_vol=0.009, seed=317)
    rt = st.short_the_downgrade(bars, ev, hold=5, cost_bps=2.0, borrow_bps_per_day=1.0)
    assert rt["pnl_gross"].mean() < 0          # shorting an up-drift tape loses
    assert rt["pnl_net"].mean() < rt["pnl_gross"].mean()


def test_cost_monotonically_lowers_net(planted):
    bars, ev, _ = planted
    means = [
        st.summarize(
            st.short_the_downgrade(bars, ev, cost_bps=c)["pnl_net"].to_numpy()
        )["mean"]
        for c in (0.0, 5.0, 10.0)
    ]
    assert means[0] > means[1] > means[2]


# ---- inference invariants -------------------------------------------------
def test_hac_tstat_finite_on_planted(planted):
    bars, ev, _ = planted
    t = st.hac_tstat(st.event_day_abnormal(bars, ev, lag=0)["abn"].to_numpy())
    assert np.isfinite(t) and t < -2.0  # planted dip clears the bar (negative)


def test_block_bootstrap_ci_brackets_mean(planted):
    bars, ev, _ = planted
    x = st.event_day_abnormal(bars, ev, lag=0)["abn"].to_numpy()
    lo, hi = st.block_bootstrap_ci(x, n_boot=2000, seed=1)
    assert lo < x.mean() < hi


def test_permutation_pvalue_small_when_event_differs():
    """A planted event distribution far from the synthetic control gives a small p."""
    bars, ev, _ = data.synthetic_spy(event_effect=0.04, n_events=40, seed=315)
    nev = data.synthetic_null_events(bars, n_events=600, window=20, seed=3)
    locs_null = st.snap_to_sessions(bars, nev, lag=0)
    event_car = st.car(bars, ev, window=(0, 5), lag=0)["car"].to_numpy()
    null_car = st.window_cars(bars, locs_null, window=(0, 5))
    obs, p = st.permutation_pvalue(event_car, null_car, n_perm=5000, seed=1)
    assert obs < 0  # event CAR is below the random-date centre
    assert p < 0.05


# ---- the two theses on the synthetic core ---------------------------------
def test_engine_recovers_dip_only_when_planted(planted, null):
    """Spine #1: with a planted dip the engine sees a significant negative event-day return;
    on the null it is indistinguishable from zero (|t| below the bar)."""
    pbars, pev, _ = planted
    nbars, nev, _ = null
    pt = st.hac_tstat(st.event_day_abnormal(pbars, pev, lag=0)["abn"].to_numpy())
    nt = st.hac_tstat(st.event_day_abnormal(nbars, nev, lag=0)["abn"].to_numpy())
    assert pt < -2.0
    assert abs(nt) < 2.0


def test_car_path_dips_after_event_when_planted(planted):
    bars, ev, _ = planted
    path = st.car_path(bars, ev, pre=10, post=20, lag=0)
    assert path.loc[0, "mean"] == pytest.approx(0.0, abs=1e-9)  # rebased at t=0
    # the path is lower after the event than before it (a downgrade sell-off)
    assert path.loc[20, "mean"] < path.loc[-10, "mean"]
    assert path.loc[20, "mean"] < 0

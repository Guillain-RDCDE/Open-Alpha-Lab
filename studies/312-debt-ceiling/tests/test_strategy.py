"""The event-study engine's invariants, the carry tax, and the study's two spines:
(1) the engine recovers a planted vol hump only when one is really there, and the null
gives nothing; (2) the long-vol carry tax mechanically drags a long-vol-into trade
negative on a flat tape (no vol move) — proving 'buy protection into the event' starts
from a hole."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debt_ceiling import data  # noqa: E402
from debt_ceiling import strategy as st  # noqa: E402


# ---- snapping & windows ---------------------------------------------------
def test_snap_to_sessions_finds_on_or_after(planted):
    bars, _, _ = planted
    # a date that falls on a weekend should snap to the next session
    ev = pd.DatetimeIndex([bars.index[100]])
    locs = st.snap_to_sessions(bars, ev, lag=0)
    assert locs.tolist() == [100]
    locs_lag = st.snap_to_sessions(bars, ev, lag=1)
    assert locs_lag.tolist() == [101]


def test_pre_ramp_columns_and_sign(planted):
    bars, ev, _ = planted
    pr = st.pre_ramp(bars, ev, pre=20)
    assert list(pr.columns) == ["event_date", "vix_start", "vix_end", "d_level", "d_log"]
    assert len(pr) > 0
    # on the planted tape vol rises into the deadline → mostly positive d_log
    assert (pr["d_log"] > 0).mean() > 0.7


def test_post_collapse_columns_and_sign(planted):
    bars, ev, _ = planted
    pc = st.post_collapse(bars, ev, post=20)
    assert list(pc.columns) == ["event_date", "vix_start", "vix_end", "d_level", "d_log"]
    # on the planted tape vol falls after the deadline → mostly negative d_log
    assert (pc["d_log"] < 0).mean() > 0.7


def test_window_log_changes_pre_vs_post(planted):
    bars, ev, _ = planted
    locs = st.snap_to_sessions(bars, ev)
    pre = st.window_log_changes(bars, locs, 20, "pre")
    post = st.window_log_changes(bars, locs, 20, "post")
    assert pre.size > 0 and post.size > 0
    assert pre.mean() > 0 > post.mean()


# ---- the carry tax (spine #2) ---------------------------------------------
def test_round_trip_columns(planted):
    bars, ev, _ = planted
    rt = st.vol_round_trip(bars, ev, pre=20, post=20, daily_carry_bps=25.0, cost_bps=5.0)
    assert list(rt.columns) == [
        "event_date", "pnl_long", "pnl_short", "pnl_gross", "pnl_net"
    ]
    assert np.allclose(rt["pnl_gross"], rt["pnl_long"] + rt["pnl_short"])
    assert np.allclose(rt["pnl_net"], rt["pnl_gross"] - 3.0 * 5.0 * 1e-4)


def test_carry_tax_drags_long_vol_on_flat_tape():
    """Spine #2: with NO planted vol move, the carry tax alone makes a long-vol-into leg
    lose — the structural cost of buying protection into an event."""
    bars, ev, _ = data.synthetic_vix(event_effect=0.0, n_events=40, seed=314)
    # zero carry: the long leg is ~ a coin (no edge, no drag)
    rt0 = st.vol_round_trip(bars, ev, pre=20, post=20, daily_carry_bps=0.0)
    # real carry: the long-vol leg is dragged systematically negative
    rtc = st.vol_round_trip(bars, ev, pre=20, post=20, daily_carry_bps=25.0)
    assert rtc["pnl_long"].mean() < rt0["pnl_long"].mean()
    # 25 bps/day over 20 days = ~5% of NAV of pure drag on the long leg
    drag = rt0["pnl_long"].mean() - rtc["pnl_long"].mean()
    assert abs(drag - 0.05) < 1e-6


def test_cost_monotonically_lowers_net(planted):
    bars, ev, _ = planted
    means = [
        st.summarize(
            st.vol_round_trip(bars, ev, cost_bps=c)["pnl_net"].to_numpy()
        )["mean"]
        for c in (0.0, 5.0, 10.0)
    ]
    assert means[0] > means[1] > means[2]


# ---- inference invariants -------------------------------------------------
def test_hac_tstat_finite_on_planted(planted):
    bars, ev, _ = planted
    t = st.hac_tstat(st.pre_ramp(bars, ev, pre=20)["d_log"].to_numpy())
    assert np.isfinite(t) and t > 2.0  # planted hump clears the bar


def test_block_bootstrap_ci_brackets_mean(planted):
    bars, ev, _ = planted
    x = st.pre_ramp(bars, ev, pre=20)["d_log"].to_numpy()
    lo, hi = st.block_bootstrap_ci(x, n_boot=2000, seed=1)
    assert lo < x.mean() < hi


def test_permutation_pvalue_small_when_event_differs():
    """A planted event distribution far from the random-date null gives a small p."""
    bars, ev, _ = data.synthetic_vix(event_effect=0.5, n_events=40, seed=312)
    nev = data.synthetic_null_events(bars, n_events=400, window=20, seed=3)
    locs_null = st.snap_to_sessions(bars, nev)
    event_pre = st.pre_ramp(bars, ev, pre=20)["d_log"].to_numpy()
    null_pre = st.window_log_changes(bars, locs_null, 20, "pre")
    obs, p = st.permutation_pvalue(event_pre, null_pre, n_perm=5000, seed=1)
    assert obs > 0  # event vol ramp exceeds random dates
    assert p < 0.05


# ---- the two theses on the synthetic core ---------------------------------
def test_engine_recovers_hump_only_when_planted(planted, null):
    """Spine #1: with a planted hump the engine sees a significant pre-ramp; on the null
    it is indistinguishable from zero (|t| below the bar)."""
    pbars, pev, _ = planted
    nbars, nev, _ = null
    pt = st.hac_tstat(st.pre_ramp(pbars, pev, pre=20)["d_log"].to_numpy())
    nt = st.hac_tstat(st.pre_ramp(nbars, nev, pre=20)["d_log"].to_numpy())
    assert pt > 2.0
    assert abs(nt) < 2.0


def test_vix_path_is_hump_shaped_when_planted(planted):
    bars, ev, _ = planted
    path = st.vix_path(bars, ev, pre=20, post=20)
    assert path.loc[0, "mean"] == pytest.approx(1.0, abs=1e-9)  # rebased at t=0
    # peak near t=0, lower in the wings (a hump)
    assert path.loc[0, "mean"] > path.loc[-20, "mean"]
    assert path.loc[0, "mean"] > path.loc[20, "mean"]

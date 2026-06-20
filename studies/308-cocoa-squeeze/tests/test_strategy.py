"""The stretch measure, the two signals, the one-lag forward-return engine with honest
costs, and the study's two spines: (1) the engine finds an edge on a planted blow-off but
*not* on a pure random walk, and (2) costs and the single execution lag behave exactly as
documented."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cocoa_squeeze import strategy as st  # noqa: E402


# ---- the stretch measure --------------------------------------------------
def test_stretch_is_high_during_the_planted_blowoff(blowoff):
    bars, truth = blowoff
    z = st.stretch_z(bars["close"])
    peak = truth["peak_idx"]
    # Somewhere in the blow-off run-up window the stretch hits its top-decile extreme.
    window = z.iloc[peak - 400: peak + 40].max()
    assert window >= np.nanquantile(z.to_numpy(), 0.90)


def test_stretch_flat_on_random_walk(random_walk):
    """On a pure random walk the stretch has no persistent regime — it crosses zero often."""
    bars, _ = random_walk
    z = st.stretch_z(bars["close"]).dropna()
    # Sign flips frequently: not a one-way blow-off.
    flips = (np.sign(z.to_numpy()[1:]) != np.sign(z.to_numpy()[:-1])).mean()
    assert flips > 0.01


# ---- the signals ----------------------------------------------------------
def test_momentum_signal_is_long_only(blowoff):
    bars, _ = blowoff
    pos = st.momentum_signal(bars["close"])
    assert set(np.unique(pos.to_numpy())) <= {0.0, 1.0}
    assert pos.sum() > 0  # it fires during the blow-off


def test_reversion_signal_is_short_only(blowoff):
    bars, _ = blowoff
    pos = st.reversion_signal(bars["close"])
    assert set(np.unique(pos.to_numpy())) <= {0.0, -1.0}
    assert (pos < 0).sum() > 0


def test_random_sign_matches_active_bars(blowoff):
    bars, _ = blowoff
    pos = st.momentum_signal(bars["close"])
    ctrl = st.random_sign_positions(pos, seed=1)
    # The control is active on exactly the same bars as the template.
    assert ((ctrl != 0) == (pos != 0)).all()
    assert set(np.unique(ctrl[ctrl != 0].to_numpy())) <= {-1.0, 1.0}


def test_random_sign_reproducible(blowoff):
    bars, _ = blowoff
    pos = st.momentum_signal(bars["close"])
    a = st.random_sign_positions(pos, seed=3)
    b = st.random_sign_positions(pos, seed=3)
    assert np.array_equal(a.to_numpy(), b.to_numpy())


# ---- the forward-return engine: one lag, honest costs ----------------------
def test_single_execution_lag(blowoff):
    """The position at the close of t earns the return of t+1 — exactly one shift."""
    bars, _ = blowoff
    close = bars["close"]
    pos = pd.Series(0.0, index=close.index)
    pos.iloc[10] = 1.0  # long only on bar 10
    r = st.book_returns(close, pos, cost_bps=0.0)
    # The single long day's return is realised on bar 11, not bar 10.
    expected = close.pct_change().iloc[11]
    assert np.isclose(r.iloc[11], expected)
    assert np.isclose(r.iloc[10], 0.0)


def test_cost_charged_one_way_on_turnover(blowoff):
    bars, _ = blowoff
    close = bars["close"]
    pos = pd.Series(0.0, index=close.index)
    pos.iloc[5:8] = 1.0          # one entry (flat->1) and one exit (1->flat)
    r0 = st.book_returns(close, pos, cost_bps=0.0)
    r1 = st.book_returns(close, pos, cost_bps=10.0)
    drag = (r0 - r1).sum()
    # Two one-way turnover events of size 1 at 10 bps each = 20 bps total.
    assert np.isclose(drag, 2 * 10e-4, atol=1e-9)


def test_shorts_pay_borrow(blowoff):
    bars, _ = blowoff
    close = bars["close"]
    short = pd.Series(-1.0, index=close.index)
    flat = pd.Series(0.0, index=close.index)
    r_short = st.book_returns(close, short, cost_bps=0.0, borrow_bps_yr=252.0)
    r_short0 = st.book_returns(close, short, cost_bps=0.0, borrow_bps_yr=0.0)
    # Borrow drag is ~1 bp/day on the held short (after the lag).
    drag = (r_short0 - r_short).mean()
    assert drag > 0


def test_cost_monotonically_lowers_mean(blowoff):
    bars, _ = blowoff
    pos = st.momentum_signal(bars["close"])
    means = [
        st.book_returns(bars["close"], pos, cost_bps=c).mean()
        for c in (0.0, 5.0, 20.0)
    ]
    assert means[0] >= means[1] >= means[2]


# ---- honest statistics ----------------------------------------------------
def test_hac_tstat_zero_on_pure_noise():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 3000)
    assert abs(st.hac_tstat(r)) < 2.0


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 3000)
    lo, hi = st.block_bootstrap_ci(r, n_boot=500)
    assert lo < r.mean() < hi


# ---- the two spines --------------------------------------------------------
def test_momentum_pays_on_the_planted_blowoff_not_on_the_null(blowoff, random_walk):
    """Spine #1: riding the parabola (long while stretched & rising) clears the bar on the
    planted bubble; on a pure random walk the same timer is indistinguishable from zero."""
    bb, _ = blowoff
    rw, _ = random_walk
    mom_bb = st.book_returns(bb["close"], st.momentum_signal(bb["close"]))
    mom_rw = st.book_returns(rw["close"], st.momentum_signal(rw["close"]))
    t_bb = st.hac_tstat(mom_bb[mom_bb != 0].to_numpy())
    t_rw = st.hac_tstat(mom_rw[mom_rw != 0].to_numpy())
    assert t_bb > 2.0            # riding the planted parabola pays, robustly
    assert abs(t_rw) < 2.0       # nothing to ride on the null


def test_post_peak_crack_is_a_negative_forward_return_on_the_blowoff(blowoff):
    """Spine #2: the planted parabola really *does* crack — forward returns from the price
    peak are negative — so mean-reversion exists in the tape (even if hard to time)."""
    bb, _ = blowoff
    info = st.find_blowoff_peak(bb["close"])
    assert info["fwd_drawdown"] < -0.10   # it gives back a chunk after the top


def test_random_direction_is_not_a_robust_edge_on_null(random_walk):
    """A random-sign timer on the same bars carries no robust edge on the null tape —
    timing alone is not direction."""
    rw, _ = random_walk
    tmpl = st.momentum_signal(rw["close"])
    ctrl = st.random_sign_positions(tmpl, seed=7)
    r = st.book_returns(rw["close"], ctrl)
    active = r[r != 0].to_numpy()
    if active.size > 5:
        assert abs(st.hac_tstat(active)) < 3.0


def test_find_blowoff_peak_locates_the_parabola(blowoff):
    bars, truth = blowoff
    info = st.find_blowoff_peak(bars["close"])
    assert info["runup_mult"] > 1.5                # a real run-up
    assert info["fwd_drawdown"] < 0.0              # it cracks afterwards


def test_crack_signal_is_short_only(blowoff):
    bars, _ = blowoff
    pos = st.crack_signal(bars["close"])
    assert set(np.unique(pos.to_numpy())) <= {0.0, -1.0}

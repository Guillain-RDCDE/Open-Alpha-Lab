"""The engine's invariants, and the study's two spines: (1) buy-and-hold's excess-of-cash
edge is detectable only when a real drift is planted, and (2) a momentum overlay beats
naive buy-and-hold only when real persistence is present — on a martingale it does not."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carbon_credits import data  # noqa: E402
from carbon_credits import strategy as st  # noqa: E402


# ---- return helpers --------------------------------------------------------
def test_to_returns_drops_first_row(null_tape):
    frame, _ = null_tape
    R = st.to_returns(frame)
    assert len(R) == len(frame) - 1
    assert list(R.columns) == ["KRBN", "CASH"]


def test_buy_and_hold_is_the_asset_return(null_tape):
    frame, _ = null_tape
    R = st.to_returns(frame)
    bh = st.buy_and_hold(R, "KRBN")
    assert np.allclose(bh.to_numpy(), R["KRBN"].to_numpy())


# ---- the momentum signal & overlay -----------------------------------------
def test_momentum_signal_is_boolean_and_no_lookahead(trend_tape):
    """The signal is boolean and uses only trailing data (warmup is False)."""
    frame, _ = trend_tape
    sig = st.momentum_signal(frame, "KRBN", lookback=63)
    assert sig.dtype == bool
    assert not sig.iloc[:63].any()  # warmup window cannot fire


def test_overlay_has_single_execution_lag(null_tape):
    """The position held over t is the signal from t-1 — exactly one shift."""
    frame, _ = null_tape
    R = st.to_returns(frame)
    sig = st.momentum_signal(frame, "KRBN", lookback=63).reindex(R.index).fillna(False)
    net = st.overlay_returns(R, sig, "KRBN", "CASH", cost_bps=0.0)
    held = sig.astype(float).shift(1).fillna(0.0)
    expected = held * R["KRBN"] + (1.0 - held) * R["CASH"]
    assert np.allclose(net.to_numpy(), expected.to_numpy())


def test_overlay_cost_monotonically_lowers_return(trend_tape):
    """More cost per flip can only reduce the overlay's mean return."""
    frame, _ = trend_tape
    R = st.to_returns(frame)
    sig = st.momentum_signal(frame, "KRBN", lookback=63)
    means = [st.overlay_returns(R, sig, "KRBN", "CASH", cost_bps=c).mean()
             for c in (0.0, 5.0, 10.0)]
    assert means[0] > means[1] > means[2]


def test_overlay_no_flip_no_cost(null_tape):
    """An always-true signal never flips after entry, so cost barely bites (one entry)."""
    frame, _ = null_tape
    R = st.to_returns(frame)
    always = pd.Series(True, index=R.index)
    net0 = st.overlay_returns(R, always, "KRBN", "CASH", cost_bps=0.0)
    net5 = st.overlay_returns(R, always, "KRBN", "CASH", cost_bps=5.0)
    # only the very first day carries the initial entry turnover
    assert (net0 - net5).abs().sum() < 6e-4


# ---- stats -----------------------------------------------------------------
def test_stats_keys_and_excess_sharpe(drift_tape):
    frame, _ = drift_tape
    R = st.to_returns(frame)
    s = st.stats(st.buy_and_hold(R), R["CASH"])
    assert set(s) >= {"cagr", "vol", "sharpe", "max_dd", "final", "equity", "net"}
    assert s["max_dd"] <= 0.0
    assert s["final"] > 0.0


def test_hac_tstat_matches_ols_on_iid_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(0.01, 1.0, 5000)
    t_hac = st.hac_tstat(x)
    t_ols = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(t_hac - t_ols) < 0.5  # HAC ≈ OLS when there is no autocorrelation


# ---- the two spines --------------------------------------------------------
def test_buyhold_edge_real_only_with_drift(null_tape, drift_tape):
    """Spine #1: the excess-of-cash HAC t clears the bar only when a drift is planted."""
    Rn = st.to_returns(null_tape[0])
    Rd = st.to_returns(drift_tape[0])
    t_null = st.excess_tstat(st.buy_and_hold(Rn), Rn["CASH"])
    t_drift = st.excess_tstat(st.buy_and_hold(Rd), Rd["CASH"])
    assert abs(t_null) < 2.0      # no planted reward → not distinguishable from cash
    assert t_drift > 2.0          # planted reward → clears the inference bar


def test_overlay_beats_buyhold_only_with_persistence(trend_tape, null_tape):
    """Spine #2: the timing overlay beats naive buy-and-hold (excess-vs-excess Sharpe) only
    when real persistence is present; on a martingale it does not reliably win."""
    # Trend tape: overlay should win clearly.
    Rt = st.to_returns(trend_tape[0])
    sig_t = st.momentum_signal(trend_tape[0], "KRBN", lookback=63)
    ov_t = st.overlay_returns(Rt, sig_t, "KRBN", "CASH", cost_bps=1.0)
    d_t = st.bootstrap_sharpe_diff(ov_t, st.buy_and_hold(Rt), Rt["CASH"], n_boot=800)
    assert d_t["point"] > 0.0 and d_t["frac_a_wins"] > 0.8

    # Null tape: averaged over seeds the overlay's edge is ~0 (a coin).
    pts = []
    for sd in range(304, 314):
        f, _ = data.synthetic_carbon(n_days=3000, mu=0.0, momentum=0.0, seed=sd)
        R = st.to_returns(f)
        sig = st.momentum_signal(f, "KRBN", lookback=63)
        ov = st.overlay_returns(R, sig, "KRBN", "CASH", cost_bps=1.0)
        pts.append(st.bootstrap_sharpe_diff(ov, st.buy_and_hold(R), R["CASH"],
                                            n_boot=300)["point"])
    assert abs(np.mean(pts)) < 0.20  # no planted persistence → no real timing edge


def test_bootstrap_sharpe_diff_is_seeded_and_bounded(trend_tape):
    Rt = st.to_returns(trend_tape[0])
    bh = st.buy_and_hold(Rt)
    sig = st.momentum_signal(trend_tape[0], "KRBN", lookback=63)
    ov = st.overlay_returns(Rt, sig, "KRBN", "CASH", cost_bps=1.0)
    d1 = st.bootstrap_sharpe_diff(ov, bh, Rt["CASH"], n_boot=500, seed=1)
    d2 = st.bootstrap_sharpe_diff(ov, bh, Rt["CASH"], n_boot=500, seed=1)
    assert d1["point"] == d2["point"]
    assert d1["ci95"][0] <= d1["point"] <= d1["ci95"][1]
    assert 0.0 <= d1["frac_a_wins"] <= 1.0

"""Signals, the forward-return engine's invariants, and the study's two spines on the
synthetic control: (1) when the edge is a confound BOTH signals share, range-position does
NOT beat the high ratio; (2) when the edge lives in the second anchor (the low), it MUST.
And the null: with no edge, neither signal clears the bar."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fifty_two_week_range import strategy as st  # noqa: E402


# ---- signals --------------------------------------------------------------
def test_range_position_bounded(null_panel):
    panel, _ = null_panel
    pos = st.range_position(panel).dropna()
    assert (pos.to_numpy() >= 0.0).all()
    assert (pos.to_numpy() <= 1.0).all()


def test_high_ratio_bounded(null_panel):
    panel, _ = null_panel
    hr = st.high_ratio(panel).dropna()
    assert (hr.to_numpy() >= 0.0).all()
    assert (hr.to_numpy() <= 1.0).all()


def test_range_position_endpoints():
    """At the rolling high pos==1; at the rolling low pos==0."""
    idx = pd.date_range("2016-01-01", periods=300, freq="B")
    # A series that ends exactly at its 252-day high.
    rising = pd.Series(np.linspace(100, 200, 300), index=idx).to_frame("X")
    pos = st.range_position(rising).iloc[-1, 0]
    assert abs(pos - 1.0) < 1e-6
    falling = pd.Series(np.linspace(200, 100, 300), index=idx).to_frame("X")
    pos_f = st.range_position(falling).iloc[-1, 0]
    assert abs(pos_f - 0.0) < 1e-6


def test_forward_returns_no_lookahead(null_panel):
    """fwd_ret at t uses close[t+hold]; the last `hold` rows are NaN."""
    panel, _ = null_panel
    fwd = st.forward_returns(panel, hold_days=5)
    assert fwd.iloc[-5:].isna().all().all()
    # Exact identity on one cell.
    expected = np.log(panel.iloc[5, 0] / panel.iloc[0, 0])
    assert abs(fwd.iloc[0, 0] - expected) < 1e-12


# ---- engine invariants ----------------------------------------------------
def test_quintile_spread_columns_and_net(null_panel):
    panel, _ = null_panel
    pos = st.range_position(panel)
    fwd = st.forward_returns(panel, 5)
    out = st.quintile_spread(pos, fwd, cost_bps=10.0)
    for k in ("Q1", "Q5", "spread_gross", "spread_net"):
        assert k in out
    # net = gross - cost, applied once.
    d = (out["spread_gross"] - out["spread_net"]).dropna()
    assert np.allclose(d.to_numpy(), 10.0e-4)


def test_cost_monotonically_lowers_spread(null_panel):
    panel, _ = null_panel
    pos = st.range_position(panel)
    fwd = st.forward_returns(panel, 5)
    means = [
        st.summarize(st.quintile_spread(pos, fwd, cost_bps=c)["spread_net"])["mean_bps"]
        for c in (0.0, 5.0, 10.0)
    ]
    assert means[0] > means[1] > means[2]


def test_diff_tstat_paired():
    a = pd.Series(np.arange(50.0), index=pd.date_range("2020-01-01", periods=50))
    b = a - 1.0
    out = st.diff_tstat(a, b)
    assert abs(out["mean_diff_bps"] - 1.0 * 1e4) < 1e-6


# ---- the theses on the synthetic control ----------------------------------
def _race(panel, hold=5):
    pos = st.range_position(panel)
    hr = st.high_ratio(panel)
    fwd = st.forward_returns(panel, hold)
    qp = st.quintile_spread(pos, fwd)
    qh = st.quintile_spread(hr, fwd)
    diff = st.diff_tstat(qp["spread_gross"], qh["spread_gross"])
    return st.summarize(qp["spread_gross"]), st.summarize(qh["spread_gross"]), diff


def test_null_neither_signal_clears_bar(null_panel):
    """Spine #0 — with no planted edge, neither spread is significant and they don't
    separate from each other."""
    panel, _ = null_panel
    sp, sh, diff = _race(panel)
    assert abs(sp["tstat"]) < 2.0
    assert abs(sh["tstat"]) < 2.0
    assert abs(diff["tstat"]) < 2.0


def test_confound_does_not_let_range_beat_high(confound_panel):
    """Spine #1 — a confound both signals capture lifts BOTH spreads but does NOT let
    range-position beat the high ratio (the second anchor adds nothing)."""
    panel, _ = confound_panel
    sp, sh, diff = _race(panel)
    assert sp["tstat"] > 2.0          # range-position works...
    assert sh["tstat"] > 2.0          # ...and so does the high ratio
    assert abs(diff["tstat"]) < 2.0   # but neither beats the other


def test_discriminating_edge_lets_range_beat_high(discriminating_panel):
    """Spine #2 — when the edge lives in the second anchor (the low), range-position
    beats the high ratio in the paired difference and the spanning regression."""
    panel, _ = discriminating_panel
    sp, sh, diff = _race(panel)
    assert diff["tstat"] > 2.0        # range-position wins the horse race
    assert diff["mean_diff_bps"] > 0  # ...in the right direction
    # Spanning regression: incremental info of pos over hr is positive & significant.
    pos = st.range_position(panel)
    hr = st.high_ratio(panel)
    fwd = st.forward_returns(panel, 5)
    sr = st.spanning_regression(pos, hr, fwd)
    assert st.hac_tstat(sr["b_target"]) > 2.0


def test_bootstrap_ci_brackets_mean(discriminating_panel):
    panel, _ = discriminating_panel
    pos = st.range_position(panel)
    fwd = st.forward_returns(panel, 5)
    spread = st.quintile_spread(pos, fwd)["spread_gross"].dropna()
    lo, hi = st.block_bootstrap_ci(spread, n_boot=500, seed=1)
    mean_bps = spread.mean() * 1e4
    assert lo <= mean_bps <= hi

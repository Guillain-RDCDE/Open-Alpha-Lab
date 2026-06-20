"""The recovery-speed engine's invariants, the shuffle-null control, and the study's
spine: the long-short book banks the planted recovery-momentum edge ONLY when it is
real, and is a coin on the null."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recovery_speed import data, strategy as st  # noqa: E402


def _event(panel, market):
    ev = st.detect_drawdown(market, min_depth=0.10)
    assert ev is not None, "the planted drawdown must be detectable"
    return ev


# ---- drawdown detection ---------------------------------------------------
def test_detect_drawdown_orders_peak_trough_recover(null_panel):
    panel, market, truth = null_panel
    ev = _event(panel, market)
    assert ev["peak"] < ev["trough"] < ev["recover"]
    assert ev["depth"] >= 0.10


def test_detect_drawdown_none_when_too_shallow():
    # An almost-flat market: no drawdown reaches a stiff threshold.
    _, market, _ = data.synthetic_panel(n_names=10, n_days=400, drawdown_depth=0.02,
                                        market_vol=0.002, idio_vol=0.002, seed=1)
    assert st.detect_drawdown(market, min_depth=0.40) is None


# ---- recovery-speed score -------------------------------------------------
def test_recovery_speed_finite_for_fallen_names(control_panel):
    panel, market, _ = control_panel
    ev = _event(panel, market)
    sp = st.recovery_speed(panel, ev["peak"], ev["trough"], ev["recover"]).dropna()
    assert len(sp) > 0
    assert np.isfinite(sp.to_numpy()).all()


def test_deciles_split_disjoint(control_panel):
    panel, market, _ = control_panel
    ev = _event(panel, market)
    sp = st.recovery_speed(panel, ev["peak"], ev["trough"], ev["recover"])
    top, bottom = st.deciles(sp, q=5)
    assert len(top) > 0 and len(bottom) > 0
    assert set(top).isdisjoint(set(bottom))


# ---- event engine invariants ----------------------------------------------
def test_run_event_one_lag_and_costs(control_panel):
    panel, market, _ = control_panel
    ev = _event(panel, market)
    gross = st.run_event(panel, market, ev, cost_bps=0.0)["ls_net"]
    net = st.run_event(panel, market, ev, cost_bps=10.0, borrow_bps_ann=50.0)["ls_net"]
    assert net < gross  # costs + borrow only ever reduce the net spread


def test_run_event_market_neutral_differs(control_panel):
    panel, market, _ = control_panel
    ev = _event(panel, market)
    raw = st.run_event(panel, market, ev, market_neutral=False)
    mn = st.run_event(panel, market, ev, market_neutral=True)
    # The long-short spread is similar (market cancels), but leg means shift by mkt fwd.
    assert raw["n_long"] == mn["n_long"]
    assert not np.isclose(raw["long_mean"], mn["long_mean"])


# ---- the shuffle-null control & the spine ----------------------------------
def test_shuffle_destroys_edge_on_control(control_panel):
    """A shuffled ranking should not systematically beat the market — the edge lives in
    the *identity* link between recovery speed and forward return."""
    panel, market, _ = control_panel
    ev = _event(panel, market)
    real = st.run_event(panel, market, ev)["ls_gross"]
    shuffled = np.mean([
        st.run_event(panel, market, ev, shuffle_seed=s)["ls_gross"]
        for s in range(30)
    ])
    assert real > shuffled


def test_rule_banks_edge_only_when_real(null_panel, control_panel):
    """Spine: on the positive-control panel the recovery-speed long-short spread sits in
    the right tail of the shuffled null (low p, t>2); on the null panel it does not."""
    np_panel, np_mkt, _ = null_panel
    cp_panel, cp_mkt, _ = control_panel

    ev_null = _event(np_panel, np_mkt)
    ev_ctrl = _event(cp_panel, cp_mkt)

    ctrl = st.edge_vs_shuffle(cp_panel, cp_mkt, ev_ctrl, n_shuffle=200, seed=1)
    null = st.edge_vs_shuffle(np_panel, np_mkt, ev_null, n_shuffle=200, seed=1)

    # Positive control: real spread positive, in the right tail, t clears the bar.
    assert ctrl["real_ls"] > 0
    assert ctrl["p_value"] < 0.05
    assert ctrl["spread_t"] > 2.0
    # Null: recovery speed predicts nothing — not a right-tail event, t below the bar.
    assert null["p_value"] > 0.10
    assert abs(null["spread_t"]) < 2.0


def test_hac_tstat_sane():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 500)
    assert abs(st.hac_tstat(z)) < 3.0          # mean-zero noise: t near 0
    pos = rng.normal(0.5, 1, 500)
    assert st.hac_tstat(pos) > 5.0             # clear positive mean: large t


def test_block_bootstrap_ci_brackets_point(control_panel):
    panel, market, _ = control_panel
    ev = _event(panel, market)
    res = st.run_event(panel, market, ev)
    lo, hi = st.block_bootstrap_ci(res, n_boot=500, seed=1)
    assert lo <= res["ls_gross"] <= hi

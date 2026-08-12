"""Offline, fixed-seed tests for the 90/10 cash + call machinery.

The engine is point-in-time (the option on day t uses only state known at t-1); Black-Scholes is a
sane call price (monotone in spot, convex, delta in [0,1], put-call parity); the notional cap keeps
the effective equity exposure from levering past full capital; the book de-risks as vol rises and
protects capital on a bear tape; a steady bull gives up upside to the premium bleed; costs reduce
the net. All offline and deterministic.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_call import strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Black–Scholes call — sanity of the marking model
# --------------------------------------------------------------------------- #
def test_bs_call_intrinsic_at_expiry():
    assert abs(float(st.bs_call(110.0, 100.0, 0.0, 0.2, 0.03)) - 10.0) < 1e-9
    assert abs(float(st.bs_call(90.0, 100.0, 0.0, 0.2, 0.03)) - 0.0) < 1e-9


def test_bs_call_monotone_and_bounded_by_spot():
    lo = float(st.bs_call(100.0, 100.0, 1.0, 0.2, 0.03))
    hi = float(st.bs_call(120.0, 100.0, 1.0, 0.2, 0.03))
    assert 0.0 < lo < hi                                 # more spot -> more call value
    assert hi < 120.0                                    # never worth more than the underlying


def test_bs_call_rises_with_vol():
    cheap = float(st.bs_call(100.0, 100.0, 1.0, 0.10, 0.03))
    dear = float(st.bs_call(100.0, 100.0, 1.0, 0.30, 0.03))
    assert dear > cheap                                  # vega > 0


def test_bs_delta_in_unit_interval_and_atm_near_half():
    d_atm = float(st.bs_call_delta(100.0, 100.0, 1.0, 0.2, 0.0))
    assert 0.0 <= d_atm <= 1.0
    assert 0.45 < d_atm < 0.65                           # ATM call delta ~ 0.5 (a touch above at r=0)
    assert float(st.bs_call_delta(200.0, 100.0, 0.5, 0.2, 0.03)) > 0.95   # deep ITM -> ~1


# --------------------------------------------------------------------------- #
# The 90/10 engine — no look-ahead, notional cap, de-risking
# --------------------------------------------------------------------------- #
def _arrays(frame):
    ret = st.to_returns(frame)
    spy = frame["SPY"].reindex(ret.index).to_numpy()
    return (spy, ret["SPY"].to_numpy(), ret["BIL"].to_numpy(),
            frame["IRX"].reindex(ret.index).to_numpy())


def test_no_lookahead_future_return_does_not_change_past_returns(bear_world):
    frame, _ = bear_world
    spy, rr, cr, irx = _arrays(frame)
    base = st.ninety_ten_book(spy, rr, cr, irx)["net"]
    spy2 = spy.copy(); spy2[-1] *= 1.5                   # perturb only the LAST day's spot
    pert = st.ninety_ten_book(spy2, rr, cr, irx)["net"]
    assert np.allclose(base[:-1], pert[:-1])             # earlier net returns untouched


def test_notional_cap_prevents_leverage_in_calm_markets(calm_world):
    # With cover capped at 100% of NAV the effective equity (delta) weight never levers up.
    frame, _ = calm_world
    spy, rr, cr, irx = _arrays(frame)
    book = st.ninety_ten_book(spy, rr, cr, irx, notional_cap=1.0)
    # delta-weight can only exceed 1 if calls go deep ITM after a big rally; on the calm tape the
    # average stays comfortably below full exposure (it is an insured book, not a levered one).
    assert book["avg_weight"] < 1.0
    assert np.nanmax(book["dweight"]) < 1.6              # no runaway leverage


def test_rule_derisks_as_vol_rises(bear_world, calm_world):
    fb, _ = bear_world
    fc, _ = calm_world
    wb = st.ninety_ten_book(*_arrays(fb))["avg_weight"]
    wc = st.ninety_ten_book(*_arrays(fc))["avg_weight"]
    assert wb < wc                                       # high-vol bear -> lower equity exposure


def test_bear_world_protects_capital(bear_world):
    frame, _ = bear_world
    d = st.synthetic_detect(frame)
    assert d["dd_protection"] > 0.15                     # 90/10 drawdown much shallower than BH
    assert d["dd_tt"] > d["dd_bh"]                        # (max_dd is negative; tt is closer to 0)


def test_calm_bull_gives_up_upside(calm_world):
    # Nothing to protect on a steady bull, so the premium bleed makes 90/10 lag on total return.
    frame, _ = calm_world
    r = st.race(frame)
    assert r["ninety_ten"]["cagr"] < r["buy_hold"]["cagr"]
    assert r["ninety_ten"]["max_dd"] > r["buy_hold"]["max_dd"] - 0.02  # DD no worse than BH


def test_rolls_happen_annually(bear_world):
    frame, _ = bear_world
    spy, rr, cr, irx = _arrays(frame)
    book = st.ninety_ten_book(spy, rr, cr, irx, roll_days=252)
    expected = (len(spy) - 1) // 252                     # ~ one roll per 252 trading days
    assert abs(book["n_rolls"] - expected) <= 1


# --------------------------------------------------------------------------- #
# The fair-race identity, costs, and the premium/cost sweeps
# --------------------------------------------------------------------------- #
def test_constant_mix_excess_sharpe_equals_risky(calm_world):
    # A constant fraction w of SPY funded from cash has the SAME excess-of-cash Sharpe as SPY.
    frame, _ = calm_world
    r = st.race(frame)
    assert abs(r["static"]["sharpe"] - r["buy_hold"]["sharpe"]) < 1e-9
    assert abs(r["sharpe_vs_bh"] - r["sharpe_vs_static"]) < 1e-9


def test_costs_reduce_net(bear_world):
    frame, _ = bear_world
    spy, rr, cr, irx = _arrays(frame)
    gross = st.ninety_ten_book(spy, rr, cr, irx, cost_bps=0.0)["equity"][-1]
    net = st.ninety_ten_book(spy, rr, cr, irx, cost_bps=50.0)["equity"][-1]
    assert net < gross


def test_premium_markup_lowers_sharpe(calm_world):
    # A richer option (variance risk premium, prem_mult > 1) thins the sleeve and lowers Sharpe.
    frame, _ = calm_world
    sweep = st.premium_sweep(frame, prem_mults=(1.0, 1.5, 2.0))
    s = sweep["tt_sharpe"].to_numpy()
    assert s[0] >= s[1] >= s[2]                           # dearer options -> weakly lower Sharpe


def test_cost_sweep_sharpe_monotone_nonincreasing(bear_world):
    # Annual-roll turnover is tiny, so costs barely move Sharpe; assert non-increasing within a
    # small FP tolerance (the economically meaningful "costs reduce net" is pinned separately).
    frame, _ = bear_world
    sweep = st.cost_sweep(frame, one_way_bps=(0.0, 25.0, 100.0))
    s = sweep["tt_sharpe"].to_numpy()
    assert s[0] >= s[1] - 1e-9 and s[1] >= s[2] - 1e-9

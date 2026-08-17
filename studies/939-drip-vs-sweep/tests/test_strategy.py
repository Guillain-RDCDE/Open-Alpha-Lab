"""Simulator invariants, the execution lag, and the study's spine — all offline.

The spine: on a planted tape (a fat premium over cash) the DRIP arm beats the sweep
arm by a positive, seed-stable margin that is **monotone in the distribution yield**
and **larger when the sweep waits a year rather than a quarter**; on the null tape
(the fund drifts at the cash rate) the gap collapses to zero. Costs and the execution
lag behave the way the honesty rules require.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drip_sweep import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #
def test_sweep_dates_are_period_ends():
    idx = pd.bdate_range("2018-01-01", "2021-12-31")
    q = st.sweep_dates(idx, "Q")
    a = st.sweep_dates(idx, "A")
    assert len(q) == 16 and len(a) == 4
    assert idx[q[0]].month == 3 and idx[a[0]].month == 12
    assert list(q) == sorted(q) and list(a) == sorted(a)


def test_pay_index_lands_on_or_after_the_lag():
    idx = pd.bdate_range("2020-01-02", periods=300)
    ex = np.array([10, 100, 200])
    m = st._pay_index(idx, ex, pay_lag_days=30)
    for i in ex:
        assert idx[m[i]] >= idx[i] + pd.Timedelta(days=30)
        assert idx[m[i]] < idx[i] + pd.Timedelta(days=37)


def test_pay_index_zero_lag_is_same_day():
    idx = pd.bdate_range("2020-01-02", periods=100)
    m = st._pay_index(idx, np.array([5, 50]), pay_lag_days=0)
    assert m == {5: 5, 50: 50}


# --------------------------------------------------------------------------- #
# Simulator invariants
# --------------------------------------------------------------------------- #
def test_simulate_columns_and_positivity(planted):
    frame, _ = planted
    sim = st.simulate(frame["close"], frame["dividend"], frame["cash"], policy="drip")
    assert {"wealth", "shares", "cash_interest", "float_cash",
            "n_trades", "cost_paid"}.issubset(sim.columns)
    assert (sim["wealth"] > 0).all()
    assert (sim["shares"] > 0).all()
    assert not sim.isna().any().any()


def test_simulate_rejects_unknown_policy(planted):
    frame, _ = planted
    with pytest.raises(ValueError):
        st.simulate(frame["close"], frame["dividend"], frame["cash"], policy="hodl")


def test_drip_never_holds_interest_bearing_cash(planted):
    frame, _ = planted
    sim = st.simulate(frame["close"], frame["dividend"], frame["cash"], policy="drip")
    assert float(sim["cash_interest"].abs().max()) == 0.0


def test_sweep_does_hold_interest_bearing_cash(planted):
    frame, _ = planted
    sim = st.simulate(frame["close"], frame["dividend"], frame["cash"],
                      policy="sweep", sweep_freq="A")
    assert float(sim["cash_interest"].max()) > 0.0


def test_share_count_is_non_decreasing(planted):
    """Neither arm ever sells: shares only ever go up (no shorting, no borrow)."""
    frame, _ = planted
    for policy in ("drip", "sweep"):
        sim = st.simulate(frame["close"], frame["dividend"], frame["cash"], policy=policy)
        assert (sim["shares"].diff().dropna() >= -1e-12).all()


def test_no_dividends_means_no_trades_and_identical_arms(planted):
    """With the dividend stream zeroed, the two policies are the same portfolio."""
    frame, _ = planted
    zero = pd.Series(0.0, index=frame.index)
    a = st.simulate(frame["close"], zero, frame["cash"], policy="drip")
    b = st.simulate(frame["close"], zero, frame["cash"], policy="sweep")
    assert int(a["n_trades"].iloc[-1]) == int(b["n_trades"].iloc[-1]) == 0
    assert np.allclose(a["wealth"].to_numpy(), b["wealth"].to_numpy())


def test_zero_lag_zero_cost_drip_reproduces_total_return(planted):
    """The study's single most important audit, on a tape where truth is known."""
    frame, _ = planted
    r = (frame["close"] + frame["dividend"]) / frame["close"].shift(1) - 1.0
    tr = (1.0 + r.fillna(0.0)).cumprod() * 100.0
    a = st.drip_tracks_total_return(frame["close"], frame["dividend"], tr)
    # The only residual is the single execution lag on each reinvestment.
    assert abs(a["terminal_ratio"] - 1.0) < 0.01
    assert abs(a["ann_tracking_bps"]) < 15.0


# --------------------------------------------------------------------------- #
# The execution lag
# --------------------------------------------------------------------------- #
def test_execution_lag_is_exactly_one_day():
    """A distribution landing on day t must buy shares at day t+1's close, not t's."""
    idx = pd.bdate_range("2020-01-02", periods=60)
    price = pd.Series(100.0, index=idx)
    price.iloc[41:] = 200.0          # the price doubles on bar 41
    divs = pd.Series(0.0, index=idx)
    divs.iloc[40] = 10.0             # goes ex (and, at lag 0, pays) on bar 40
    cash = pd.Series(1.0, index=idx)
    sim = st.simulate(price, divs, cash, policy="drip", pay_lag_days=0, cost_bps=0.0,
                      capital=10_000.0)
    shares0 = 10_000.0 / 100.0
    # No purchase on bar 40 itself; the buy happens on bar 41, at the NEW price 200.
    assert sim["shares"].iloc[40] == pytest.approx(shares0)
    assert sim["shares"].iloc[41] == pytest.approx(shares0 + shares0 * 10.0 / 200.0)
    assert int(sim["n_trades"].iloc[41]) == 1


def test_no_lookahead_future_prices_do_not_move_past_wealth(planted):
    frame, _ = planted
    n = len(frame)
    cut = n - 200
    a = st.simulate(frame["close"], frame["dividend"], frame["cash"], policy="sweep")
    px2 = frame["close"].copy()
    px2.iloc[cut:] *= 3.0            # perturb only the future
    b = st.simulate(px2, frame["dividend"], frame["cash"], policy="sweep")
    assert np.allclose(a["wealth"].to_numpy()[:cut - 1], b["wealth"].to_numpy()[:cut - 1])


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def test_costs_monotonically_reduce_wealth(planted):
    frame, _ = planted
    terminals = []
    for c in (0.0, 5.0, 25.0, 100.0):
        sim = st.simulate(frame["close"], frame["dividend"], frame["cash"],
                          policy="sweep", cost_bps=c)
        terminals.append(float(sim["wealth"].iloc[-1]))
    assert terminals[0] >= terminals[1] >= terminals[2] >= terminals[3]


def test_cost_is_charged_on_the_amount_traded_not_the_portfolio(planted):
    """A 100 bp cost on ~3%/yr of reinvested NAV must cost far less than 100 bp of NAV."""
    frame, _ = planted
    free = st.simulate(frame["close"], frame["dividend"], frame["cash"],
                       policy="drip", cost_bps=0.0)
    dear = st.simulate(frame["close"], frame["dividend"], frame["cash"],
                       policy="drip", cost_bps=100.0)
    drag = 1.0 - float(dear["wealth"].iloc[-1] / free["wealth"].iloc[-1])
    n_years = len(frame) / 252
    assert 0.0 < drag < 0.01 * n_years * 0.10


def test_raising_only_the_sweep_cost_widens_the_gap(planted):
    frame, _ = planted
    cheap = st.race(frame["close"], frame["dividend"], frame["cash"], sweep_cost_bps=0.0)
    dear = st.race(frame["close"], frame["dividend"], frame["cash"], sweep_cost_bps=25.0)
    assert dear["gap_bps_per_year"] > cheap["gap_bps_per_year"]


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean_and_is_quiet_on_noise():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_t_finite(planted):
    frame, _ = planted
    r = st.race(frame["close"], frame["dividend"], frame["cash"])
    assert np.isfinite(st.one_sample_t(r["dlog"].to_numpy()))


def test_bootstrap_ci_brackets_the_point(planted):
    frame, _ = planted
    r = st.race(frame["close"], frame["dividend"], frame["cash"])
    ci = st.bootstrap_gap_ci(r["dlog"], n_boot=400, seed=939)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
    assert ci["block"] == 63


def test_bootstrap_is_reproducible(planted):
    frame, _ = planted
    r = st.race(frame["close"], frame["dividend"], frame["cash"])
    a = st.bootstrap_gap_ci(r["dlog"], n_boot=300, seed=7)
    b = st.bootstrap_gap_ci(r["dlog"], n_boot=300, seed=7)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]


# --------------------------------------------------------------------------- #
# The study's spine — the machinery is unbiased
# --------------------------------------------------------------------------- #
def test_planted_premium_makes_drip_win(planted):
    frame, _ = planted
    d = st.synthetic_detect(frame)
    assert d["gap_bps_per_year"] > 3.0
    assert d["terminal_ratio"] > 1.0


def test_null_gap_collapses(null):
    frame, _ = null
    d = st.synthetic_detect(frame)
    assert abs(d["gap_bps_per_year"]) < 4.0


def test_planted_beats_null_across_seeds():
    kw = dict(gen_kw=dict(n_years=10), sweep_freq="Q")
    pl = st.seed_sweep(data.synthetic_daily, 1.0, n_seeds=5, **kw)
    nl = st.seed_sweep(data.synthetic_daily, 0.0, n_seeds=5, **kw)
    assert pl["mean"] > nl["mean"] + 4.0 * max(pl["se"], nl["se"])
    assert abs(nl["mean"]) < 4.0 * max(nl["se"], 1e-9) + 2.0


def test_gap_is_monotone_in_distribution_yield():
    """More cash in transit = more to lose by parking it. The gap must scale."""
    frames, truth = data.synthetic_panel(n_years=12, signal_strength=1.0, seed=939)
    gaps = [st.synthetic_detect(frames[tk])["gap_bps_per_year"] for tk in truth["tickers"]]
    assert gaps[0] < gaps[1] < gaps[2]


def test_annual_sweep_costs_more_than_quarterly(planted):
    frame, _ = planted
    rows = {r["sweep_freq"]: r for r in
            st.frequency_sweep(frame["close"], frame["dividend"], frame["cash"])}
    assert rows["A"]["gap_bps_per_year"] > rows["Q"]["gap_bps_per_year"]
    assert rows["A"]["n_trades_sweep"] < rows["Q"]["n_trades_sweep"]


# --------------------------------------------------------------------------- #
# Sweeps & cuts return usable shapes
# --------------------------------------------------------------------------- #
def test_pay_lag_sweep_shape(planted):
    frame, _ = planted
    rows = st.pay_lag_sweep(frame["close"], frame["dividend"], frame["cash"],
                            lags=(0, 30))
    assert [r["pay_lag_days"] for r in rows] == [0, 30]
    assert all(np.isfinite(r["gap_bps_per_year"]) for r in rows)


def test_cost_sweep_shape(planted):
    frame, _ = planted
    rows = st.cost_sweep(frame["close"], frame["dividend"], frame["cash"],
                         grid=((0.0, 0.0), (0.0, 5.0)))
    assert len(rows) == 2
    assert all(np.isfinite(r["gap_bps_per_year"]) for r in rows)


def test_era_cut_returns_both_halves(planted):
    frame, _ = planted
    eras = st.era_cut(frame["close"], frame["dividend"], frame["cash"],
                      split="2010-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["gap_bps_per_year"]) and e["years"] > 1.0


def test_rate_regime_cut_runs(planted):
    """With a constant synthetic cash rate every year lands in one bucket."""
    frame, _ = planted
    out = st.rate_regime_cut(frame["close"], frame["dividend"], frame["cash"],
                             threshold=0.02)
    filled = [v for v in out.values() if v is not None]
    assert len(filled) >= 1
    assert all(np.isfinite(v["gap_bps_per_year"]) for v in filled)


def test_race_reports_both_arms_and_the_window(planted):
    frame, _ = planted
    r = st.race(frame["close"], frame["dividend"], frame["cash"])
    assert r["drip"]["n_days"] == r["sweep"]["n_days"]
    assert r["start"] < r["end"]
    assert r["terminal_ratio"] == pytest.approx(r["terminal_drip"] / r["terminal_sweep"])
    assert np.isfinite(r["sharpe_gap"])

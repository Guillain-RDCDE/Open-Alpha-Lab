"""Strategy tests — window geometry, no look-ahead, inference primitives, and the control.

All offline and synthetic. The two load-bearing behavioural tests are the last two: the
harness must **recover a planted announcement jump** and must **stay quiet on the null**.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dutch_auction import data, strategy as st  # noqa: E402


def _panel(fixture):
    prices, events, truth = fixture
    return st.EventPanel(prices, events, market="MKT", min_price=0.0), events, truth


# --------------------------------------------------------------------------- #
# Panel construction & screens
# --------------------------------------------------------------------------- #
def test_panel_screens_are_mechanical(planted):
    prices, events, _ = planted
    p = st.EventPanel(prices, events, market="MKT", min_price=0.0)
    assert len(p.events) == len(events)
    # A $1e9 price floor must drop every event; a screen must never keep more than it got.
    p2 = st.EventPanel(prices, events, market="MKT", min_price=1e9)
    assert p2.events == []


def test_panel_requires_the_market_leg(planted):
    prices, events, _ = planted
    with pytest.raises(KeyError):
        st.EventPanel({k: v for k, v in prices.items() if k != "MKT"}, events, market="MKT")


def test_abnormal_return_is_stock_minus_market(planted):
    prices, events, _ = planted
    p, _, _ = _panel(planted)
    tk = events[0][0]
    a, b = 300, 320
    px, mk = prices[tk], prices["MKT"]
    expected = (px.iloc[b] / px.iloc[a] - 1.0) - (mk.iloc[b] / mk.iloc[a] - 1.0)
    assert p.abn(tk, a, b) == pytest.approx(expected, abs=1e-12)


def test_abn_of_a_zero_length_window_is_zero(planted):
    p, events, _ = _panel(planted)
    assert p.abn(events[0][0], 400, 400) == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------- #
# Window geometry — exactly one execution lag, no look-ahead
# --------------------------------------------------------------------------- #
def test_exactly_one_execution_lag():
    """The tradable window must start at t+1 and never touch day t's return."""
    assert st.ENTRY_LAG == 1


def test_tradable_window_excludes_the_event_day(planted):
    """Blowing up the day-0 return must move ar0 and leave the tradable window untouched."""
    prices, events, _ = planted
    p, _, _ = _panel(planted)
    tbl = st.event_table(p)
    tk = tbl["ticker"].iloc[0]
    pos = [e for e in p.events if e["ticker"] == tk][0]["pos"]

    bumped = {k: v.copy() for k, v in prices.items()}
    s = bumped[tk].to_numpy(copy=True)
    s[pos:] *= 1.5                      # a +50% jump on the event day, carried forward
    bumped[tk] = pd.Series(s, index=prices[tk].index)

    p2 = st.EventPanel(bumped, events, market="MKT", min_price=0.0)
    t2 = st.event_table(p2)
    r1 = tbl[tbl.ticker == tk].iloc[0]
    r2 = t2[t2.ticker == tk].iloc[0]
    assert r2["ar0"] > r1["ar0"] + 0.4                      # the jump lands on day 0
    assert r2["window"] == pytest.approx(r1["window"], abs=1e-12)   # ... and nowhere else
    assert r2["m6"] == pytest.approx(r1["m6"], abs=1e-12)


def test_event_table_columns_and_dtypes(planted):
    p, _, _ = _panel(planted)
    tbl = st.event_table(p)
    for c in ("ticker", "t0", "ar0", "pre5", "window", "m1", "m3", "m6"):
        assert c in tbl.columns
    assert len(tbl) > 0
    assert np.isfinite(tbl[["ar0", "pre5", "window", "m1", "m3", "m6"]].to_numpy()).all()


def test_date_shift_moves_the_measured_day(planted):
    p, _, _ = _panel(planted)
    at0 = st.event_table(p, shift=0)["ar0"].mean()
    at3 = st.event_table(p, shift=3)["ar0"].mean()
    assert at0 > at3            # the planted jump is date-locked to shift 0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_matches_the_textbook_formula():
    x = np.array([0.01, -0.02, 0.03, 0.005, 0.012])
    expected = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert st.one_sample_t(x) == pytest.approx(expected)
    assert np.isnan(st.one_sample_t([0.01]))


def test_newey_west_equals_ols_t_at_zero_lags():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 500)
    assert st.newey_west_t(x, lags=0) == pytest.approx(
        x.mean() / (x.std(ddof=0) / np.sqrt(len(x))), rel=1e-9)


def test_newey_west_widens_the_se_on_a_positively_autocorrelated_series():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 4000)
    x = np.zeros(4000)
    for i in range(1, 4000):
        x[i] = 0.7 * x[i - 1] + e[i]
    x = x * 0.001 + 0.0004
    assert abs(st.newey_west_t(x, lags=20)) < abs(st.one_sample_t(x))


def test_jackknife_range_brackets_the_full_sample_t():
    rng = np.random.default_rng(2)
    x = rng.normal(0.02, 0.05, 80)
    j = st.jackknife_t(x)
    assert j["t_min"] <= j["t"] <= j["t_max"]
    assert j["n"] == 80


def test_jackknife_flags_a_single_driving_observation():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(0.0, 0.01, 40), [1.0]])
    j = st.jackknife_t(x)
    assert j["t"] > 0.5                       # the outlier alone carries the mean
    assert j["t_min"] < 0.1 * j["t"]          # drop it and the t collapses to nothing


def test_bootstrap_ci_brackets_the_mean_and_respects_blocks():
    rng = np.random.default_rng(3)
    x = rng.normal(0.03, 0.04, 200)
    for block in (1, 5):
        b = st.bootstrap_mean_ci(x, n_boot=800, seed=927, block=block)
        assert b["ci_low"] < b["mean"] < b["ci_high"]
        assert b["n"] == 200
    assert np.isnan(st.bootstrap_mean_ci([0.1, 0.2])["mean"])


def test_placebo_p_is_a_share_of_draws():
    draws = np.array([-0.02, 0.0, 0.01, 0.05])
    assert st.placebo_p(0.03, draws) == pytest.approx(0.25)
    assert st.placebo_p(0.0, draws) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Portfolios, races and sweeps
# --------------------------------------------------------------------------- #
def test_calendar_time_portfolio_costs_only_hurt(planted):
    p, _, _ = _panel(planted)
    tbl = st.event_table(p)
    free = st.calendar_time_portfolio(p, tbl, cost_bps=0.0, borrow_bps_yr=0.0)
    dear = st.calendar_time_portfolio(p, tbl, cost_bps=50.0, borrow_bps_yr=500.0)
    assert free["gross"]["mean_daily_bps"] == pytest.approx(dear["gross"]["mean_daily_bps"])
    assert dear["net"]["mean_daily_bps"] < free["net"]["mean_daily_bps"]
    assert free["net"]["mean_daily_bps"] == pytest.approx(free["gross"]["mean_daily_bps"])
    assert 0 < free["live_days"] <= free["net"]["n_days"]


def test_costs_are_charged_at_the_position_s_real_weight(planted):
    """A slot that is one of N live names is 1/N of NAV and can only cost 1/N of a trip.

    Charging a full-NAV round trip per event — the tempting shortcut — inflates the drag by
    the average number of live names and lets the cost model, not the tape, carry a
    negative verdict. The booked turnover must therefore be strictly below one round trip
    per event whenever the sleeve ever holds more than one name at a time.
    """
    p, _, _ = _panel(planted)
    tbl = st.event_table(p)
    cal = st.calendar_time_portfolio(p, tbl, cost_bps=10.0, borrow_bps_yr=0.0)
    assert cal["avg_names_live"] > 1.0
    assert 0 < cal["nav_round_trips"] < cal["n_events"]
    assert cal["nav_round_trips"] == pytest.approx(
        cal["n_events"] / cal["avg_names_live"], rel=0.35)
    # And the drag must be exactly what those round trips cost, not more.
    free = st.calendar_time_portfolio(p, tbl, cost_bps=0.0, borrow_bps_yr=0.0)
    booked = free["net"]["mean_daily_bps"] - cal["net"]["mean_daily_bps"]
    assert booked == pytest.approx(cal["cost_drag_bps_day"], rel=1e-9)


def test_cost_drag_scales_linearly_with_the_cost_assumption(planted):
    p, _, _ = _panel(planted)
    tbl = st.event_table(p)
    a = st.calendar_time_portfolio(p, tbl, cost_bps=10.0, borrow_bps_yr=0.0)
    b = st.calendar_time_portfolio(p, tbl, cost_bps=20.0, borrow_bps_yr=0.0)
    assert b["cost_drag_bps_day"] == pytest.approx(2.0 * a["cost_drag_bps_day"], rel=1e-9)
    assert a["nav_round_trips"] == pytest.approx(b["nav_round_trips"])


def test_long_only_race_charges_both_arms(planted):
    """The benchmark arm is not free either — it buys once and sells once."""
    prices, events, _ = planted
    p = st.EventPanel(prices, events, market="MKT", min_price=0.0)
    free = st.long_only_race(p, prices["MKT"], prices["CASH"], cost_bps=0.0)
    paid = st.long_only_race(p, prices["MKT"], prices["CASH"], cost_bps=25.0)
    assert paid["spy"]["mean_daily_bps"] < free["spy"]["mean_daily_bps"]
    assert paid["basket_net"]["mean_daily_bps"] < free["basket_net"]["mean_daily_bps"]


def test_long_only_race_is_excess_of_cash(planted):
    prices, events, _ = planted
    p = st.EventPanel(prices, events, market="MKT", min_price=0.0)
    tbl = st.event_table(p)
    race = st.long_only_race(p, prices["MKT"], prices["CASH"], cost_bps=10.0)
    for k in ("basket_gross", "basket_net", "spy"):
        assert np.isfinite(race[k]["sharpe_excess_cash"])
    # Costs can only reduce the net arm.
    assert race["basket_net"]["mean_daily_bps"] <= race["basket_gross"]["mean_daily_bps"]
    assert np.isfinite(race["t_diff_hac"])


def test_cost_and_borrow_sweeps_are_monotone(planted):
    p, _, _ = _panel(planted)
    tbl = st.event_table(p)
    cs = st.cost_sweep(p, tbl, cost_grid=(0.0, 10.0, 50.0))
    assert [r["cost_bps"] for r in cs] == [0.0, 10.0, 50.0]
    assert cs[0]["net_window_pct"] > cs[1]["net_window_pct"] > cs[2]["net_window_pct"]
    bs = st.borrow_sweep(p, tbl, borrow_grid=(0.0, 100.0, 400.0))
    assert bs[0]["mean_daily_bps"] > bs[1]["mean_daily_bps"] > bs[2]["mean_daily_bps"]


def test_expiry_and_era_sweeps_return_full_rows(planted):
    p, _, _ = _panel(planted)
    tbl = st.event_table(p)
    rows = st.expiry_sweep(p, grid=(15, 20))
    assert len(rows) == 2 and all(r["n"] > 0 for r in rows)
    eras = st.era_cut(tbl, split=str(tbl["t0"].median().date()))
    assert eras["early"]["n"] > 0 and eras["late"]["n"] > 0
    assert eras["early"]["n"] + eras["late"]["n"] == len(tbl)


def test_summarise_reports_every_window(planted):
    p, _, _ = _panel(planted)
    s = st.summarise(st.event_table(p))
    assert list(s.index) == ["pre5", "ar0", "window", "m1", "m3", "m6"]
    assert (s["n"] > 0).all() and (s["hit_rate"].between(0, 1)).all()


# --------------------------------------------------------------------------- #
# The synthetic control — recover the planted effect, stay quiet on the null
# --------------------------------------------------------------------------- #
def test_control_recovers_the_planted_announcement_jump(planted):
    prices, events, truth = planted
    d = st.synthetic_detect(prices, events, market="MKT")
    planted_bps = truth["planted_jump"] * 1e4
    assert d["ar0_bps"] == pytest.approx(planted_bps, rel=0.25)
    assert d["t_ar0"] > 5.0


def test_control_recovers_the_planted_post_expiry_drift(planted):
    prices, events, truth = planted
    d = st.synthetic_detect(prices, events, market="MKT")
    assert d["m6_bps"] > 0.4 * truth["planted_drift_6m"] * 1e4
    assert d["t_m6"] > 1.0


def test_control_is_quiet_on_the_null(null_panel):
    prices, events, truth = null_panel
    d = st.synthetic_detect(prices, events, market="MKT")
    assert truth["planted_jump"] == 0.0
    assert abs(d["ar0_bps"]) < 60.0          # planted world sits at ~470 bps
    assert abs(d["t_ar0"]) < 2.5


def test_control_is_quiet_across_seeds_on_the_null():
    """Seed-robust null: the day-0 t must behave like a standard normal, not fire."""
    ts = []
    for s in range(12):
        prices, events, _ = data.synthetic_panel(n_events=60, n_days=900,
                                                 signal_strength=0.0, seed=927 + s)
        ts.append(st.synthetic_detect(prices, events, market="MKT")["t_ar0"])
    ts = np.array(ts)
    assert abs(ts.mean()) < 1.0
    assert (np.abs(ts) >= 2.0).sum() <= 3     # ~5% nominal, small-sample slack

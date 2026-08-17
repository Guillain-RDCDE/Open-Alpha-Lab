"""Round-trip algebra, execution-lag hygiene and the study's spine — offline/synthetic.

The spine: on a panel with a *planted* announcement run-up and a *planted* post-expiry
give-back the odd-lot machinery recovers both; on the null it finds neither. Around that,
the algebra of the two round trips is pinned exactly (the identities in the module
docstring are tested, not asserted), costs only ever reduce returns, odd-lot priority is
worth exactly zero when the offer is under-subscribed, and the breakeven premium really is
the premium at which the hedged round trip is flat.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from odd_lot import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Event-window extraction: the single execution lag
# --------------------------------------------------------------------------- #
def test_event_row_marks_and_lag(planted):
    panel, events, truth = planted
    name, date, _, _ = events[0]
    row = st.event_row(panel[name], panel["MKT"], date, expiry_days=truth["expiry_days"])
    assert row is not None
    idx = panel[name].index
    i0 = int(idx.searchsorted(pd.Timestamp(date)))
    assert row["t0"] == idx[i0]
    assert row["entry_date"] == idx[i0 + 1]          # the ONE execution lag
    assert row["expiry_date"] == idx[i0 + truth["expiry_days"]]
    assert row["post_date"] == idx[i0 + truth["expiry_days"] + st.POST_DAYS_DEFAULT]


def test_event_row_returns_none_when_window_runs_off_the_tape(planted):
    panel, events, _ = planted
    name = events[0][0]
    s = panel[name]
    assert st.event_row(s, panel["MKT"], str(s.index[2].date())) is None      # no room left
    assert st.event_row(s, panel["MKT"], str(s.index[-3].date())) is None     # no room right


def test_runup_uses_only_prices_up_to_entry(planted):
    """Perturbing the tape strictly after the entry close cannot move the run-up."""
    panel, events, truth = planted
    name, date, _, _ = events[3]
    s = panel[name].copy()
    i0 = int(s.index.searchsorted(pd.Timestamp(date)))
    base = st.event_row(s, panel["MKT"], date, expiry_days=truth["expiry_days"])
    s2 = s.copy()
    s2.iloc[i0 + 2:] *= 2.5
    pert = st.event_row(s2, panel["MKT"], date, expiry_days=truth["expiry_days"])
    assert base["runup"] == pytest.approx(pert["runup"])
    assert base["p_entry"] == pytest.approx(pert["p_entry"])
    assert base["drift"] != pytest.approx(pert["drift"])   # the future did move


def test_build_event_table_skips_unknown_tickers_and_sorts(planted):
    panel, events, _ = planted
    events2 = list(events) + [("NOT_IN_PANEL", events[0][1], "0000000000-00-000000", "x")]
    tbl = st.build_event_table(panel, events2, market_key="MKT")
    assert len(tbl) == len(st.build_event_table(panel, events, market_key="MKT"))
    assert tbl["date"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Round-trip algebra
# --------------------------------------------------------------------------- #
def test_odd_lot_identity_premium_over_runup(planted):
    """1 + r_odd_gross must equal (1 + premium) / (1 + run-up), exactly."""
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    ev = st.odd_lot_returns(tbl, premium=0.13)
    lhs = 1.0 + ev["r_odd_gross"]
    rhs = 1.13 / (1.0 + ev["runup"])
    assert np.allclose(lhs.to_numpy(), rhs.to_numpy())


def test_priority_value_identity(planted):
    """r_odd − r_round must equal (1 − f) × [(p_clear − p_post)/p_entry + one-way cost]."""
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    f, c = 0.35, 15.0 * 1e-4
    ev = st.odd_lot_returns(tbl, premium=0.13, proration=f, cost_bps=15.0)
    expect = (1.0 - f) * ((ev["p_clear"] - ev["p_post"]) / ev["p_entry"] + c)
    assert np.allclose(ev["priority_value"].to_numpy(), expect.to_numpy())


def test_priority_is_worthless_when_offer_is_undersubscribed(planted):
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    ev = st.odd_lot_returns(tbl, proration=1.0)
    assert np.allclose(ev["priority_value"].to_numpy(), 0.0)


def test_breakeven_premium_flattens_the_hedged_round_trip(planted):
    """Re-pricing each event at its own breakeven premium must give exactly zero."""
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    ev = st.odd_lot_returns(tbl, premium=0.13, cost_bps=15.0, tender_fee_usd=30.0,
                            position_usd=5000.0, borrow_bps=40.0)
    p_clear_be = ev["p_pre"] * (1.0 + ev["breakeven_premium"])
    r = (p_clear_be / ev["p_entry"] - 1.0
         - 15.0e-4 - 30.0 / 5000.0 - ev["mkt_entry_exp"] - 2 * 15.0e-4
         - 40.0e-4 * ev["expiry_days"] / st.TRADING_DAYS)
    assert np.allclose(r.to_numpy(), 0.0, atol=1e-12)


def test_costs_and_fees_only_ever_reduce_the_round_trip(planted):
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    means = [st.odd_lot_returns(tbl, cost_bps=c, tender_fee_usd=f)["r_odd_hedged"].mean()
             for c, f in [(0.0, 0.0), (15.0, 30.0), (40.0, 50.0)]]
    assert means[0] > means[1] > means[2]


def test_higher_premium_raises_the_round_trip_one_for_one(planted):
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    a = st.odd_lot_returns(tbl, premium=0.10)["r_odd_gross"]
    b = st.odd_lot_returns(tbl, premium=0.20)["r_odd_gross"]
    ratio = ((1 + b) / (1 + a)).to_numpy()
    assert np.allclose(ratio, 1.20 / 1.10)


def test_smaller_position_means_a_bigger_flat_fee_drag(planted):
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    big = st.odd_lot_returns(tbl, position_usd=9900.0)["r_odd_net"].mean()
    small = st.odd_lot_returns(tbl, position_usd=1000.0)["r_odd_net"].mean()
    assert big > small


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_mean_and_stays_quiet_on_noise():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.01 + rng.normal(0, 0.02, 800)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.02, 800))) < 3


def test_wilson_interval_brackets_the_point():
    lo, hi = st.wilson_interval(70, 100)
    assert lo < 0.70 < hi


def test_block_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(2)
    x = 0.03 + rng.normal(0, 0.10, 200)
    ci = st.block_bootstrap_mean_ci(x, n_boot=800, seed=928)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_summarise_events_can_skip_the_bootstrap(planted):
    panel, events, _ = planted
    ev = st.odd_lot_returns(st.build_event_table(panel, events, market_key="MKT"))
    s = st.summarise_events(ev, "runup", boot_n=0)
    assert np.isnan(s["ci_low_pct"]) and np.isfinite(s["t_hac"])


# --------------------------------------------------------------------------- #
# The spine — planted effects recovered, null stays quiet
# --------------------------------------------------------------------------- #
def test_planted_runup_is_recovered(planted):
    panel, events, truth = planted
    d = st.synthetic_detect(panel, events)
    assert d["runup_t"] > 3.0
    assert d["runup_bps"] > 0.5 * truth["runup_planted_bps"]


def test_planted_giveback_is_recovered(planted):
    panel, events, truth = planted
    d = st.synthetic_detect(panel, events)
    assert d["giveback_t"] < -3.0
    assert d["giveback_bps"] < -0.5 * truth["giveback_planted_bps"]


def test_planted_breakeven_premium_is_lifted_by_the_runup(planted):
    panel, events, _ = planted
    assert st.synthetic_detect(panel, events)["breakeven_pct"] > 2.0


def test_null_shows_no_runup_and_no_giveback(null_panel):
    panel, events, _ = null_panel
    d = st.synthetic_detect(panel, events)
    assert abs(d["runup_t"]) < 3.0
    assert abs(d["giveback_t"]) < 3.0
    assert abs(d["breakeven_pct"]) < 2.5


def test_null_is_centred_across_seeds():
    runups, gives = [], []
    for s in range(6):
        panel, events, _ = data.synthetic_panel(n_events=60, signal_strength=0.0,
                                                seed=928 + s)
        d = st.synthetic_detect(panel, events)
        runups.append(d["runup_bps"]); gives.append(d["giveback_bps"])
    runups, gives = np.array(runups), np.array(gives)
    assert abs(runups.mean()) < 150.0     # bps — well inside the planted 500
    assert abs(gives.mean()) < 150.0


# --------------------------------------------------------------------------- #
# Aggregates: headline, sweeps, era cut, calendar-time sleeve
# --------------------------------------------------------------------------- #
def test_headline_reports_every_leg(planted):
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    h = st.headline(tbl, boot_n=400)
    for key in ("runup", "giveback", "r_odd_net", "r_odd_hedged", "r_round_net",
                "priority_value"):
        assert np.isfinite(h["legs"][key]["mean_pct"])
    assert h["n_events"] == len(tbl)
    assert 0.0 <= h["share_premium_below_breakeven"] <= 1.0


def test_premium_sweep_is_monotone(planted):
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    sw = st.premium_sweep(tbl, grid=(0.05, 0.13, 0.25))
    assert sw["mean_pct"].is_monotonic_increasing


def test_proration_sweep_ends_at_zero_priority(planted):
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    sw = st.proration_sweep(tbl, grid=(0.25, 0.50, 1.00))
    assert sw["priority_pct"].is_monotonic_decreasing
    assert sw["priority_pct"].iloc[-1] == pytest.approx(0.0)


def test_capacity_sweep_dollars_rise_with_the_lot(planted):
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    sw = st.capacity_sweep(tbl, grid=(1000.0, 5000.0, 9900.0))
    assert sw["dollar_per_event"].is_monotonic_increasing


def test_era_cut_returns_both_halves(planted):
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    split = str(tbl["date"].quantile(0.5).date())
    eras = st.era_cut(tbl, split=split, boot_n=0)
    assert eras["early"] is not None and eras["late"] is not None
    for tag in ("early", "late"):
        e = eras[tag]
        assert np.isfinite(e["hedged_pct"]) and np.isfinite(e["t_hac"])


def test_era_cut_tests_the_contrast_not_just_the_two_halves(planted):
    """A sub-period claim is a claim about a difference — the difference must be tested."""
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    split = str(tbl["date"].quantile(0.5).date())
    eras = st.era_cut(tbl, split=split, boot_n=0)
    d = eras["diff"]
    assert d is not None
    for col in ("runup", "giveback", "r_odd_hedged", "breakeven_premium"):
        assert np.isfinite(d[col]["welch_t"])
        assert d[col]["gap_pct"] == pytest.approx(d[col]["late_pct"] - d[col]["early_pct"])
    # On a homogeneous planted panel there is no true era effect to find.
    assert abs(d["runup"]["welch_t"]) < 3.0


def test_priority_value_is_not_a_tape_fact_it_moves_with_the_premium(planted):
    """The value of odd-lot priority inherits the assumed premium, one for one.

    ``priority_value`` is linear in the assumed premium with a strictly positive slope, so
    it crosses zero at a finite premium and its sign is an assumption, not a measurement.
    """
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    sw = st.premium_sweep(tbl, grid=(0.02, 0.13, 0.30))
    assert sw["priority_pct"].is_monotonic_increasing
    assert sw["priority_t"].is_monotonic_increasing
    lo, hi = sw.iloc[0], sw.iloc[-1]
    slope = (hi["priority_pct"] - lo["priority_pct"]) / (hi["premium_pct"] - lo["premium_pct"])
    assert slope > 0.3                      # ~ (1 - f) x p_pre/p_entry
    crossing = lo["premium_pct"] - lo["priority_pct"] / slope
    assert np.isfinite(crossing)
    flipped = st.premium_sweep(tbl, grid=(crossing / 100.0 - 0.05,
                                          crossing / 100.0 + 0.05))
    assert flipped["priority_pct"].iloc[0] < 0.0 < flipped["priority_pct"].iloc[-1]


def test_cost_sweep_covers_a_microcap_touch(planted):
    """The bps grid must reach a wide-spread case, and cost must bite three times over."""
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    assert max(data.COST_BPS_GRID) >= 100.0
    sw = st.cost_sweep(tbl, fee_grid=(30.0,), bps_grid=(15.0, 100.0))
    lo = float(sw.loc[sw["cost_bps"] == 15.0, "mean_pct"].iloc[0])
    hi = float(sw.loc[sw["cost_bps"] == 100.0, "mean_pct"].iloc[0])
    # 85 bps of extra touch costs 3 x 85 bps = 2.55 pp of round trip
    assert (lo - hi) == pytest.approx(3 * 85e-4 * 100, abs=1e-6)


def test_calendar_time_portfolio_is_inside_the_event_span(planted):
    panel, events, _ = planted
    tbl = st.build_event_table(panel, events, market_key="MKT")
    ev = st.odd_lot_returns(tbl)
    cal = st.calendar_time_portfolio(panel, ev, market_key="MKT", cash_key="CASH")
    assert cal.index.min() >= ev["entry_date"].min()
    assert cal.index.max() <= ev["expiry_date"].max()
    assert (cal["n_live"] >= 0).all()
    idle = cal[cal["n_live"] == 0]
    if len(idle) > 5:
        assert np.allclose(idle["r_port"].to_numpy(), idle["r_cash"].to_numpy())


def test_portfolio_race_is_finite_and_excess_of_cash(planted):
    panel, events, _ = planted
    ev = st.odd_lot_returns(st.build_event_table(panel, events, market_key="MKT"))
    cal = st.calendar_time_portfolio(panel, ev, market_key="MKT", cash_key="CASH")
    race = st.portfolio_race(cal)
    for k in ("sharpe_port", "sharpe_mkt", "sharpe_adv", "t_diff", "invested_frac"):
        assert np.isfinite(race[k])
    assert 0.0 < race["invested_frac"] <= 1.0

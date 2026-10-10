"""Strategy tests for Study 1013 — the lead, the announcement test, the power, the rules.

Synthetic-first: every detector is shown to fire on the planted world AND to stay quiet on the
matched null. Real-tape checks are gated on the bundled tapes being present.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from nberclock import data, strategy as st  # noqa: E402

needs_real = pytest.mark.skipif(not data.have_real(),
                                reason="bundled tapes not available — synthetic tests cover "
                                       "the logic")


def _xs(world):
    return st.excess_index(world["returns"])


def _tr(world):
    return (1.0 + world["returns"]["mkt"]).cumprod()


# --------------------------------------------------------------------------- #
# Utilities: dates, bars, the one lag
# --------------------------------------------------------------------------- #
def test_months_between():
    assert st.months_between(pd.Timestamp("2007-12-31"), pd.Timestamp("2008-12-01")) == 12
    assert st.months_between(pd.Timestamp("2009-06-30"), pd.Timestamp("2009-02-28")) == -4


def test_bar_position_monthly_is_the_month_containing_the_date():
    idx = pd.date_range("2008-01-31", periods=24, freq="ME")
    pos = st.bar_position(idx, "2008-12-01")
    assert idx[pos] == pd.Timestamp("2008-12-31")


def test_bar_position_daily_is_the_day_itself_or_next_session():
    idx = pd.bdate_range("2008-11-24", periods=10)
    assert idx[st.bar_position(idx, "2008-12-01")] == pd.Timestamp("2008-12-01")
    assert idx[st.bar_position(idx, "2008-11-29")] == pd.Timestamp("2008-12-01")  # Saturday


def test_bar_position_outside_the_tape_is_none():
    idx = pd.bdate_range("1990-01-02", periods=300)
    assert st.bar_position(idx, "1985-06-03") is None
    assert st.bar_position(idx, "2030-01-01") is None


def test_sign_test_drops_ties_and_is_exact():
    r = st.sign_test([1, 1, 1, 1, 1, 0, 0])
    assert r["n_pos"] == 5 and r["n_zero"] == 2
    assert r["p"] == pytest.approx(2 * 0.5 ** 5)


def test_excess_index_is_ratio_of_wealths():
    m = pd.DataFrame({"mkt": [0.10, 0.0], "rf": [0.0, 0.05]},
                     index=pd.date_range("2000-01-31", periods=2, freq="ME"))
    x = st.excess_index(m)
    assert x.iloc[-1] == pytest.approx(1.10 / 1.05)


# --------------------------------------------------------------------------- #
# The lead
# --------------------------------------------------------------------------- #
def test_lead_lag_recovers_the_planted_lead(planted):
    ll = st.lead_lag(_tr(planted), planted["cycles"])
    s = st.lead_summary(ll)
    lead = planted["truth"]["lead_months"]
    assert abs(s["trough_median"] - lead) <= 1
    assert s["trough_pos"] >= 0.9 * s["n"]


def test_lead_rotation_fires_on_planted(planted):
    r = st.lead_rotation_test(_tr(planted), planted["cycles"])
    assert r["observed_share"] > r["null_share_mean"]
    assert r["p_share"] < 0.05


def test_lead_rotation_quiet_on_null(null):
    r = st.lead_rotation_test(_tr(null), null["cycles"])
    assert r["p_share"] > 0.10


def test_the_estimator_leans_towards_market_first_even_on_the_null(null):
    """Why the calendar-rotation null exists: a 50% sign test would be too generous."""
    r = st.lead_rotation_test(_tr(null), null["cycles"])
    assert r["null_share_mean"] > 0.55


def test_lead_lag_flags_censored_cycles():
    lvl = pd.Series(np.linspace(1, 2, 60), index=pd.date_range("2000-01-31", periods=60,
                                                                  freq="ME"))
    cyc = pd.DataFrame({"peak": [pd.Timestamp("2001-03-31")],
                        "trough": [pd.Timestamp("2001-11-30")]})
    ll = st.lead_lag(lvl, cyc)
    assert bool(ll["censored"].iloc[0]) is True


# --------------------------------------------------------------------------- #
# The announcement table
# --------------------------------------------------------------------------- #
def test_announcement_table_measures_the_fall_already_done():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    x = np.r_[np.full(20, 100.0), np.linspace(100, 60, 10), np.linspace(60, 120, 30)]
    lvl = pd.Series(x, index=idx)
    anns = pd.DataFrame({"kind": ["peak"], "reference": [idx[24]], "date": [idx[25]]})
    t = st.announcement_table(lvl, anns, horizons=(12,))
    r = t.iloc[0]
    assert r["already"] == pytest.approx(x[25] / 100 - 1)
    assert r["total_fall"] == pytest.approx(-0.40)
    assert 0 < r["share_done"] < 1
    assert r["fwd_12m"] == pytest.approx(x[37] / x[25] - 1)


def test_announcement_table_measures_the_rally_already_missed():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    x = np.r_[np.linspace(100, 50, 20), np.linspace(50, 100, 40)]
    lvl = pd.Series(x, index=idx)
    anns = pd.DataFrame({"kind": ["trough"], "reference": [idx[24]], "date": [idx[35]]})
    r = st.announcement_table(lvl, anns).iloc[0]
    assert r["already"] == pytest.approx(x[35] / 50 - 1)
    assert r["months_since_extreme"] == 16


# --------------------------------------------------------------------------- #
# The rotation test — fires on planted, quiet on null
# --------------------------------------------------------------------------- #
def test_rotation_test_fires_on_planted(planted):
    lvl = _xs(planted)
    r = st.rotation_test(lvl, st.event_positions(lvl.index, planted["announcements"]), 12)
    assert r["n_events"] == planted["truth"]["n_announcements"]
    assert r["excess_vs_random"] > 0.05
    assert r["p_one_sided"] < 0.05


def test_rotation_test_quiet_on_null(null):
    lvl = _xs(null)
    r = st.rotation_test(lvl, st.event_positions(lvl.index, null["announcements"]), 12)
    assert r["p_one_sided"] > 0.05
    assert abs(r["excess_vs_random"]) < 0.06


def test_rotation_test_is_exact_and_seed_free(planted):
    lvl = _xs(planted)
    pos = st.event_positions(lvl.index, planted["announcements"])
    a = st.rotation_test(lvl, pos, 12)
    b = st.rotation_test(lvl, pos, 12)
    assert a["p_one_sided"] == b["p_one_sided"]
    assert a["n_shifts"] == len(lvl) - 12
    assert 1.0 / a["n_shifts"] <= a["p_one_sided"] <= 1.0
    assert a["observed"] == pytest.approx(np.mean(a["event_returns"]))


def test_rotation_null_mean_is_the_unconditional_mean(planted):
    lvl = _xs(planted)
    pos = st.event_positions(lvl.index, planted["announcements"])
    r = st.rotation_test(lvl, pos, 12)
    assert r["null_mean"] == pytest.approx(st.forward_returns(lvl, 12).mean())


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #
def test_power_curve_has_size_near_nominal_and_rises():
    pc = st.power_curve(effects=(0.0, 0.15, 0.40), n_sims=40, n_cycles=5, n_years=60,
                        vol=0.16)
    assert pc.loc[0.0, "reject_rate"] <= 0.15
    assert pc.loc[0.40, "reject_rate"] > pc.loc[0.15, "reject_rate"] >= pc.loc[0.0,
                                                                               "reject_rate"]
    assert pc.loc[0.40, "reject_rate"] >= 0.9


def test_minimum_detectable_effect_interpolates():
    c = pd.DataFrame({"reject_rate": [0.05, 0.5, 0.9]}, index=[0.0, 0.1, 0.2])
    assert st.minimum_detectable_effect(c, 0.8) == pytest.approx(0.175)
    assert np.isnan(st.minimum_detectable_effect(c, 0.95))


# --------------------------------------------------------------------------- #
# Rules: one lag, costs, excess-vs-excess
# --------------------------------------------------------------------------- #
def _toy():
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    ret = pd.Series([0.10, -0.20, 0.30, 0.05, -0.05, 0.02], index=idx)
    rf = pd.Series(0.001, index=idx)
    return idx, ret, rf


def test_timing_backtest_applies_exactly_one_lag():
    idx, ret, rf = _toy()
    sig = pd.Series([1, 0, 0, 1, 1, 1], index=idx, dtype=float)
    b = st.timing_backtest(ret, rf, sig, cost_bps=0.0)
    g = b["gross_series"]
    # decided out at the close of bar 1 -> bar 1 still earns equity, bar 2 earns bills
    assert g.iloc[1] == pytest.approx(-0.20)
    assert g.iloc[2] == pytest.approx(0.001)
    assert g.iloc[3] == pytest.approx(0.001)
    assert g.iloc[4] == pytest.approx(-0.05)


def test_costs_charged_one_way_per_switch():
    idx, ret, rf = _toy()
    sig = pd.Series([1, 0, 0, 1, 1, 1], index=idx, dtype=float)
    b = st.timing_backtest(ret, rf, sig, cost_bps=25.0)
    diff = b["gross_series"] - b["net_series"]
    assert diff.sum() == pytest.approx(2 * 25e-4)
    assert b["switches"] == 2


def test_always_invested_signal_is_buy_and_hold():
    idx, ret, rf = _toy()
    b = st.timing_backtest(ret, rf, pd.Series(1.0, index=idx), cost_bps=10.0)
    assert np.allclose(b["net_series"], ret)
    assert b["exposure"] == 1.0


def test_signal_avoid_sits_in_bills_between_announcements():
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    anns = pd.DataFrame({"kind": ["peak", "trough"],
                         "reference": [idx[0], idx[5]],
                         "date": [idx[4] - pd.Timedelta(days=10), idx[12] - pd.Timedelta(days=10)]})
    s = st.signal_avoid(idx, anns)
    assert (s.iloc[:4] == 1).all() and (s.iloc[4:12] == 0).all() and (s.iloc[12:] == 1).all()


def test_signal_buy_after_holds_for_the_window():
    idx = pd.date_range("2000-01-31", periods=30, freq="ME")
    anns = pd.DataFrame({"kind": ["peak"], "reference": [idx[0]], "date": [idx[5]]})
    s = st.signal_buy_after(idx, anns, 12)
    assert s.sum() == 12 and s.iloc[5] == 1 and s.iloc[17] == 0


def test_buy_after_rule_beats_shifted_dates_on_planted_not_on_null(planted, null):
    ps = []
    for w in (planted, null):
        r = w["returns"]
        sig = st.signal_buy_after(r.index, w["announcements"], 12)
        ps.append(st.rule_rotation_test(r["mkt"], r["rf"], sig)["p_one_sided"])
    assert ps[0] < 0.05
    assert ps[1] > 0.05


def test_compare_rules_reports_excess_sharpe_and_hac(planted):
    r = planted["returns"]
    sig = {"buy": st.signal_buy_after(r.index, planted["announcements"], 12)}
    c = st.compare_rules(r["mkt"], r["rf"], sig)
    assert list(c.index) == ["buy-and-hold", "buy"]
    assert np.isfinite(c.loc["buy", "active_hac_t"])
    assert c.loc["buy", "sharpe_gross"] >= c.loc["buy", "sharpe_net"]


def test_cost_sweep_is_monotone(planted):
    r = planted["returns"]
    sig = st.signal_avoid(r.index, planted["announcements"])
    sw = st.cost_sweep(r["mkt"], r["rf"], sig, costs=(0, 10, 50))
    assert sw["sharpe_net"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# The verdict — both directions
# --------------------------------------------------------------------------- #
def _h(**kw):
    h = {"lead_trough_sign_p": 0.001, "lead_trough_median": 4.0, "lead_rot_p_share": 0.03,
         "lead_rot_null_share": 0.7, "lead_rot_p_mean": 0.8, "lead_rot_obs_mean": 4.4,
         "n_cycles_lead": 15, "lead_trough_pos": 14, "lead_peak_pos": 12,
         "lead_peak_median": 3.0, "dd_already_median": -0.07, "missed_median": 0.63,
         "n_events_monthly": 10, "n_events_daily": 8, "buy_obs_pooled": 0.03,
         "buy_null_pooled": 0.085, "buy_p_pooled": 0.82, "buy_p_peak": 0.43,
         "buy_p_trough": 0.93, "buy_p_pooled_daily": 0.54, "mde80": 0.17, "cost_bps": 10.0,
         "rule_gain_monthly": -0.38, "rule_gain_daily": -0.27, "rule_hac_t_monthly": -3.2,
         "rule_perm_p_monthly": 0.76, "rule_sharpe_monthly": 0.14, "bh_sharpe_monthly": 0.52,
         "rule_exposure": 0.24, "avoid_gain_monthly": -0.09, "avoid_gain_daily": -0.07}
    h.update(kw)
    return h


def test_verdict_mixed_when_only_the_lead_is_real():
    v = st.verdict(_h())
    assert v["signal"] == "Mixed" and v["trad"] == "Mirage"
    assert v["lead_real"] and not v["buy_real"]


def test_verdict_real_and_investable_when_everything_clears():
    v = st.verdict(_h(buy_p_pooled=0.01, rule_gain_monthly=0.2, rule_gain_daily=0.1,
                      rule_hac_t_monthly=2.5, rule_perm_p_monthly=0.01))
    assert v["signal"] == "Real" and v["trad"] == "Investable"


def test_verdict_none_when_neither_leg_holds():
    v = st.verdict(_h(lead_trough_sign_p=0.3, lead_rot_p_share=0.4))
    assert v["signal"] == "None"


def test_verdict_weak_when_buy_leg_is_only_suggestive():
    v = st.verdict(_h(lead_trough_sign_p=0.3, buy_p_pooled=0.12))
    assert v["signal"] == "Weak"


def test_verdict_lead_needs_the_calendar_rotation_too():
    """A sign test against 50% alone does not make the lead real (the estimator leans)."""
    v = st.verdict(_h(lead_rot_p_share=0.2))
    assert not v["lead_real"]


def test_verdict_fragile_needs_both_tapes():
    assert st.verdict(_h(rule_gain_monthly=0.05, rule_gain_daily=0.02,
                         rule_hac_t_monthly=1.2))["trad"] == "Fragile"
    assert st.verdict(_h(rule_gain_monthly=0.05, rule_gain_daily=-0.02,
                         rule_hac_t_monthly=1.2))["trad"] == "Mirage"


def test_verdict_investable_is_refused_without_a_real_buy_signal():
    v = st.verdict(_h(buy_p_pooled=0.3, rule_gain_monthly=0.2, rule_gain_daily=0.1,
                      rule_hac_t_monthly=2.5, rule_perm_p_monthly=0.01))
    assert v["trad"] == "Fragile"


# --------------------------------------------------------------------------- #
# Real tapes (gated)
# --------------------------------------------------------------------------- #
@needs_real
def test_real_lead_lag_covers_all_sixteen_cycles():
    ll = st.lead_lag_all(data.total_return_index(data.load_monthly()), data.load_daily(),
                         data.cycles())
    assert len(ll) == 16
    assert ll["censored"].sum() == 1
    assert (ll["tape"] == "S&P 500 price").sum() == 1


@needs_real
def test_real_market_trough_usually_leads():
    ll = st.lead_lag_all(data.total_return_index(data.load_monthly()), data.load_daily(),
                         data.cycles())
    s = st.lead_summary(ll)
    assert s["trough_pos"] >= 12 and s["trough_median"] >= 1


@needs_real
def test_real_announcement_counts_by_tape():
    anns = data.announcements()
    m = data.load_monthly()
    d = data.load_daily()
    xs = st.excess_index(m)
    assert len(st.event_positions(xs.index, anns)) == 10
    assert len(st.event_positions(d.index, anns)) == 8


@needs_real
def test_real_trough_announcements_come_after_big_rebounds():
    anns = data.announcements()
    at_m = st.announcement_table(data.total_return_index(data.load_monthly()), anns)
    at_d = st.announcement_table(data.load_daily(), anns, bars_per_month=21)
    best = st.best_available(at_m, at_d)
    assert len(best) == 12
    assert (best.loc[best.kind == "trough", "already"] > 0.2).all()

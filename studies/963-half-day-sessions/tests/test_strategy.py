"""Strategy tests for Study 963 — all offline, all deterministic.

The calendar rule is pinned against dates that can be checked by hand; the inference is
pinned against a planted world (a known bump must be recovered) and against its null (an
apparatus that finds an effect in the null is worthless).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from half_day import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The calendar rule
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("year,day", [(2020, 26), (2021, 25), (2022, 24), (2023, 23), (2024, 28)])
def test_thanksgiving_is_the_fourth_thursday(year, day):
    tg = st.thanksgiving(year)
    assert (tg.month, tg.day) == (11, day)
    assert tg.dayofweek == 3  # Thursday


def test_rule_candidates_finds_the_three_families():
    idx = pd.bdate_range("2015-01-01", "2019-12-31")
    cands = st.rule_candidates(idx)
    fams = set(cands.unique())
    assert fams == set(st.FAMILIES)
    # Black Friday 2018 = 2018-11-23; Christmas Eve 2018 was a Monday.
    assert cands.get(pd.Timestamp("2018-11-23")) == "black_friday"
    assert cands.get(pd.Timestamp("2018-12-24")) == "dec24"


def test_rule_never_proposes_a_non_trading_day():
    """2016-12-24 was a Saturday and 2017-07-03 a Monday: only the Monday may appear."""
    idx = pd.bdate_range("2016-01-01", "2017-12-31")
    cands = st.rule_candidates(idx)
    assert pd.Timestamp("2016-12-24") not in cands.index
    assert cands.get(pd.Timestamp("2017-07-03")) == "jul3"


def test_candidates_are_subset_of_the_index():
    idx = pd.bdate_range("2005-01-01", "2012-12-31")
    cands = st.rule_candidates(idx)
    assert cands.index.isin(idx).all()
    assert cands.index.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Session arithmetic
# --------------------------------------------------------------------------- #
def test_session_legs_compose_exactly(planted):
    bars, _ = planted
    s = st.session_frame(bars)
    lhs = (1 + s["r_on"]) * (1 + s["r_oc"])
    assert np.allclose(lhs.dropna(), (1 + s["r_cc"]).reindex(lhs.dropna().index), atol=1e-12)


def test_volume_ratio_is_strictly_trailing(planted):
    """The benchmark median must not see the day it is judging."""
    bars, _ = planted
    vr = st.volume_ratio(bars, window=20)
    med = bars["volume"].rolling(20, min_periods=10).median().shift(1)
    assert np.allclose(vr.dropna(), (bars["volume"] / med).dropna())
    assert np.isnan(vr.iloc[0])


def test_volume_ratio_centres_near_one_on_ordinary_days(planted):
    bars, _ = planted
    vr = st.volume_ratio(bars).dropna()
    assert 0.9 < vr.median() < 1.1


# --------------------------------------------------------------------------- #
# Confirmation and recall
# --------------------------------------------------------------------------- #
def test_confirmation_keeps_thin_days_and_rejects_normal_ones(planted):
    bars, _ = planted
    thin_bars, dates = st.plant_half_days(bars, every=84)
    sessions = st.session_frame(thin_bars)
    cands = pd.Series("dec24", index=dates, name="family")
    conf = st.confirm_candidates(sessions, cands)
    # Not 100%: synthetic volume is noisy, so a planted "half day" occasionally prints
    # above the threshold — exactly the failure mode the confirmation step exists to show.
    assert conf["confirmed"].mean() > 0.85
    # The same rule pointed at ordinary days confirms only the occasional quiet one.
    ordinary = sessions.index[10:400:37]
    conf2 = st.confirm_candidates(sessions, pd.Series("dec24", index=ordinary))
    assert conf2["confirmed"].mean() < 0.20
    assert conf2["confirmed"].mean() < conf["confirmed"].mean() / 3


def test_unclaimed_thin_days_are_reported(planted):
    bars, _ = planted
    thin_bars, dates = st.plant_half_days(bars, every=84)
    sessions = st.session_frame(thin_bars)
    # Hide a third of the planted dates from the rule: they must resurface as unclaimed.
    partial = pd.Series("dec24", index=dates[::3])
    unclaimed = st.unclaimed_thin_days(sessions, partial)
    assert len(unclaimed) >= len(dates) - len(partial) - 5


# --------------------------------------------------------------------------- #
# Inference: the planted world and its null
# --------------------------------------------------------------------------- #
def test_planted_bump_is_recovered(planted):
    bars, _ = planted
    bumped, dates = st.plant_half_days(bars, every=84, bump_bps=60.0)
    det = st.synthetic_detect(bumped, dates)
    assert det["diff_bps"] > 30.0
    assert det["t_diff"] > 2.0
    assert det["ci_low"] > 0.0


def test_null_world_is_not_systematically_detected():
    """Thin volume with no return bump must behave like a fair test, not a discovery.

    One seed proves nothing about a 5%-size test, so this runs twenty worlds and looks at
    the *distribution* of the statistic: the mean |t| of a well-behaved two-sided test is
    about 0.8, and clearing |t| = 2 should be rare. (An earlier revision that applied a
    Newey-West kernel across the scattered event days failed this badly — mean |t| well
    above 1 and hits in a third of the worlds. The test is here because it caught that.)
    """
    ts = []
    for s in range(20):
        bars, _ = data.synthetic_ohlc(n_years=10, signal_strength=1.0, seed=963 + s)
        flat, dates = st.plant_half_days(bars, every=84, bump_bps=0.0)
        sessions = st.session_frame(flat)
        ts.append(st.group_stats(sessions["r_cc"], st.event_mask(sessions, dates, 0))["t_diff"])
    ts = np.abs(np.array(ts))
    assert ts.mean() < 1.3
    assert (ts >= 2.0).mean() <= 0.20


def test_group_stats_handles_a_tiny_event_set(planted):
    bars, _ = planted
    sessions = st.session_frame(bars)
    stt = st.group_stats(sessions["r_cc"], st.event_mask(sessions, sessions.index[:2], 0))
    assert np.isnan(stt["t_diff"]) and stt["n_event"] == 2


def test_bootstrap_ci_brackets_the_point_estimate(planted):
    bars, _ = planted
    bumped, dates = st.plant_half_days(bars, every=84, bump_bps=60.0)
    sessions = st.session_frame(bumped)
    mask = st.event_mask(sessions, dates, 0)
    point = st.group_stats(sessions["r_cc"], mask)["diff_bps"]
    ci = st.bootstrap_diff_ci(sessions["r_cc"], mask, n_boot=1500)
    assert ci["ci_low"] < point < ci["ci_high"]


def test_event_mask_offsets_shift_by_one_session(planted):
    bars, _ = planted
    sessions = st.session_frame(bars)
    dates = sessions.index[100:400:50]
    m0 = st.event_mask(sessions, dates, 0)
    m1 = st.event_mask(sessions, dates, 1)
    assert m0.sum() == m1.sum()
    assert (m1.to_numpy()[1:] == m0.to_numpy()[:-1]).sum() >= m0.sum() - 1


def test_window_and_family_tables_have_the_expected_shape(planted):
    bars, _ = planted
    bumped, dates = st.plant_half_days(bars, every=84, bump_bps=40.0)
    sessions = st.session_frame(bumped)
    wt = st.window_table(sessions, dates)
    assert list(wt.index) == [-1, 0, 1]
    assert wt.loc[0, "diff_bps"] > wt.loc[-1, "diff_bps"]
    conf = st.confirm_candidates(sessions, pd.Series("dec24", index=dates))
    ft = st.family_table(sessions, conf)
    assert list(ft.index) == list(st.FAMILIES)
    # family_table reads only the CONFIRMED dates — that is the point of the step.
    assert ft.loc["dec24", "n_event"] == int(conf["confirmed"].sum())
    assert np.isnan(ft.loc["jul3", "t_diff"])  # no dates in that family here


def test_era_cut_splits_the_sample(planted):
    bars, _ = planted
    bumped, dates = st.plant_half_days(bars, every=84, bump_bps=40.0)
    sessions = st.session_frame(bumped)
    ec = st.era_cut(sessions, dates, split=str(sessions.index[len(sessions) // 2].date()))
    assert list(ec.index) == ["early", "late"]
    assert ec["n_event"].sum() >= len(dates) - 2


# --------------------------------------------------------------------------- #
# The cost arithmetic
# --------------------------------------------------------------------------- #
def test_cost_arithmetic_is_monotone_and_breaks_even_where_it_should():
    tbl = st.cost_arithmetic(edge_bps=10.0, sessions_per_year=3.0)
    assert tbl["net_bps_per_year"].is_monotonic_decreasing
    assert tbl.loc[0.0, "net_bps_per_year"] == pytest.approx(30.0)
    assert tbl.loc[5.0, "net_bps_per_year"] == pytest.approx(0.0)
    assert st.breakeven_cost_bps(10.0) == pytest.approx(5.0)


def test_expected_false_positives_is_plain_arithmetic():
    assert st.expected_false_positives(45) == pytest.approx(2.25)


# --------------------------------------------------------------------------- #
# The verdict rule itself — a decision rule that is not tested is not a rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_hits": 4, "expected_hits": 2.25, "n_cells": 45, "n_cells0": 15,
         "n_hits0": 0, "expected_hits0": 0.75, "max_abs_t0": 1.2,
         "median_volume_ratio": 0.42, "spy_edge_bps": 1.0, "spy_t": 0.2,
         "tickers": list("ABCDE"), "sessions_per_year": 2.0, "best_ticker": "GLD",
         "best_edge_bps": 8.0, "best_ci_low": -20.0, "best_ci_high": 40.0,
         "net_at_1bp": 18.0, "breakeven_bps": 4.0}
    h.update(over)
    return h


def test_verdict_says_none_when_the_half_day_battery_is_at_chance():
    v = st.verdict(_headline())
    assert v["signal"] == "None" and v["trad"] == "Mirage"


def test_verdict_says_weak_when_hits_beat_luck_but_no_cell_is_strong():
    assert st.verdict(_headline(n_hits0=3, max_abs_t0=2.2))["signal"] == "Weak"


def test_verdict_says_real_only_with_both_conditions():
    assert st.verdict(_headline(n_hits0=3, max_abs_t0=3.0))["signal"] == "Real"
    assert st.verdict(_headline(n_hits0=3, max_abs_t0=2.2))["signal"] == "Weak"
    assert st.verdict(_headline(n_hits0=0, max_abs_t0=4.0))["signal"] == "None"


def test_verdict_ignores_the_pre_holiday_cells():
    """A battery lit up entirely at offset -1 must not earn this study's Signal stamp."""
    loud = _headline(n_hits=12, expected_hits=2.25, n_hits0=0, max_abs_t0=1.1)
    assert st.verdict(loud)["signal"] == "None"


def test_verdict_tradability_ladder():
    rich = dict(net_at_1bp=200.0, best_ci_low=5.0)
    assert st.verdict(_headline(**rich))["trad"] == "Investable"
    # Same money, but an interval that contains zero is never Investable.
    assert st.verdict(_headline(net_at_1bp=200.0, best_ci_low=-5.0))["trad"] == "Fragile"
    assert st.verdict(_headline(net_at_1bp=40.0))["trad"] == "Fragile"
    assert st.verdict(_headline(net_at_1bp=-5.0))["trad"] == "Mirage"


def test_verdict_prose_quotes_the_numbers_it_was_given():
    v = st.verdict(_headline(spy_edge_bps=12.3))
    assert "+12.3" in v["signal_why"] and "%" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

"""Strategy tests for Study 985 — hindsight against real time."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lasthike import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The hard-coded policy record
# --------------------------------------------------------------------------- #
def test_the_policy_record_is_internally_consistent():
    """Every recorded move must actually move the target in the stated direction."""
    p = st.policy_path()
    d = p["target"].diff().dropna()
    dirs = p["direction"].iloc[1:]
    assert (np.sign(d.to_numpy()) == dirs.to_numpy()).all()


def test_the_policy_record_is_sorted_and_unique():
    p = st.policy_path()
    assert p.index.is_monotonic_increasing
    assert not p.index.has_duplicates
    assert p.index[0] >= pd.Timestamp("1994-01-01")


def test_the_target_never_goes_negative():
    assert (st.policy_path()["target"] >= 0).all()


def test_the_known_cycles_are_found():
    """The five tightening cycles every macro desk would name."""
    cycles = st.tightening_cycles(st.policy_path())
    lasts = {str(d.date()) for d in cycles["last_hike"]}
    for known in ("1995-02-01", "2000-05-16", "2006-06-29", "2018-12-19", "2023-07-26"):
        assert known in lasts, f"missing the cycle ending {known}"


def test_a_single_isolated_hike_is_not_a_cycle():
    """1997's lone hike must not become a sixth 'cycle'."""
    cycles = st.tightening_cycles(st.policy_path(), min_hikes=3)
    assert not any(str(d.date()).startswith("1997") for d in cycles["last_hike"])


def test_the_cycle_definition_is_a_choice_that_matters():
    loose = st.tightening_cycles(st.policy_path(), min_hikes=1, gap_months=24)
    tight = st.tightening_cycles(st.policy_path(), min_hikes=5, gap_months=6)
    assert len(loose) > len(tight)


def test_cycles_report_their_size():
    c = st.tightening_cycles(st.policy_path())
    assert (c["n_hikes"] >= 3).all()
    assert (c["total_tightening"] > 0).all()
    assert (c["last_hike"] > c["first_hike"]).all()


# --------------------------------------------------------------------------- #
# Recognition and false alarms
# --------------------------------------------------------------------------- #
def test_the_live_rule_fires_after_the_quiet_period():
    p = st.policy_path()
    c = st.tightening_cycles(p)
    rd = st.recognition_delay(p, c, quiet_months=6)
    assert len(rd) == len(c)
    assert ((rd["declared_over"] - rd["last_hike"]).dt.days > 150).all()


def test_a_longer_quiet_period_means_a_later_call():
    p = st.policy_path()
    c = st.tightening_cycles(p)
    short = st.recognition_delay(p, c, 3)["declared_over"]
    long = st.recognition_delay(p, c, 12)["declared_over"]
    assert (long.to_numpy() > short.to_numpy()).all()


def test_the_live_rule_produces_false_alarms():
    """The thing a hindsight event study can never see."""
    fa = st.false_alarms(st.policy_path(), quiet_months=6)
    assert len(fa) > len(st.tightening_cycles(st.policy_path()))
    assert not fa["was_the_end"].all()
    assert fa["was_the_end"].any()


def test_a_longer_quiet_period_buys_fewer_false_alarms():
    p = st.policy_path()
    short = st.false_alarms(p, 3)
    long = st.false_alarms(p, 12)
    assert (1 - short["was_the_end"].mean()) >= (1 - long["was_the_end"].mean())


def test_false_alarms_never_fire_before_a_hike_that_already_happened():
    fa = st.false_alarms(st.policy_path(), 6)
    assert (fa["signal_date"] > fa["after_hike"]).all()


# --------------------------------------------------------------------------- #
# The event study
# --------------------------------------------------------------------------- #
def _series(n=8000, drift=0.08, seed=985):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1994-01-03", periods=n)
    r = drift / 252 + rng.normal(0, 0.16 / np.sqrt(252), n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


def test_forward_returns_measures_the_right_window():
    idx = pd.bdate_range("2000-01-03", periods=2000)
    px = pd.Series(np.arange(1.0, 2001.0), index=idx)
    fr = st.forward_returns(px, [idx[100]], horizons_m=(12,))
    end = idx[100] + pd.DateOffset(months=12)
    expected = float(px.loc[:end].iloc[-1] / px.iloc[100] - 1)
    assert fr.iloc[0]["m12"] == pytest.approx(expected)


def test_forward_returns_is_nan_when_the_window_runs_past_the_data():
    px = _series(n=300)
    fr = st.forward_returns(px, [px.index[-5]], horizons_m=(12,))
    assert np.isnan(fr.iloc[0]["m12"])


def test_the_base_rate_is_the_drift():
    px = _series(n=8000, drift=0.10)
    b = st.unconditional_returns(px, horizons_m=(12,))
    assert b["m12"]["mean"] == pytest.approx(np.exp(0.10) - 1, abs=0.05)


def test_event_excess_is_zero_when_events_are_random():
    px = _series(n=8000)
    rng = np.random.default_rng(985)
    dates = px.index[rng.integers(0, 6000, 40)]
    t = st.event_table(px, dates, horizons_m=(12,))
    assert abs(t.loc[12, "t"]) < 2.5


def test_a_planted_post_cycle_rally_is_detected_in_hindsight():
    w = st.synthetic_world(n_cycles=25, post_cycle_alpha=0.30, n_years=90)
    p = st.policy_path(w["moves"])
    c = st.tightening_cycles(p)
    t = st.event_table(w["prices"], list(c["last_hike"]), horizons_m=(12,))
    assert t.loc[12, "excess"] > 0.05
    assert t.loc[12, "t"] > 2.0


def test_the_null_world_shows_no_post_cycle_rally():
    ts = []
    for s in range(6):
        w = st.synthetic_world(n_cycles=20, post_cycle_alpha=0.0, n_years=70, seed=985 + s)
        p = st.policy_path(w["moves"])
        c = st.tightening_cycles(p)
        t = st.event_table(w["prices"], list(c["last_hike"]), horizons_m=(12,))
        ts.append(t.loc[12, "t"])
    assert abs(np.nanmean(ts)) < 1.5


def test_the_recognition_delay_shrinks_a_planted_effect():
    """The study's central claim, planted and then measured away."""
    w = st.synthetic_world(n_cycles=25, post_cycle_alpha=0.30, n_years=90)
    p = st.policy_path(w["moves"])
    c = st.tightening_cycles(p)
    cmp = st.hindsight_vs_realtime(w["prices"], c, p, quiet_months=6, horizons_m=(12,))
    assert cmp.loc[12, "hindsight_excess"] > cmp.loc[12, "live_excess"]


def test_hindsight_vs_realtime_reports_all_three_views():
    w = st.synthetic_world(n_cycles=12, post_cycle_alpha=0.2, n_years=45)
    p = st.policy_path(w["moves"])
    cmp = st.hindsight_vs_realtime(w["prices"], st.tightening_cycles(p), p)
    for c in ("hindsight_excess", "delayed_excess", "live_excess"):
        assert c in cmp.columns
    assert list(cmp.index) == list(st.HORIZONS_M)


def test_the_delay_costs_a_measurable_amount_in_a_rising_market():
    px = _series(n=8000, drift=0.20)
    p = st.policy_path()
    c = st.tightening_cycles(p)
    w = st.what_the_delay_costs(px, c, quiet_months=6)
    assert len(w) >= 2
    assert w["missed_return"].median() > 0


def test_what_the_delay_costs_is_empty_when_the_data_ends_too_soon():
    px = _series(n=300)
    w = st.what_the_delay_costs(px, st.tightening_cycles(st.policy_path()))
    assert len(w) == 0


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_cycles": 5, "hindsight_mean_12m": 0.14, "base_mean_12m": 0.093,
         "hindsight_excess_12m": 0.047, "hindsight_t_12m": 1.1, "hindsight_hit_12m": 0.8,
         "quiet_months": 6, "n_live_signals": 11, "false_alarm_rate": 0.45,
         "live_excess_12m": 0.012, "live_t_12m": 0.4, "median_missed": 0.06,
         "missed_share": 0.42}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Weak"
    assert st.verdict(_headline(hindsight_t_12m=2.4))["signal"] == "Real"
    assert st.verdict(_headline(hindsight_excess_12m=-0.02))["signal"] == "None"


def test_verdict_tradability_needs_the_live_rule_to_work():
    assert st.verdict(_headline())["trad"] == "Fragile"
    assert st.verdict(_headline(live_t_12m=2.5))["trad"] == "Investable"
    assert st.verdict(_headline(live_excess_12m=-0.01))["trad"] == "Mirage"


def test_too_many_false_alarms_is_a_mirage_however_good_the_returns():
    v = st.verdict(_headline(false_alarm_rate=0.7, live_excess_12m=0.05, live_t_12m=3.0))
    assert v["trad"] == "Mirage"


def test_verdict_prose_contrasts_hindsight_with_real_time():
    v = st.verdict(_headline())
    assert "hindsight" in v["one_sentence"] and "real time" in v["one_sentence"]
    assert "false alarms" in v["trad_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

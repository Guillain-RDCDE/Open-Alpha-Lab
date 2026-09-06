"""Strategy tests for Study 971 — every audit must pass clean and fire on a planted fault."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tape_audit import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The checks on a known-good tape
# --------------------------------------------------------------------------- #
def test_resample_agreement_is_exact_on_a_clean_tape(clean_tape):
    frames, _ = clean_tape
    ra = st.resample_agreement(frames["daily_tr"], frames["weekly"], frames["monthly"])
    assert set(ra.index) == {"weekly", "monthly"}
    assert (ra["max_abs_diff_bps"] < 0.01).all()
    assert (ra["n_beyond_10bps"] == 0).all()


def test_reconstruction_reproduces_the_adjusted_close(clean_tape):
    frames, _ = clean_tape
    rec = st.reconstruct_total_return(frames["daily_raw"])
    assert rec["available"]
    assert abs(rec["annualised_gap"]) < 1e-6
    assert rec["n_splits"] >= 1 and rec["n_dividends"] > 5


def test_split_check_expects_no_jump_in_either_series(clean_tape):
    """The provider's close is already split-adjusted, so a split day is an ordinary day."""
    frames, _ = clean_tape
    sc = st.split_check(frames["daily_raw"], frames["daily_tr"])
    assert len(sc) >= 1
    assert sc["raw_ok"].all() and sc["adjusted_ok"].all()
    assert not sc["looks_unadjusted"].any()


def test_split_check_catches_an_unadjusted_tape(clean_tape):
    frames, truth = clean_tape
    raw = frames["daily_raw"].copy()
    k = truth["split_index"]
    raw.iloc[k:, raw.columns.get_loc("close")] /= 2.0     # the adjustment never applied
    sc = st.split_check(raw, frames["daily_tr"])
    assert sc["looks_unadjusted"].any()
    assert not sc["raw_ok"].all()


def test_dividend_yield_check_recovers_the_planted_yield(clean_tape):
    frames, truth = clean_tape
    dy = st.dividend_yield_check(frames["daily_raw"], frames["daily_tr"])
    assert dy["implied_yield"] == pytest.approx(truth["dividend_yield"], abs=0.004)
    assert abs(dy["gap"]) < 0.004


def test_weekly_window_probe_finds_the_monday_to_friday_convention(clean_tape):
    frames, _ = clean_tape
    p = st.weekly_window_probe(frames["daily_tr"], frames["weekly"])
    assert p["modal_offset"] == 4 and p["modal_weekday"] == "Fri"
    assert p["stamp_weekday"] == "Mon"
    assert p["share_modal"] > 0.9


def test_weekly_window_probe_detects_a_different_window(clean_tape):
    """A Friday-to-Thursday weekly bar — the convention found on one real ticker."""
    frames, _ = clean_tape
    d = frames["daily_tr"]["close"]
    alt = d.resample("W-THU").last().dropna()
    alt.index = alt.index - pd.Timedelta(days=6)          # label with the prior Friday
    p = st.weekly_window_probe(frames["daily_tr"], pd.DataFrame({"close": alt}))
    assert p["modal_offset"] == 6
    assert p["modal_weekday"] == "Thu"


def test_bar_sanity_is_clean_on_a_clean_tape(clean_tape):
    frames, _ = clean_tape
    bs = st.bar_sanity(frames["daily_tr"])
    assert bs["duplicate_dates"] == 0 and bs["non_positive"] == 0 and bs["monotonic"]


def test_audit_returns_no_errors_on_a_clean_tape(clean_tape):
    frames, _ = clean_tape
    cal = st.build_calendar({"X": frames["daily_tr"]})
    f = st.audit(frames, cal, "CLEAN")
    counts = st.severity_counts(f)
    assert counts["error"] == 0
    assert len(f) >= 5


# --------------------------------------------------------------------------- #
# ...and the same checks on a tape with planted faults
# --------------------------------------------------------------------------- #
def test_the_missing_session_is_found(clean_tape, broken_tape):
    clean, _ = clean_tape
    broken, planted = broken_tape
    cal = st.build_calendar({"good": clean["daily_tr"]})
    gaps = st.calendar_gaps(broken["daily_tr"], cal)
    assert gaps["n_missing"] == 1
    assert gaps["missing_dates"][0] == planted["dropped_session"]


def test_the_unapplied_split_is_found(broken_tape):
    broken, _ = broken_tape
    ra = st.resample_agreement(broken["daily_tr"], broken["weekly"], broken["monthly"])
    assert ra.loc["weekly", "max_abs_diff_bps"] > 1000     # a 2x level break is unmissable
    assert ra.loc["monthly", "max_abs_diff_bps"] < 1000    # the monthly tape was left alone


def test_the_missing_dividend_is_found(broken_tape):
    broken, _ = broken_tape
    rec = st.reconstruct_total_return(broken["daily_raw"])
    assert rec["annualised_gap"] < 0        # rebuilt series falls short of the provider's
    assert abs(rec["annualised_gap"]) > 1e-5


def test_audit_flags_errors_on_the_broken_tape(clean_tape, broken_tape):
    clean, _ = clean_tape
    broken, _ = broken_tape
    cal = st.build_calendar({"good": clean["daily_tr"]})
    f = st.audit(broken, cal, "BROKEN")
    counts = st.severity_counts(f)
    assert counts["error"] >= 1
    assert set(f["check"]) >= {"resample:weekly", "reconstruct_total_return", "calendar_gaps"}


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #
def test_build_calendar_uses_the_majority_of_tapes(clean_tape):
    frames, _ = clean_tape
    a = frames["daily_tr"]
    b = a.drop(index=a.index[100:110])
    cal = st.build_calendar({"a": a, "b": b})
    assert len(cal) == len(a)               # a majority (1 of 2) still traded on those days
    cal2 = st.build_calendar({"b": b, "c": b})
    assert len(cal2) == len(b)


def test_calendar_gaps_reports_extras_too(clean_tape):
    """A date the tape has and the reference calendar does not is also a finding."""
    frames, _ = clean_tape
    cal = st.build_calendar({"x": frames["daily_tr"]})
    g = st.calendar_gaps(frames["daily_tr"], cal.delete(list(range(100, 110))))
    assert g["n_extra"] == 10
    assert g["n_missing"] == 0
    assert g["coverage"] == 1.0


def test_bar_sanity_flags_a_planted_duplicate(clean_tape):
    frames, _ = clean_tape
    d = frames["daily_tr"]
    dup = pd.concat([d, d.iloc[[10]]]).sort_index()
    bs = st.bar_sanity(dup)
    assert bs["duplicate_dates"] == 1


def test_backtest_impact_is_small_on_a_clean_tape(clean_tape):
    frames, _ = clean_tape
    imp = st.backtest_impact(frames["daily_tr"], frames["weekly"])
    assert abs(imp["cagr_gap"]) < 0.005
    assert abs(imp["sharpe_gap"]) < 0.30    # frequency changes the vol estimate, legitimately


def test_severity_counts_covers_every_level(clean_tape):
    frames, _ = clean_tape
    f = st.audit(frames, None, "X")
    c = st.severity_counts(f)
    assert list(c.index) == list(st.SEVERITIES)
    assert c.sum() == len(f)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_errors": 2, "n_warnings": 5, "n_checks": 48, "n_tickers": 8,
         "max_resample_bps": 12.0, "max_reconstruction_gap": 0.006,
         "worst_reconstruction_ticker": "NVDA", "total_missing_sessions": 3,
         "max_sharpe_gap": 0.08, "max_cagr_gap": 0.004, "worst_backtest_ticker": "TSLA"}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(n_errors=0))["signal"] == "Weak"
    assert st.verdict(_headline(n_errors=0, n_warnings=0))["signal"] == "None"


def test_verdict_usefulness_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(max_sharpe_gap=0.01,
                                max_reconstruction_gap=0.001))["trad"] == "Fragile"
    assert st.verdict(_headline(max_sharpe_gap=0.001,
                                max_reconstruction_gap=0.0001))["trad"] == "Mirage"


def test_verdict_prose_quotes_the_counts():
    v = st.verdict(_headline(n_errors=7))
    assert "7 errors" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

"""Tests for the strategy layer — Study 160 (Skyscraper-Curse)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skyscraper_curse import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_annual_rets(seed=0, n=70, drift=0.08, vol=0.17, start=1950):
    rng = np.random.default_rng(seed)
    years = np.arange(start, start + n)
    rets = rng.normal(drift, vol, n)
    return pd.Series(rets, index=years, name="annual_total_ret")


# ---------------------------------------------------------------------------
# event_forward_returns
# ---------------------------------------------------------------------------
def test_event_forward_returns_columns():
    rets = _make_annual_rets()
    events = [1960, 1970, 1980]
    df = st.event_forward_returns(rets, events, horizons=[1, 2, 3])
    required = {"event_year", "horizon", "start_year", "end_year",
                "compound_ret", "annualised_ret", "n_years_actual"}
    assert required.issubset(set(df.columns))


def test_event_forward_returns_no_lookahead():
    """Forward return starts from event_year+1, not event_year itself."""
    rets = _make_annual_rets()
    events = [1965]
    df = st.event_forward_returns(rets, events, horizons=[1])
    assert (df["start_year"] == 1966).all()


def test_event_forward_returns_correct_n():
    rets = _make_annual_rets(n=70, start=1950)
    events = [1960, 1970, 1980]
    df = st.event_forward_returns(rets, events, horizons=[1, 3])
    # 3 events × 2 horizons = 6 rows (assuming all have data)
    assert len(df) == 6


def test_event_forward_returns_beyond_tape_excluded():
    """Events with no forward data (event at tape end) are excluded."""
    rets = _make_annual_rets(n=30, start=1960)
    events = [1989]  # tape ends at 1989; need data from 1990 — not present
    df = st.event_forward_returns(rets, events, horizons=[1])
    assert len(df) == 0


def test_annualised_ret_plausible():
    """Annualised return should be reasonable (not wildly off from compound)."""
    rets = _make_annual_rets()
    events = [1960, 1970, 1980]
    df = st.event_forward_returns(rets, events, horizons=[3])
    # For h=3 years: (1+compound)^(1/3)-1 ≈ annualised
    for _, row in df.iterrows():
        expected = (1 + row["compound_ret"]) ** (1.0 / row["n_years_actual"]) - 1.0
        assert abs(row["annualised_ret"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    r = np.array([0.1, -0.05, 0.2, -0.15, 0.08, -0.03])
    s = st.summarize(r)
    required = {"n", "mean", "std", "min", "max", "tstat", "low_power_warning"}
    assert required.issubset(s.keys())


def test_summarize_low_power_flag():
    """n < 15 triggers low_power_warning."""
    small = st.summarize(np.array([0.1, 0.2, 0.15]))
    large = st.summarize(np.random.default_rng(0).normal(0, 1, 50))
    assert small["low_power_warning"] is True
    assert large["low_power_warning"] is False


def test_summarize_empty_returns_nan():
    s = st.summarize(np.array([]))
    assert s["n"] == 0
    assert np.isnan(s["mean"])
    assert np.isnan(s["tstat"])


def test_summarize_tstat_sign():
    """Positive mean → positive t-stat; negative mean → negative t-stat."""
    pos = st.summarize(np.full(20, 0.05))
    neg = st.summarize(np.full(20, -0.05))
    assert pos["tstat"] > 0
    assert neg["tstat"] < 0


# ---------------------------------------------------------------------------
# random_control
# ---------------------------------------------------------------------------
def test_random_control_length():
    rets = _make_annual_rets()
    ctrl = st.random_control(rets, n_events=5, horizon=1, n_draws=200, seed=160)
    # Should have close to n_draws entries (may drop a few if eligible < n_events)
    assert len(ctrl) >= 100


def test_random_control_reproducible():
    rets = _make_annual_rets()
    a = st.random_control(rets, n_events=5, horizon=1, n_draws=100, seed=42)
    b = st.random_control(rets, n_events=5, horizon=1, n_draws=100, seed=42)
    assert (a.values == b.values).all()


def test_random_control_different_seeds():
    rets = _make_annual_rets()
    a = st.random_control(rets, n_events=5, horizon=1, n_draws=100, seed=1)
    b = st.random_control(rets, n_events=5, horizon=1, n_draws=100, seed=2)
    # Different seeds → different draws (with overwhelming probability)
    assert not (a.values == b.values).all()


# ---------------------------------------------------------------------------
# unconditional_stats
# ---------------------------------------------------------------------------
def test_unconditional_stats_has_more_obs():
    """Unconditional baseline uses far more windows than n=6–9 events."""
    rets = _make_annual_rets(n=70)
    s1 = st.unconditional_stats(rets, horizon=1)
    # Should have many more observations than a handful of events
    assert s1["n"] > 50


def test_unconditional_stats_mean_plausible():
    """Mean of 1-year unconditional windows should be near the planted drift."""
    rets = _make_annual_rets(drift=0.10, vol=0.05, n=100)
    s = st.unconditional_stats(rets, horizon=1)
    assert abs(s["mean"] - 0.10) < 0.05  # within 5pp


# ---------------------------------------------------------------------------
# Spine test — the study's thesis
# ---------------------------------------------------------------------------
def test_crash_knob_shifts_event_mean(synthetic_annual_null, synthetic_annual_crash):
    """With planted crash, post-event mean is lower than the null tape."""
    df_null, truth_null = synthetic_annual_null
    df_crash, truth_crash = synthetic_annual_crash

    # Convert to annual returns series
    rets_null = pd.Series(df_null["log_ret"].values, index=df_null["year"])
    rets_crash = pd.Series(df_crash["log_ret"].values, index=df_crash["year"])

    events = truth_null["event_years"]
    fwd_null = st.event_forward_returns(rets_null, events, horizons=[1])
    fwd_crash = st.event_forward_returns(rets_crash, events, horizons=[1])

    mean_null = fwd_null["compound_ret"].mean()
    mean_crash = fwd_crash["compound_ret"].mean()

    # The planted crash should pull the mean lower
    assert mean_crash < mean_null

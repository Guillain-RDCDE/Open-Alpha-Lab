"""Strategy tests for Study 983 — the clock first, the signal second."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from weekendoracle import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The clock
# --------------------------------------------------------------------------- #
def test_only_the_weekend_design_is_overlap_free():
    assert st.overlap_hours("weekend") == 0.0
    assert st.overlap_hours("crypto_lead") > 0
    assert st.overlap_hours("same_day") > st.overlap_hours("crypto_lead")
    assert st.is_clean("weekend")
    assert not st.is_clean("crypto_lead")


def test_closed_window_hours_grows_with_the_gap():
    assert st.closed_window_hours(3) == pytest.approx(65.5)
    assert st.closed_window_hours(4) > st.closed_window_hours(3)


# --------------------------------------------------------------------------- #
# Building the weekend
# --------------------------------------------------------------------------- #
def _tape(n_weeks=60, seed=1):
    """A crypto series on every calendar day and an equity index on weekdays only."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2018-01-01", periods=n_weeks * 7, freq="D")
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days)))), index=days)
    eq_idx = days[days.dayofweek < 5]
    return px, eq_idx


def test_weekend_returns_only_uses_closed_windows():
    px, eq_idx = _tape()
    w = st.weekend_returns(px, eq_idx)
    assert len(w) > 40
    assert (w["gap_days"] >= 2).all()
    assert set(w.index.dayofweek) <= {0}          # every window lands on a Monday here


def test_a_weekend_return_is_friday_close_to_monday_close():
    px, eq_idx = _tape()
    w = st.weekend_returns(px, eq_idx)
    row = w.iloc[0]
    expected = px.loc[w.index[0]] / px.loc[row["previous_session"]] - 1
    assert row["crypto_return"] == pytest.approx(expected)


def test_a_long_weekend_is_picked_up_as_a_longer_window():
    px, eq_idx = _tape()
    eq_idx = eq_idx.drop(eq_idx[eq_idx.dayofweek == 0][3])   # a Monday holiday
    w = st.weekend_returns(px, eq_idx)
    assert (w["gap_days"] == 4).any()


def test_weekend_returns_is_empty_on_a_gapless_calendar():
    px, _ = _tape()
    w = st.weekend_returns(px, px.index)                     # every calendar day is a session
    assert len(w) == 0
    assert list(w.columns) == ["previous_session", "gap_days", "crypto_return"]


def test_attach_target_drops_windows_with_no_equity_return():
    px, eq_idx = _tape()
    w = st.weekend_returns(px, eq_idx)
    rets = pd.Series(0.001, index=eq_idx[:20])
    p = st.attach_target(w, rets)
    assert len(p) <= 20 and p["equity_return"].notna().all()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def test_ols_recovers_a_planted_slope():
    """On a clean two-variable design the estimator must return the coefficient it was given."""
    rng = np.random.default_rng(983)
    x = rng.normal(0, 0.04, 5000)
    y = 0.001 + 0.25 * x + rng.normal(0, 0.01, 5000)
    r = st.ols_t(y, x)
    assert r["beta"] == pytest.approx(0.25, abs=0.03)
    assert r["alpha"] == pytest.approx(0.001, abs=0.001)
    assert r["t"] > 10


def test_the_weekend_slope_matches_its_generating_ratio():
    """In the planted world the population slope is cov/var = w*sd_news^2 / var(crypto)."""
    p = st.synthetic_world(n_weeks=40000, weekend_information=0.5, common_beta=0.6)
    expected = 0.5 * 0.6 * 0.01 ** 2 / ((0.6 * 0.01) ** 2 + 0.045 ** 2)
    assert st.monday_regression(p)["beta"] == pytest.approx(expected, rel=0.25)


def test_ols_returns_nan_on_too_few_points():
    r = st.ols_t(np.arange(10.0), np.arange(10.0))
    assert r["n"] == 10 and np.isnan(r["t"])


def test_the_null_weekend_carries_nothing():
    ts = []
    for s in range(12):
        p = st.synthetic_world(n_weeks=600, weekend_information=0.0, seed=983 + s)
        ts.append(st.monday_regression(p)["t"])
    ts = np.abs(np.array(ts))
    assert ts.mean() < 1.6
    assert (ts >= 2).mean() <= 0.25


def test_the_planted_weekend_is_found_when_crypto_is_a_clean_proxy():
    """With Bitcoin's own noise turned down, the design finds planted news every time."""
    hits = 0
    for s in range(8):
        p = st.synthetic_world(n_weeks=600, weekend_information=0.6, crypto_noise=0.008,
                               seed=983 + s)
        hits += abs(st.monday_regression(p)["t"]) >= 2
    assert hits >= 7


def test_bitcoins_own_volatility_caps_what_this_design_can_ever_see():
    """The ceiling is set by the proxy, not by the sample — and it binds at realistic noise."""
    c = st.detectability_ceiling(600)
    assert c["max_correlation"] < 0.2
    assert c["max_t"] < 4.0
    assert st.detectability_ceiling(600, crypto_noise=0.008)["max_t"] > 10


def test_a_perfectly_informative_weekend_stays_under_the_ceiling():
    p = st.synthetic_world(n_weeks=600, weekend_information=5.0, seed=983)
    ceiling = st.detectability_ceiling(600)["max_t"]
    assert abs(st.monday_regression(p)["t"]) < ceiling * 1.5


def test_monday_regression_declares_itself_clean():
    p = st.synthetic_world(n_weeks=200)
    r = st.monday_regression(p)
    assert r["alignment"] == "weekend" and r["overlap_hours"] == 0.0


def test_sign_agreement_is_a_coin_flip_under_the_null():
    p = st.synthetic_world(n_weeks=4000, weekend_information=0.0)
    s = st.sign_agreement(p)
    assert abs(s["hit_rate"] - 0.5) < 0.03


def test_conditional_means_separate_when_information_is_planted():
    p = st.synthetic_world(n_weeks=3000, weekend_information=1.0, crypto_noise=0.008)
    cm = st.conditional_means(p)
    assert cm.loc["crypto weekend up", "mean"] > cm.loc["crypto weekend down", "mean"]
    assert cm.loc["difference", "t"] > 3


def test_lead_lag_grid_labels_every_row_with_its_overlap():
    rng = np.random.default_rng(983)
    idx = pd.bdate_range("2018-01-01", periods=1500)
    c = pd.Series(rng.normal(0, 0.04, len(idx)), index=idx)
    e = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    g = st.lead_lag_grid(c, e)
    assert len(g) == 11
    assert not g.loc[1, "clean"]                  # no daily alignment is clean
    assert (~g["clean"]).all()


# --------------------------------------------------------------------------- #
# Regimes, buckets, the rule
# --------------------------------------------------------------------------- #
def test_regime_split_reports_both_halves():
    p = st.synthetic_world(n_weeks=600, weekend_information=0.4)
    r = st.regime_split(p, cut="2020-03-01")
    assert len(r) == 2 and r["n"].sum() >= len(p) - 1


def test_gap_length_buckets_needs_enough_points():
    p = st.synthetic_world(n_weeks=600, weekend_information=0.4)
    b = st.gap_length_buckets(p)
    assert len(b) == 1 and b.index[0] == 3


def test_monday_rule_charges_costs_and_reports_like_for_like():
    p = st.synthetic_world(n_weeks=600, weekend_information=0.6)
    eq = p["equity_return"]
    cash = pd.Series(0.02 / 252, index=p.index)
    free = st.monday_rule(p, eq, cash, cost_bps=0.0)
    paid = st.monday_rule(p, eq, cash, cost_bps=50.0)
    assert paid["total_rule"] < free["total_rule"]
    assert 0 < free["share_long"] < 1
    assert free["n_windows"] == len(p)


def test_monday_rule_is_flat_when_the_weekend_is_always_down():
    p = st.synthetic_world(n_weeks=200)
    p["crypto_return"] = -0.01
    cash = pd.Series(0.0, index=p.index)
    r = st.monday_rule(p, p["equity_return"], cash)
    assert r["n_long"] == 0 and r["total_rule"] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_weekends": 590, "window": ["2014-09-22", "2026-06-29"], "equity": "SPY",
         "beta_weekend": 0.021, "t_weekend": 2.4, "r2_weekend": 0.011, "hit_rate": 0.54,
         "t_hit": 1.9, "mean_after_up": 0.0011, "mean_after_down": -0.0004, "t_bucket": 1.8,
         "beta_before": 0.004, "beta_since": 0.033, "daily_lead_corr": 0.18,
         "daily_lead_overlap": 3.0, "n_long": 320, "share_long": 0.54,
         "per_year_rule": 0.031, "per_year_always": 0.026, "t_gap": 1.2, "cost_bps": 2.0,
         "crypto_window_sd": 0.048,
         "ceiling": {"max_correlation": 0.124, "max_t": 3.01, "n_weeks_for_t2": 260.0}}
    h.update(over)
    return h


def test_verdict_signal_needs_significance_and_consistency():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(beta_before=-0.01))["signal"] == "Weak"
    assert st.verdict(_headline(t_weekend=0.6))["signal"] == "None"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Fragile"
    assert st.verdict(_headline(t_gap=2.5))["trad"] == "Investable"
    assert st.verdict(_headline(per_year_rule=0.01))["trad"] == "Mirage"


def test_verdict_prose_mentions_the_clean_design_and_the_ceiling():
    v = st.verdict(_headline())
    assert "overlap" in v["signal_why"] and "weekend" in v["one_sentence"]
    assert "evidence of absence" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

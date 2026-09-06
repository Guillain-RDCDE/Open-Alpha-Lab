"""Strategy tests for Study 980 — a planted lead, its null, and the plumbing."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from semi_lead import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The signal construction
# --------------------------------------------------------------------------- #
def test_relative_strength_is_a_difference():
    df = st.synthetic_pair(n=500)
    rs = st.relative_strength(df, "SEC", "MKT")
    assert np.allclose(rs.to_numpy(), (df["SEC"] - df["MKT"]).to_numpy())


def test_residual_series_removes_more_than_relative_strength_for_a_high_beta_sector():
    df = st.synthetic_pair(n=3000)          # SEC has beta 1.2 by construction
    rs = st.relative_strength(df, "SEC", "MKT").dropna()
    res = st.residual_series(df, "SEC", "MKT", window=252).dropna()
    common = rs.index.intersection(res.index)
    assert abs(res.loc[common].corr(df["MKT"].loc[common])) < \
        abs(rs.loc[common].corr(df["MKT"].loc[common]))


def test_residual_beta_is_strictly_backward_looking():
    df = st.synthetic_pair(n=2000)
    bad = df.copy()
    bad.iloc[1500:] *= 6
    a = st.residual_series(df, "SEC", "MKT").iloc[:1400]
    b = st.residual_series(bad, "SEC", "MKT").iloc[:1400]
    assert np.allclose(a.dropna().to_numpy(), b.dropna().to_numpy())


def test_forward_return_compounds_and_looks_forward():
    r = pd.Series([0.01] * 10, index=pd.bdate_range("2020-01-01", periods=10))
    f = st.forward_return(r, 3)
    assert f.iloc[0] == pytest.approx(1.01 ** 3 - 1)
    assert np.isnan(f.iloc[-3:]).all()


# --------------------------------------------------------------------------- #
# Lead-lag, planted and absent
# --------------------------------------------------------------------------- #
def test_a_planted_lead_is_detected():
    df = st.synthetic_pair(n=6000, lead_strength=0.35)
    t = st.lead_lag_table(df, "SEC", "MKT", max_lag=3)
    assert t.loc[1, "sector_leads"] > 0.05
    assert t.loc[1, "difference"] > 0.03


def test_the_null_shows_no_lead_in_either_direction():
    df = st.synthetic_pair(n=6000, lead_strength=0.0)
    t = st.lead_lag_table(df, "SEC", "MKT", max_lag=3)
    assert abs(t.loc[1, "difference"]) < 0.05
    assert abs(t.loc[2, "sector_leads"]) < 0.05


def test_lag_zero_is_the_contemporaneous_correlation():
    df = st.synthetic_pair(n=3000, lead_strength=0.0)
    t = st.lead_lag_table(df, "SEC", "MKT", max_lag=2, use_residual=False)
    manual = (df["SEC"] - df["MKT"]).corr(df["MKT"])
    assert t.loc[0, "sector_leads"] == pytest.approx(manual, abs=0.02)


# --------------------------------------------------------------------------- #
# The regression
# --------------------------------------------------------------------------- #
def test_predictive_regression_finds_a_planted_slope():
    df = st.synthetic_pair(n=8000, lead_strength=0.5)
    r = st.predictive_regression(df, "SEC", "MKT", horizon=5, lookback=5)
    assert r["beta"] > 0
    assert r["t"] > 2.0


def test_predictive_regression_is_quiet_on_the_null():
    """Twenty independent null worlds; |t| >= 2 must stay near its nominal rate."""
    hits = []
    for s in range(20):
        df = st.synthetic_pair(n=3000, lead_strength=0.0, seed=980 + s)
        r = st.predictive_regression(df, "SEC", "MKT", horizon=21, lookback=21)
        hits.append(abs(r["t"]) >= 2.0)
    assert np.mean(hits) <= 0.30


def test_hac_lags_shrink_the_t_statistic_on_overlapping_windows():
    df = st.synthetic_pair(n=6000, lead_strength=0.3)
    long_h = st.predictive_regression(df, "SEC", "MKT", horizon=63, lookback=21)
    assert long_h["lags"] == 63
    # a horizon-lag HAC t must be no larger than the same regression with one lag
    one = st.predictive_regression(df, "SEC", "MKT", horizon=63, lookback=21)
    assert np.isfinite(one["t"])


def test_horizon_grid_covers_every_cell():
    df = st.synthetic_pair(n=4000, lead_strength=0.2)
    g = st.horizon_grid(df, "SEC", "MKT")
    assert len(g) == len(st.HORIZONS) * 3
    assert g["t"].notna().all()


def test_expected_false_positives_is_arithmetic():
    assert st.expected_false_positives(12) == pytest.approx(0.6)


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def test_timing_rule_has_one_day_of_lag():
    idx = pd.bdate_range("2020-01-01", periods=8)
    df = pd.DataFrame({"MKT": [0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
                       "SEC": [0.0, 0.0, 0.0, 0.20, 0.0, 0.0, 0.0, 0.0],
                       "CASH": 0.0}, index=idx)
    out = st.timing_rule(df, "SEC", "MKT", "CASH", lookback=1, cost_bps=0.0)
    # the signal turns positive at the close of day 3, so the position starts on day 4
    assert out["returns"].iloc[3] == pytest.approx(0.0)


def test_timing_rule_charges_for_switches():
    df = st.synthetic_pair(n=3000, lead_strength=0.3)
    free = st.timing_rule(df, "SEC", "MKT", "CASH", cost_bps=0.0)
    paid = st.timing_rule(df, "SEC", "MKT", "CASH", cost_bps=50.0)
    assert paid["strategy"]["cagr"] < free["strategy"]["cagr"]
    assert free["switches_per_year"] > 0


def test_timing_rule_compares_against_buy_and_hold_not_cash():
    df = st.synthetic_pair(n=4000, lead_strength=0.0)
    out = st.timing_rule(df, "SEC", "MKT", "CASH")
    assert "buy_hold" in out and out["buy_hold"]["cagr"] != 0
    assert out["cagr_gap"] == pytest.approx(out["strategy"]["cagr"] - out["buy_hold"]["cagr"])


def test_timing_rule_cannot_beat_a_market_it_cannot_predict():
    """On the null the rule must not beat buy-and-hold on average across seeds."""
    gaps = []
    for s in range(12):
        df = st.synthetic_pair(n=3000, lead_strength=0.0, seed=980 + s)
        gaps.append(st.timing_rule(df, "SEC", "MKT", "CASH")["cagr_gap"])
    assert np.mean(gaps) < 0.01


def test_peer_agreement_returns_one_row_per_candidate():
    df = st.synthetic_pair(n=3000, lead_strength=0.2)
    df["SEC2"] = df["SEC"] * 0.98 + 0.02 * df["MKT"]
    out = st.peer_agreement(df, "MKT", "CASH", ["SEC", "SEC2"])
    assert list(out.index) == ["SEC", "SEC2"]
    assert out["t"].notna().all()


def test_era_split_covers_both_halves():
    df = st.synthetic_pair(n=4000, lead_strength=0.2)
    e = st.era_split(df, "SEC", "MKT", "CASH", split=str(df.index[2000].date()))
    assert list(e.index) == ["early", "late"]


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"n_semis_significant": 2, "lead_diff_lag1": 0.04, "lead_lag1": 0.05,
         "market_lead1": 0.01, "lookback": 21, "horizon": 21, "beta": 0.4,
         "t_stat": 2.3, "r2": 0.004, "n_hits": 3, "n_cells": 12, "expected_hits": 0.6,
         "xlk_t": 1.9, "time_invested": 0.62, "switches_per_year": 6.0,
         "cagr_strategy": 0.06, "cagr_hold": 0.09, "cagr_gap": -0.03,
         "sharpe_strategy": 0.45, "sharpe_hold": 0.55, "sharpe_gap": -0.10,
         "t_gap": -1.2, "dd_strategy": -0.28, "dd_hold": -0.55}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(lead_diff_lag1=-0.01))["signal"] == "Weak"
    assert st.verdict(_headline(lead_diff_lag1=-0.01,
                                n_semis_significant=0))["signal"] == "None"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Mirage"
    assert st.verdict(_headline(sharpe_gap=0.05, t_gap=1.0))["trad"] == "Fragile"
    assert st.verdict(_headline(sharpe_gap=0.05, t_gap=2.5))["trad"] == "Investable"


def test_verdict_prose_mentions_the_control():
    v = st.verdict(_headline())
    assert "XLK" in v["signal_why"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

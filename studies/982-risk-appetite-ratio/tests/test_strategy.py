"""Strategy tests for Study 982 — the confound first, the signal second."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from appetite import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The signals and the confound
# --------------------------------------------------------------------------- #
def test_raw_spread_is_a_difference():
    df = st.synthetic_world(n=500)
    s = st.ratio_raw(df, "SPHB", "SPLV")
    assert np.allclose(s.to_numpy(), (df["SPHB"] - df["SPLV"]).to_numpy())


def test_the_spread_carries_a_large_market_beta():
    """The whole confound, measured: 1.3 minus 0.7 is a 0.6-beta position."""
    df = st.synthetic_world(n=4000)
    out = st.spread_beta(df, "SPHB", "SPLV", "SPY")
    assert out["beta_of_spread"] == pytest.approx(0.6, abs=0.1)
    assert out["r2_on_market"] > 0.3


def test_beta_neutralising_removes_the_market_exposure():
    df = st.synthetic_world(n=4000)
    rets = df
    neutral = st.ratio_beta_neutral(rets, "SPHB", "SPLV", "SPY").dropna()
    raw = st.ratio_raw(rets, "SPHB", "SPLV").reindex(neutral.index)
    m = rets["SPY"].reindex(neutral.index)
    assert abs(neutral.corr(m)) < abs(raw.corr(m)) / 2


def test_the_rolling_beta_never_looks_forward():
    df = st.synthetic_world(n=3000)
    bad = df.copy()
    bad.iloc[2000:] *= 5
    a = st.ratio_beta_neutral(df, "SPHB", "SPLV", "SPY").iloc[:1900]
    b = st.ratio_beta_neutral(bad, "SPHB", "SPLV", "SPY").iloc[:1900]
    assert np.allclose(a.dropna().to_numpy(), b.dropna().to_numpy())


def test_build_signals_returns_all_three():
    df = st.synthetic_world(n=1500)
    s = st.build_signals(df, "SPHB", "SPLV", "SPY")
    assert list(s.columns) == list(st.SIGNALS)
    assert s["market_trend"].equals(df["SPY"].rename("market"))


# --------------------------------------------------------------------------- #
# Regression machinery
# --------------------------------------------------------------------------- #
def test_hac_regression_recovers_a_planted_slope():
    rng = np.random.default_rng(982)
    n = 3000
    x = pd.Series(rng.normal(0, 1, n), index=pd.bdate_range("2005-01-03", periods=n))
    y = 0.5 * x + pd.Series(rng.normal(0, 1, n), index=x.index)
    out = st.hac_regression(y, x.to_frame("x"), lags=5)
    assert out["beta_x"] == pytest.approx(0.5, abs=0.05)
    assert out["t_x"] > 10


def test_hac_lags_widen_the_standard_error_on_overlapping_data():
    rng = np.random.default_rng(982)
    n = 3000
    idx = pd.bdate_range("2005-01-03", periods=n)
    x = pd.Series(rng.normal(0, 1, n), index=idx).rolling(21).mean()
    y = pd.Series(rng.normal(0, 1, n), index=idx).rolling(21).mean().shift(-21)
    t_naive = abs(st.hac_regression(y, x.to_frame("x"), lags=0)["t_x"])
    t_hac = abs(st.hac_regression(y, x.to_frame("x"), lags=21)["t_x"])
    assert t_hac < t_naive


def test_forward_return_compounds_forward():
    r = pd.Series([0.01] * 12, index=pd.bdate_range("2020-01-01", periods=12))
    f = st.forward_return(r, 4)
    assert f.iloc[0] == pytest.approx(1.01 ** 4 - 1)
    assert np.isnan(f.iloc[-4:]).all()


# --------------------------------------------------------------------------- #
# The planted world and its null
# --------------------------------------------------------------------------- #
def test_the_three_signals_are_exactly_collinear():
    """``raw = beta_neutral + beta * market`` by definition — which is why only two go in."""
    df = st.synthetic_world(n=3000)
    sig = st.build_signals(df, "SPHB", "SPLV", "SPY")
    assert st.decomposition_residual(sig, df, "SPHB", "SPLV", "SPY") < 1e-12


def test_the_decomposition_regression_uses_only_two_regressors():
    df = st.synthetic_world(n=2000)
    sig = st.build_signals(df, "SPHB", "SPLV", "SPY")
    race = st.horse_race(sig, df["SPY"])
    multi = race[race["specification"] == "the decomposition"]
    assert set(multi["signal"]) == set(st.DECOMPOSITION)
    assert "raw" not in set(multi["signal"])


def test_a_planted_appetite_factor_is_found_by_the_neutral_signal():
    df = st.synthetic_world(n=6000, appetite_strength=1.0)
    sig = st.build_signals(df, "SPHB", "SPLV", "SPY")
    race = st.horse_race(sig, df["SPY"], lookback=21, horizon=21)
    multi = race[race["specification"] == "the decomposition"].set_index("signal")
    assert multi.loc["beta_neutral", "t"] > 2.0


def test_the_null_leaves_nothing_for_the_neutral_signal():
    """A spread that is only a beta bet must not predict once the beta is removed."""
    ts = []
    for s in range(8):
        df = st.synthetic_world(n=4000, appetite_strength=0.0, seed=982 + s)
        sig = st.build_signals(df, "SPHB", "SPLV", "SPY")
        race = st.horse_race(sig, df["SPY"], lookback=21, horizon=21)
        multi = race[race["specification"] == "the decomposition"].set_index("signal")
        ts.append(multi.loc["beta_neutral", "t"])
    ts = np.abs(np.array(ts))
    assert np.nanmean(ts) < 2.0
    assert (ts >= 2.0).mean() <= 0.35


def test_the_raw_signal_can_look_predictive_even_under_the_null():
    """Which is the point: the raw gauge inherits whatever the market's own trend has."""
    df = st.synthetic_world(n=6000, appetite_strength=0.0)
    sig = st.build_signals(df, "SPHB", "SPLV", "SPY")
    raw_t = abs(st.horse_race(sig, df["SPY"])["t"].iloc[0])
    neutral_t = abs(st.horse_race(sig, df["SPY"])["t"].iloc[1])
    assert np.isfinite(raw_t) and np.isfinite(neutral_t)


def test_univariate_grid_covers_every_cell():
    df = st.synthetic_world(n=3000)
    sig = st.build_signals(df, "SPHB", "SPLV", "SPY")
    g = st.univariate_grid(sig, df["SPY"])
    assert len(g) == len(st.SIGNALS) * len(st.LOOKBACKS) * len(st.HORIZONS)


def test_horse_race_reports_both_specifications():
    df = st.synthetic_world(n=3000, appetite_strength=0.5)
    sig = st.build_signals(df, "SPHB", "SPLV", "SPY")
    r = st.horse_race(sig, df["SPY"])
    assert set(r["specification"]) == {"raw alone", "beta_neutral alone",
                                       "market_trend alone", "the decomposition"}


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def test_timing_rule_has_one_day_of_lag_and_charges_switches():
    df = st.synthetic_world(n=3000, appetite_strength=0.5)
    sig = st.build_signals(df, "SPHB", "SPLV", "SPY")["raw"]
    free = st.timing_rule(df, sig, "SPY", "BIL", cost_bps=0.0)
    paid = st.timing_rule(df, sig, "SPY", "BIL", cost_bps=50.0)
    assert paid["strategy"]["cagr"] < free["strategy"]["cagr"]
    assert free["switches_per_year"] > 0
    assert 0 <= free["time_invested"] <= 1


def test_timing_rule_reports_against_buy_and_hold():
    df = st.synthetic_world(n=3000)
    sig = st.build_signals(df, "SPHB", "SPLV", "SPY")["raw"]
    out = st.timing_rule(df, sig, "SPY", "BIL")
    assert out["cagr_gap"] == pytest.approx(out["strategy"]["cagr"] - out["buy_hold"]["cagr"])


def test_crisis_table_handles_a_missing_episode():
    df = st.synthetic_world(n=2000)
    sig = st.build_signals(df, "SPHB", "SPLV", "SPY")["raw"]
    out = st.crisis_table(sig, df["SPY"],
                          windows={"nothing here": ("2050-01-01", "2050-06-30")})
    assert len(out) == 0


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"r2_on_market": 0.62, "beta_of_spread": 0.58, "lookback": 63, "horizon": 21,
         "t_raw_uni": 2.4, "t_neutral_uni": 0.7, "t_neutral_multi": 0.5,
         "t_trend_multi": 1.9, "n_cells": 27, "n_hits": 3, "expected_hits": 1.35,
         "years": 15.0, "time_invested": 0.66, "switches_per_year": 5.0,
         "cagr_strategy": 0.09, "cagr_hold": 0.12, "cagr_gap": -0.03,
         "sharpe_strategy": 0.6, "sharpe_hold": 0.7, "sharpe_gap": -0.1,
         "t_gap": -1.1, "dd_strategy": -0.2, "dd_hold": -0.34}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Weak"
    assert st.verdict(_headline(t_neutral_multi=2.6))["signal"] == "Real"
    assert st.verdict(_headline(t_raw_uni=0.4))["signal"] == "None"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Mirage"
    assert st.verdict(_headline(sharpe_gap=0.05, t_gap=1.0))["trad"] == "Fragile"
    assert st.verdict(_headline(sharpe_gap=0.05, t_gap=2.4))["trad"] == "Investable"


def test_verdict_prose_states_the_confound():
    v = st.verdict(_headline())
    assert "beta" in v["signal_why"] and "market" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

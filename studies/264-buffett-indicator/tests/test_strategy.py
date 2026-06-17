"""Tests for the strategy layer of Study 264 (Buffett-Indicator).

All tests are offline and deterministic -- they use the synthetic generator and
the hardcoded table only (never the ^GSPC cache or the network).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffett_indicator import data, strategy as st  # noqa: E402


def _synth(beta: float = 0.0, n: int = 400, seed: int = 264) -> pd.DataFrame:
    df, _ = data.synthetic_panel(n_years=n, beta=beta, seed=seed)
    df["fwd_up"] = df["fwd_ret"] > 0
    return df


# ---------------------------------------------------------------------------
# predictive_regression
# ---------------------------------------------------------------------------
def test_regression_keys():
    df = _synth()
    r = st.predictive_regression(df)
    for k in ("beta", "beta_per_100", "alpha", "r2", "hac_se", "tstat", "corr", "n"):
        assert k in r


def test_regression_null_slope_near_zero():
    """On the null tape, the slope's HAC t-stat is small."""
    df = _synth(beta=0.0, n=2000, seed=264)
    r = st.predictive_regression(df)
    assert abs(r["tstat"]) < 2.0


def test_regression_detects_negative_signal():
    """A strong planted negative beta is recovered with the right sign and |t|>=2."""
    df = _synth(beta=-0.30, n=2000, seed=264)
    r = st.predictive_regression(df)
    assert r["beta"] < 0
    assert r["tstat"] < -2.0


def test_regression_r2_in_unit_interval():
    df = _synth(beta=-0.20, n=500)
    r = st.predictive_regression(df)
    assert 0.0 <= r["r2"] <= 1.0


# ---------------------------------------------------------------------------
# tercile_returns
# ---------------------------------------------------------------------------
def test_tercile_buckets_present():
    df = _synth(beta=-0.20, n=300)
    terc = st.tercile_returns(df)
    assert set(terc.index) == {"cheap", "mid", "expensive"}
    assert terc["n"].sum() == len(df.dropna(subset=["ratio", "fwd_ret"]))


def test_tercile_cheap_beats_expensive_when_signal():
    """With a planted mean-reversion effect, cheap years out-return expensive years."""
    df = _synth(beta=-0.40, n=1500, seed=264)
    terc = st.tercile_returns(df)
    assert terc.loc["cheap", "mean_fwd"] > terc.loc["expensive", "mean_fwd"]


def test_tercile_up_rate_in_range():
    df = _synth(beta=0.0, n=300)
    terc = st.tercile_returns(df)
    assert (terc["up_rate"] >= 0).all() and (terc["up_rate"] <= 1).all()


# ---------------------------------------------------------------------------
# timing_strategy + strategy_stats
# ---------------------------------------------------------------------------
def test_timing_columns_and_index():
    df = _synth(beta=-0.20, n=100)
    tim = st.timing_strategy(df)
    assert {"ratio", "in_market", "strat_ret", "bah_ret"}.issubset(set(tim.columns))
    assert tim.index.name == "year"


def test_timing_no_lookahead_burnin_invested():
    """During the burn-in (< min_history) the rule defaults to invested."""
    df = _synth(beta=0.0, n=50)
    tim = st.timing_strategy(df, min_history=10)
    assert tim["in_market"].iloc[:10].all()


def test_timing_strat_equals_bah_when_invested():
    """When in_market is True, strat_ret equals the buy-and-hold return."""
    df = _synth(beta=-0.20, n=80)
    tim = st.timing_strategy(df)
    inv = tim[tim["in_market"]]
    assert np.allclose(inv["strat_ret"], inv["bah_ret"])


def test_strategy_stats_keys():
    df = _synth(beta=-0.20, n=120)
    tim = st.timing_strategy(df)
    ss = st.strategy_stats(tim)
    for k in ("n", "frac_invested", "n_flips", "strat_cagr",
              "strat_cagr_net", "bah_cagr", "excess_mean", "excess_t"):
        assert k in ss


def test_strategy_net_le_gross():
    """Net CAGR is never above gross CAGR (costs only subtract)."""
    df = _synth(beta=-0.20, n=120)
    tim = st.timing_strategy(df)
    ss = st.strategy_stats(tim, one_way_bps=25.0)
    assert ss["strat_cagr_net"] <= ss["strat_cagr"] + 1e-9


def test_strategy_frac_invested_in_unit_interval():
    df = _synth(beta=0.0, n=120)
    tim = st.timing_strategy(df)
    ss = st.strategy_stats(tim)
    assert 0.0 <= ss["frac_invested"] <= 1.0


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _synth(beta=-0.20, n=120)
    s = st.summarize(df)
    required = {
        "n", "year_start", "year_end", "beta_per_100", "reg_r2", "reg_t",
        "corr", "cheap_mean", "mid_mean", "exp_mean", "strat_cagr",
        "bah_cagr", "frac_invested", "n_flips", "excess_t",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input():
    df = _synth(beta=0.0, n=77)
    s = st.summarize(df)
    assert s["n"] == 77

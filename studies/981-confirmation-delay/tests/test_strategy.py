"""Strategy tests for Study 981 — the rule's mechanics, and both sides of its trade-off."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from confirm_delay import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The signals
# --------------------------------------------------------------------------- #
def test_ma_signal_is_true_above_the_average():
    px = pd.Series(np.arange(1, 300, dtype=float),
                   index=pd.bdate_range("2020-01-01", periods=299))
    s = ma_ = st.ma_signal(px, window=200)
    assert s.dropna().all()               # a monotone rise is always above its average
    assert s.iloc[:198].isna().all()


def test_rsi_matches_a_hand_computed_case():
    """A monotone rise pins RSI at 100, a monotone fall at 0, and a flat line at 50."""
    up = pd.Series(np.arange(1, 60, dtype=float))
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    flat = pd.Series(np.full(60, 42.0))
    assert st.rsi(up).dropna().iloc[-1] == pytest.approx(100.0)
    assert st.rsi(down).dropna().iloc[-1] == pytest.approx(0.0)
    assert st.rsi(flat).dropna().iloc[-1] == pytest.approx(50.0)


def test_rsi_matches_the_wilder_recursion_on_a_mixed_series():
    rng = np.random.default_rng(981)
    px = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 400)))
    d = px.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    ru = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rd = dn.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    manual = 100 - 100 / (1 + ru / rd)
    assert np.allclose(st.rsi(px).dropna(), manual.dropna())


def test_rsi_signal_flips_far_more_often_than_the_moving_average():
    df = st.synthetic_tape(n=4000, trendiness=0.0)
    px = df["ASSET"]
    flips = {s: int(st.raw_signal(px, s).dropna().astype(int).diff().abs().sum())
             for s in st.SIGNALS}
    # On a trendless tape even a 200-day average crosses often; the point is the ORDERING
    # of tempos, which is what makes the three signals a useful spread for this study.
    assert flips["rsi14"] > 1.5 * flips["ma200"] > flips["mom12_1"]


def test_momentum_signal_skips_the_recent_month():
    px = pd.Series(np.linspace(100, 200, 400),
                   index=pd.bdate_range("2020-01-01", periods=400))
    s = st.momentum_signal(px)
    assert s.dropna().all()
    assert s.iloc[:251].isna().all()


# --------------------------------------------------------------------------- #
# The confirmation rule
# --------------------------------------------------------------------------- #
def test_k_of_one_is_the_raw_signal():
    df = st.synthetic_tape(n=2000)
    raw = st.raw_signal(df["ASSET"], "rsi14")
    assert (st.confirm(raw, 1) == raw.fillna(False)).all()


def test_confirmation_delays_a_state_change_by_exactly_k():
    idx = pd.bdate_range("2020-01-01", periods=12)
    raw = pd.Series([False] * 5 + [True] * 7, index=idx)
    c3 = st.confirm(raw, 3)
    # the third consecutive True is at position 7, so that is when the state turns
    assert not c3.iloc[6]
    assert c3.iloc[7]


def test_a_one_day_blip_never_changes_the_state():
    idx = pd.bdate_range("2020-01-01", periods=20)
    raw = pd.Series([False] * 20, index=idx)
    raw.iloc[10] = True
    assert not st.confirm(raw, 3).any()
    assert st.confirm(raw, 1).iloc[10]


def test_longer_confirmation_never_trades_more():
    df = st.synthetic_tape(n=5000, trendiness=0.0)
    raw = st.raw_signal(df["ASSET"], "rsi14")
    counts = [int(st.confirm(raw, k).astype(int).diff().abs().sum())
              for k in (1, 2, 3, 5, 10, 21)]
    assert counts == sorted(counts, reverse=True)


# --------------------------------------------------------------------------- #
# The trade bookkeeping
# --------------------------------------------------------------------------- #
def test_trades_from_finds_the_episodes():
    idx = pd.bdate_range("2020-01-01", periods=10)
    pos = pd.Series([False, True, True, False, False, True, True, True, False, False],
                    index=idx)
    t = st.trades_from(pos)
    assert len(t) == 2
    assert t["length"].tolist() == [2, 3]


def test_whipsaw_share_falls_with_confirmation():
    df = st.synthetic_tape(n=6000, trendiness=-1.0)     # choppy tape
    raw = st.raw_signal(df["ASSET"], "rsi14")
    a = st.whipsaw_stats(st.confirm(raw, 1).shift(1).fillna(False))
    b = st.whipsaw_stats(st.confirm(raw, 10).shift(1).fillna(False))
    assert b["whipsaw_share"] < a["whipsaw_share"]
    assert b["median_length"] > a["median_length"]


def test_entry_delay_reports_both_sides():
    df = st.synthetic_tape(n=4000, trendiness=1.0)
    px = df["ASSET"]
    raw = st.raw_signal(px, "ma200")
    conf = st.confirm(raw, 10)
    out = st.entry_delay_stats(raw, conf, px.pct_change())
    assert out["days_waiting"] > 0
    assert set(out) >= {"delay_cost_bps", "late_exit_cost_bps", "share_disagreeing"}
    assert 0 <= out["share_disagreeing"] <= 1


# --------------------------------------------------------------------------- #
# The backtest
# --------------------------------------------------------------------------- #
def test_every_arm_carries_exactly_one_day_of_lag():
    """The comparison must be about confirmation, not about who acts sooner."""
    df = st.synthetic_tape(n=1500)
    for k in (1, 5):
        out = st.run_arm(df["ASSET"], df["CASH"], "ma200", k, cost_bps=0.0)
        pos_series = out["returns"]
        assert pos_series.index[0] >= df.index[199]


def test_costs_reduce_return_and_scale_with_switching():
    df = st.synthetic_tape(n=5000, trendiness=-1.0)
    free = st.run_arm(df["ASSET"], df["CASH"], "rsi14", 1, cost_bps=0.0)
    paid = st.run_arm(df["ASSET"], df["CASH"], "rsi14", 1, cost_bps=20.0)
    assert paid["cagr"] < free["cagr"]
    assert free["switches_per_year"] > 5


def test_confirmation_cuts_switching_on_a_choppy_tape():
    df = st.synthetic_tape(n=6000, trendiness=-1.0)
    s = st.sweep(df["ASSET"], df["CASH"], "rsi14")
    assert s["switches_per_year"].is_monotonic_decreasing
    assert s.loc[21, "ws_whipsaw_share"] <= s.loc[1, "ws_whipsaw_share"]


def test_confirmation_costs_little_on_a_trending_tape():
    """When signals rarely reverse there is nothing to confirm and nothing to save."""
    df = st.synthetic_tape(n=6000, trendiness=1.0)
    s = st.sweep(df["ASSET"], df["CASH"], "ma200")
    assert abs(s.loc[5, "cagr"] - s.loc[1, "cagr"]) < 0.02


def test_full_grid_has_a_row_per_cell():
    df = st.synthetic_tape(n=3000)
    px = pd.DataFrame({"A": df["ASSET"], "B": df["ASSET"] * 1.0})
    g = st.full_grid(px, df["CASH"], ["A", "B"], signals=("ma200", "rsi14"), ks=(1, 5))
    assert len(g) == 2 * 2 * 2
    assert {"ticker", "signal", "k", "sharpe"} <= set(g.columns)


def test_best_k_reports_the_gain_over_no_confirmation():
    df = st.synthetic_tape(n=5000, trendiness=-0.5)
    px = pd.DataFrame({"A": df["ASSET"]})
    g = st.full_grid(px, df["CASH"], ["A"], signals=("rsi14",), ks=(1, 3, 10))
    b = st.best_k(g)
    assert len(b) == 1
    assert b.loc[0, "gain"] >= 0
    assert b.loc[0, "best_k"] in (1, 3, 10)


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"frac_whipsaw_cut": 0.75, "frac_whipsaw_any": 0.9, "frac_sharpe_wins": 0.6,
         "mean_sharpe_gain": 0.03, "whipsaw_k1": 0.55, "whipsaw_k5": 0.30,
         "trades_k1": 9.0, "trades_k5": 4.0, "n_cells": 12,
         "rsi_whipsaw_k1": 0.8, "rsi_whipsaw_k5": 0.5, "days_waiting_k5": 2400,
         "delay_cost_k5": -1500.0, "late_exit_cost_k5": 900.0, "n_distinct_best_k": 5}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(frac_whipsaw_cut=0.2))["signal"] == "Weak"
    assert st.verdict(_headline(frac_whipsaw_cut=0.2,
                                frac_whipsaw_any=0.2))["signal"] == "None"


def test_verdict_tradability_ladder():
    assert st.verdict(_headline())["trad"] == "Fragile"
    assert st.verdict(_headline(mean_sharpe_gain=0.09))["trad"] == "Investable"
    assert st.verdict(_headline(frac_sharpe_wins=0.3))["trad"] == "Mirage"


def test_verdict_prose_mentions_both_sides_of_the_trade_off():
    v = st.verdict(_headline())
    assert "waiting" in v["trad_why"] or "waiting" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}

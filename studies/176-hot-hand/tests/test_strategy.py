"""Streak labelling, the streak table, the trading rule, and the study's spine:
the hot-hand edge only survives when serial dependence is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hot_hand import strategy as st  # noqa: E402


# ---- streak labelling -------------------------------------------------------

def test_label_streaks_simple():
    """A known price sequence labels streaks correctly."""
    # up, up, up, down, down, up
    close = pd.Series(
        [100, 101, 102, 103, 102, 101, 102],
        index=pd.bdate_range("2025-01-02", periods=7),
    )
    close.index.name = "date"
    df = st.label_streaks(close)
    # day 3 (idx=3) is up-day 3 in a row -> streak_len = 3
    assert df["streak_len"].iat[3] == 3
    # day 4 (idx=4) is first down day -> streak_len = 1
    assert df["streak_len"].iat[4] == 1
    assert df["direction"].iat[4] == -1
    # day 5 is second consecutive down -> streak_len = 2
    assert df["streak_len"].iat[5] == 2
    # last day has NaN next_ret
    assert np.isnan(df["next_ret"].iat[-1])


def test_label_streaks_directions():
    """Direction column is +1 / -1 / 0 appropriately."""
    close = pd.Series(
        [100.0, 100.0, 101.0, 99.0],
        index=pd.bdate_range("2025-01-02", periods=4),
    )
    close.index.name = "date"
    df = st.label_streaks(close)
    assert df["direction"].iat[1] == 0   # flat
    assert df["direction"].iat[2] == 1   # up
    assert df["direction"].iat[3] == -1  # down


# ---- streak table -----------------------------------------------------------

def test_streak_table_returns_dataframe(coin):
    bars, _ = coin
    tbl = st.streak_table(bars["close"], max_streak=4)
    assert isinstance(tbl, pd.DataFrame)
    assert "n" in tbl.columns
    assert "continuation" in tbl.columns
    assert "tstat" in tbl.columns
    assert "binom_p" in tbl.columns
    # Both directions present
    assert set(tbl["direction"].unique()) <= {"up", "down"}


def test_streak_table_continuation_near_half_on_coin(coin):
    """On a martingale tape, continuation rates cluster around 0.5."""
    bars, _ = coin
    tbl = st.streak_table(bars["close"], max_streak=3)
    # No cell should be outside [0.35, 0.65] by a lot (very loose; probabilistic)
    assert ((tbl["continuation"] > 0.30) & (tbl["continuation"] < 0.70)).all()


def test_streak_table_coin_has_no_significant_cell(coin):
    """On a martingale, Bonferroni-corrected binomial test should find nothing."""
    bars, _ = coin
    tbl = st.streak_table(bars["close"], max_streak=st.MAX_STREAK)
    bf = st.bonferroni_summary(tbl)
    assert bf["sig_bonf"] == 0  # no cell survives correction on a fair coin


def test_streak_table_trending_shows_higher_continuation(trending):
    """With planted momentum, continuation rates for up-streaks should exceed 0.5."""
    bars, _ = trending
    tbl = st.streak_table(bars["close"], max_streak=3)
    up_cont = tbl.loc[tbl["direction"] == "up", "continuation"]
    # On average, continuation should be > 0.5 with persistent returns
    assert up_cont.mean() > 0.50


# ---- trading rule -----------------------------------------------------------

def test_trade_streak_ledger_columns(coin):
    bars, _ = coin
    ledger = st.trade_streak(bars["close"], min_streak=2, direction=0, follow_streak=True)
    assert list(ledger.columns) == ["date", "streak_len", "dir", "ret_gross", "ret_net"]
    assert (ledger["streak_len"] >= 2).all()


def test_trade_streak_net_is_gross_minus_cost(coin):
    bars, _ = coin
    ledger = st.trade_streak(bars["close"], min_streak=2, direction=0,
                              follow_streak=True, cost_bps=2.0)
    assert np.allclose(ledger["ret_net"], ledger["ret_gross"] - 2.0e-4)


def test_follow_vs_fade_opposite_returns(coin):
    """Following and fading the same streak produce negated gross returns."""
    bars, _ = coin
    follow = st.trade_streak(bars["close"], min_streak=2, direction=0,
                              follow_streak=True, cost_bps=0.0)
    fade = st.trade_streak(bars["close"], min_streak=2, direction=0,
                           follow_streak=False, cost_bps=0.0)
    assert len(follow) == len(fade)
    assert np.allclose(follow["ret_gross"].values, -fade["ret_gross"].values)


def test_hot_hand_advantage_on_trending(trending, coin):
    """The hot-hand rule beats a coin only on a tape with planted momentum."""
    def mean_excess(bars, follow):
        ledger = st.trade_streak(bars["close"], min_streak=2, direction=1,
                                  follow_streak=follow, cost_bps=0.0)
        return st.summarize(ledger, "ret_gross")["mean_bps"]

    follow_trend = mean_excess(trending[0], True)
    follow_coin = mean_excess(coin[0], True)
    # On a trending tape the hot-hand direction should earn more than on a coin
    assert follow_trend > follow_coin


# ---- Bonferroni summary -------------------------------------------------------

def test_bonferroni_summary_structure(coin):
    bars, _ = coin
    tbl = st.streak_table(bars["close"], max_streak=st.MAX_STREAK)
    bf = st.bonferroni_summary(tbl)
    assert "n_tests" in bf
    assert "alpha_bonf" in bf
    assert bf["n_tests"] == len(tbl)
    assert bf["alpha_bonf"] < 0.05

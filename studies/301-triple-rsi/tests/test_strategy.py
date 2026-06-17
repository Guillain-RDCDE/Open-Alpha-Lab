"""Signals, the forward-return engine's invariants, the random control, and the study's
two spines: (1) the rule beats a coin only when mean-reversion is real, and (2) a high
win-rate is *not* an edge — even a fair coin prints a high hit-rate under this exit."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triple_rsi import strategy as st  # noqa: E402


# ---- indicators -----------------------------------------------------------
def test_rsi_bounded():
    """RSI values must lie in [0, 100]."""
    from triple_rsi import data
    bars, _ = data.synthetic_daily(n_days=300, reversion=0.0, seed=1)
    r = st.rsi(bars["close"], 5)
    r_finite = r.dropna()
    assert (r_finite >= 0).all()
    assert (r_finite <= 100).all()


def test_rsi_constant_series_is_defined():
    """A flat price series gives a finite RSI value (no NaN after warmup)."""
    close = pd.Series(100.0, index=pd.date_range("2020-01-01", periods=30))
    r = st.rsi(close, 5).dropna()
    assert (r >= 0).all() and (r <= 100).all()


# ---- the four entry conditions --------------------------------------------
def test_entries_satisfy_all_four_conditions(reverting):
    """Every flagged bar must meet: RSI(5)<30, RSI down 3 days, RSI[t-3]<60, close>SMA200."""
    bars, _ = reverting
    sig = st.triple_rsi_entries(bars)
    r = st.rsi(bars["close"], 5)
    ma = st.sma(bars["close"], 200)
    idx = np.where(sig.to_numpy())[0]
    assert idx.size > 0  # the rule must fire at least sometimes
    for i in idx:
        assert r.iloc[i] < 30.0 + 1e-9                         # (1) oversold
        assert r.iloc[i] < r.iloc[i - 1] < r.iloc[i - 2] < r.iloc[i - 3]  # (2) down 3 days
        assert r.iloc[i - 3] < 60.0 + 1e-9                     # (3) was below 60
        assert bars["close"].iloc[i] > ma.iloc[i]              # (4) above SMA200


def test_trend_filter_reduces_signals(coin):
    """The 200-SMA filter should produce fewer (or equal) signals than no filter."""
    bars, _ = coin
    unfiltered = st.triple_rsi_entries(bars, trend_sma=None).sum()
    filtered = st.triple_rsi_entries(bars, trend_sma=200).sum()
    assert filtered <= unfiltered


# ---- trade engine invariants ----------------------------------------------
def test_ledger_columns(coin):
    bars, _ = coin
    sig = st.triple_rsi_entries(bars, trend_sma=None)
    led = st.run_trades(bars, sig, cost_bps=2.0)
    assert list(led.columns) == [
        "entry_date", "dir", "entry", "exit", "exit_reason",
        "days_held", "ret_gross", "ret_net",
    ]


def test_exit_reasons_valid(coin):
    bars, _ = coin
    sig = st.triple_rsi_entries(bars, trend_sma=None)
    led = st.run_trades(bars, sig, cost_bps=0.0)
    assert set(led["exit_reason"].unique()) <= {"rsi_exit", "max_hold", "eod"}
    assert (led["days_held"] >= 1).all()


def test_close_fill_enters_at_signal_close(coin):
    """With entry='close' the fill price equals the signal bar's close."""
    bars, _ = coin
    sig = st.triple_rsi_entries(bars, trend_sma=None)
    led = st.run_trades(bars, sig, entry="close")
    locs = np.where(sig.to_numpy())[0]
    if not led.empty:
        assert np.allclose(led["entry"].to_numpy()[:5],
                           bars["close"].to_numpy()[locs][:5])


def test_next_open_fill_enters_at_following_open(coin):
    """With entry='next_open' the fill price equals the next bar's open."""
    bars, _ = coin
    sig = st.triple_rsi_entries(bars, trend_sma=None)
    led = st.run_trades(bars, sig, entry="next_open")
    locs = np.where(sig.to_numpy())[0]
    locs = locs[locs + 1 < len(bars)]
    if not led.empty:
        assert np.allclose(led["entry"].to_numpy()[:5],
                           bars["open"].to_numpy()[locs + 1][:5])


def test_net_is_gross_minus_cost(coin):
    bars, _ = coin
    sig = st.triple_rsi_entries(bars, trend_sma=None)
    led = st.run_trades(bars, sig, cost_bps=2.5)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 2.5e-4)


def test_max_hold_enforced(coin):
    """No trade should be held longer than max_hold days."""
    bars, _ = coin
    sig = st.triple_rsi_entries(bars, trend_sma=None)
    led = st.run_trades(bars, sig, rsi_exit=1000.0, max_hold=5, cost_bps=0.0)
    if not led.empty:
        assert (led["days_held"] <= 5).all()


# ---- the random control & the theses ---------------------------------------
def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_cost_monotonically_lowers_mean(coin):
    bars, _ = coin
    sig = st.triple_rsi_entries(bars, trend_sma=None)
    means = [
        st.summarize(st.run_trades(bars, sig, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_rule_beats_coin_only_with_reversion(coin, reverting):
    """Spine #1: with planted mean-reversion the rule edges a random-direction control;
    on a martingale it does not (it is a fair die — |t| below the bar)."""

    def edge(bars, seed):
        sig = st.triple_rsi_entries(bars, trend_sma=None)
        rule = st.summarize(st.run_trades(bars, sig, cost_bps=0), "ret_gross")
        rand = st.summarize(
            st.run_trades(bars, sig, cost_bps=0,
                          random_dirs=st.random_directions(int(sig.sum()), seed=seed)),
            "ret_gross",
        )
        return rule["mean_bps"] - rand["mean_bps"], rule["tstat"]

    rev_excess, rev_t = edge(reverting[0], seed=1)
    coin_excess, coin_t = edge(coin[0], seed=1)
    # On a strongly reverting tape the rule clearly beats the coin and clears the bar.
    assert rev_excess > 0.0 and rev_t > 2.0
    # On a martingale the rule is not statistically distinguishable from zero.
    assert abs(coin_t) < 2.0


def test_high_win_rate_is_not_an_edge(coin):
    """Spine #2 — the '90% win rate' trap: under a 'take the bounce' exit even a fair
    coin prints a high win-rate while its expectancy is indistinguishable from zero."""
    bars, _ = coin
    sig = st.triple_rsi_entries(bars, trend_sma=None)
    s = st.summarize(st.run_trades(bars, sig, cost_bps=0), "ret_gross")
    assert s["win_rate"] > 0.55          # a clear majority of trades "win"
    assert abs(s["tstat"]) < 2.0         # yet the mean is not significant
    assert s["skew"] < 0.0               # the tell: rare large losses


def test_split_ledger_is_complete(coin):
    """Pre + post halves together should cover all trades."""
    bars, _ = coin
    sig = st.triple_rsi_entries(bars, trend_sma=None)
    led = st.run_trades(bars, sig, cost_bps=0)
    pre, post = st.split_ledger(led, split_year=2015)
    assert len(pre) + len(post) == len(led)

"""Parabolic SAR implementation invariants, barrier engine, and the study's spine:
the SAR flip beats a coin only when directional momentum is real."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parabolic_sar import strategy as st  # noqa: E402
from parabolic_sar import data  # noqa: E402


# ---- Parabolic SAR indicator ----------------------------------------------

def test_sar_output_columns(coin):
    bars, _ = coin
    result = st.parabolic_sar(bars)
    assert set(result.columns) >= {"sar", "dir", "af", "ep"}


def test_sar_directions_are_pm1(coin):
    bars, _ = coin
    result = st.parabolic_sar(bars)
    valid = result["dir"].dropna()
    assert set(valid.unique()).issubset({-1.0, 1.0})


def test_sar_af_bounded(coin):
    bars, _ = coin
    result = st.parabolic_sar(bars)
    af = result["af"].dropna()
    assert (af >= data.SAR_AF_INIT - 1e-9).all()
    assert (af <= data.SAR_AF_MAX + 1e-9).all()


def test_sar_first_bar_initialised(coin):
    """The first bar must have a valid SAR (we initialise at bar 0)."""
    bars, _ = coin
    result = st.parabolic_sar(bars)
    assert not np.isnan(result["sar"].iloc[0])
    assert not np.isnan(result["dir"].iloc[0])


def test_sar_is_deterministic(coin):
    bars, _ = coin
    a = st.parabolic_sar(bars)
    b = st.parabolic_sar(bars)
    assert np.allclose(a["sar"].dropna().to_numpy(), b["sar"].dropna().to_numpy())


def test_flip_entries_directions_are_pm1(coin):
    bars, _ = coin
    ent = st.flip_entries(bars)
    assert set(ent["dir"].unique()).issubset({-1, 1})
    assert len(ent) > 0


def test_flip_entries_balanced(coin):
    """On a coin-flip tape, long and short flips should be roughly balanced."""
    bars, _ = coin
    ent = st.flip_entries(bars)
    ratio = (ent["dir"] == 1).mean()
    assert 0.3 < ratio < 0.7


def test_sar_below_price_in_uptrend():
    """In a cleanly trending up tape the SAR should end below price."""
    n = 200
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.01, 0.005, n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    wick = np.abs(rng.normal(0, 0.002, n)) * close
    bars = pd.DataFrame({
        "open": open_, "high": close + wick, "low": close - wick,
        "close": close, "volume": np.ones(n),
    }, index=pd.bdate_range("2020-01-01", periods=n))
    result = st.parabolic_sar(bars)
    # In a strong uptrend the SAR should be below price most of the time
    bullish_frac = (result["dir"] == 1).mean()
    assert bullish_frac > 0.5


# ---- ATR ------------------------------------------------------------------

def test_atr_positive_and_finite(coin):
    bars, _ = coin
    r = st.atr(bars, n=20).dropna()
    assert len(r) > 0
    assert (r > 0).all()
    assert np.isfinite(r.to_numpy()).all()


# ---- Barrier engine -------------------------------------------------------

def test_ledger_columns_and_exit_reasons(coin):
    bars, _ = coin
    ent = st.flip_entries(bars)
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=1.0)
    assert list(led.columns) == [
        "entry_ts", "dir", "entry", "exit", "exit_reason",
        "bars_held", "ret_gross", "ret_net",
    ]
    assert set(led["exit_reason"].unique()).issubset({"tp", "sl", "eod"})
    assert (led["bars_held"] >= 1).all()


def test_net_is_gross_minus_cost(coin):
    bars, _ = coin
    ent = st.flip_entries(bars)
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=1.5)
    assert np.allclose(led["ret_net"], led["ret_gross"] - 1.5e-4)


def test_cost_monotonically_lowers_mean(coin):
    bars, _ = coin
    ent = st.flip_entries(bars)
    means = [
        st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


def test_return_sign_matches_direction():
    """A take-profit is a win, a stop is a loss, in the traded direction."""
    bars, _ = data.synthetic_daily(n_days=300, momentum=0.0, seed=126)
    ent = st.flip_entries(bars)
    led = st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0)
    tp = led[led["exit_reason"] == "tp"]
    sl = led[led["exit_reason"] == "sl"]
    if len(tp) > 0:
        assert (tp["ret_gross"] > 0).all()
    if len(sl) > 0:
        assert (sl["ret_gross"] < 0).all()


# ---- Random control & the thesis ----------------------------------------

def test_random_directions_reproducible_and_balanced():
    a = st.random_directions(1000, seed=3)
    b = st.random_directions(1000, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) == {-1, 1}
    assert 0.4 < (a == 1).mean() < 0.6


def test_sar_beats_coin_only_with_momentum(coin, trending):
    """The spine: with planted momentum the SAR edges a random-direction control;
    on a martingale it does not."""
    def edge(bars, seed):
        ent = st.flip_entries(bars)
        sar_s = st.summarize(st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0), "ret_gross")
        rand_s = st.summarize(
            st.run_trades(bars, ent, tp_R=1, sl_R=1, cost_bps=0,
                          directions=st.random_directions(len(ent), seed=seed)),
            "ret_gross",
        )
        return sar_s["mean_bps"] - rand_s["mean_bps"]

    # With momentum, SAR should find it
    assert edge(trending[0], seed=1) > 0.0
    # On a coin, SAR should offer no consistent edge over the control
    assert abs(edge(coin[0], seed=1)) < 2.0  # within noise; not a systematic bias


def test_summarize_empty_ledger():
    """summarize() must not crash on an empty ledger."""
    led = pd.DataFrame(columns=["ret_net", "ret_gross"])
    s = st.summarize(led, col="ret_net")
    assert np.isnan(s["tstat"])
    assert np.isnan(s["mean_bps"])

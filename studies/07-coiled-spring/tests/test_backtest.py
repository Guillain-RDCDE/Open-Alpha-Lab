"""The simulator must respect entry/stop/exit mechanics, costs, and arithmetic identities."""

import numpy as np
import pandas as pd

from coiled_spring import backtest, data, signals
from coiled_spring.backtest import ExitRules, run_trades, trade_stats


def _toy_uptrend():
    """A clean frame with one obvious springboard so a trade definitely fires."""
    frames, _ = data.synthetic_universe(seed=0, n_springs=2, n_noise=0)
    return frames


def test_net_is_gross_minus_cost():
    frames = _toy_uptrend()
    f = next(iter(frames.values()))
    led = run_trades(f, cost_bps=25.0)
    assert len(led)
    np.testing.assert_allclose(led["ret_net"], led["ret_gross"] - 25.0 / 1e4, atol=1e-12)


def test_zero_cost_equals_gross():
    frames = _toy_uptrend()
    f = next(iter(frames.values()))
    led = run_trades(f, cost_bps=0.0)
    np.testing.assert_allclose(led["ret_net"], led["ret_gross"], atol=1e-12)


def test_stop_caps_the_loss():
    frames = _toy_uptrend()
    f = next(iter(frames.values()))
    led = run_trades(f, cost_bps=0.0)
    # No gross loss can exceed the initial stop distance by much (gaps aside): the worst
    # trade should be a small negative, never a catastrophic one, because the stop fires.
    assert led["ret_gross"].min() > -0.20


def test_exit_reason_and_bars_are_sane():
    frames = _toy_uptrend()
    f = next(iter(frames.values()))
    rules = ExitRules(max_hold=15)
    led = run_trades(f, rules=rules, cost_bps=0.0)
    assert set(led["exit_reason"]).issubset({"stop", "time"})
    assert (led["bars_held"] >= 0).all()
    assert (led["bars_held"] <= 15).all()


def test_forward_returns_align_with_entry_price():
    frames = _toy_uptrend()
    f = next(iter(frames.values()))
    sig = signals.find_signals(f)
    fr = backtest.forward_returns(f, sig, horizons=(5,))
    pos = {d: i for i, d in enumerate(f.index)}
    close = f["Close"].to_numpy(float)
    for _, row in fr.dropna().iterrows():
        i0 = pos[row["entry_date"]]
        want = close[i0 + 5] / row["entry_price"] - 1.0
        assert abs(row["fwd_5"] - want) < 1e-9


def test_trade_stats_on_empty():
    assert trade_stats(pd.DataFrame()) == {"n_trades": 0}


def test_synthetic_strategy_is_profitable():
    """On the planted universe the rule SHOULD pay — validates the machinery end to end."""
    frames, _ = data.synthetic_universe(seed=0)
    led = backtest.run_universe(frames, cost_bps=15.0)
    st = trade_stats(led)
    assert st["n_trades"] > 20
    assert st["mean_net"] > 0.0
    assert st["win_rate"] > 0.6

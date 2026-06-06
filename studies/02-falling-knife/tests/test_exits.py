"""Exit resolution: target, stop, time, and the conservative tie-break."""

import pandas as pd

from falling_knife.exits import ExitRule, resolve_trade


def _frame(rows):
    idx = pd.bdate_range("2020-01-01", periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def test_time_exit():
    # Flat market, no target/stop -> exit at the close of the max_hold bar.
    ohlc = _frame([[100, 101, 99, 100]] * 6)
    tr = resolve_trade(ohlc, entry_pos=0, entry_price=100.0, rule=ExitRule(max_hold=3))
    assert tr.reason == "time" and tr.holding_days == 3


def test_target_hit():
    ohlc = _frame([
        [100, 100, 100, 100],   # entry bar
        [100, 103, 99, 102],    # high 103 reaches +2% target
        [100, 100, 100, 100],
    ])
    tr = resolve_trade(ohlc, 0, 100.0, ExitRule(max_hold=5, target=0.02, stop=0.10))
    assert tr.reason == "target"
    assert abs(tr.exit_price - 102.0) < 1e-9  # filled at the target, not the high


def test_stop_hit():
    ohlc = _frame([
        [100, 100, 100, 100],
        [100, 101, 94, 96],     # low 94 breaches -5% stop
        [100, 100, 100, 100],
    ])
    tr = resolve_trade(ohlc, 0, 100.0, ExitRule(max_hold=5, target=0.10, stop=0.05))
    assert tr.reason == "stop"
    assert abs(tr.exit_price - 95.0) < 1e-9


def test_stop_wins_when_both_touched():
    # A single bar whose range spans BOTH the target and the stop -> stop wins.
    ohlc = _frame([
        [100, 100, 100, 100],
        [100, 110, 90, 100],    # touches +5% target AND -5% stop
    ])
    tr = resolve_trade(ohlc, 0, 100.0, ExitRule(max_hold=5, target=0.05, stop=0.05))
    assert tr.reason == "stop"


def test_no_bar_after_entry_returns_none():
    ohlc = _frame([[100, 101, 99, 100]])
    assert resolve_trade(ohlc, 0, 100.0, ExitRule(max_hold=3)) is None

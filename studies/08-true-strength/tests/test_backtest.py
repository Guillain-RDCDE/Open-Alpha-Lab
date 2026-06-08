"""The engine must be causal, charge turnover, and segment trades correctly."""

import numpy as np
import pandas as pd

from true_strength import backtest


def _close():
    rng = np.random.default_rng(1)
    return pd.Series(np.exp(np.cumsum(0.01 * rng.standard_normal(400))) + 5.0,
                     index=pd.bdate_range("2020-01-01", periods=400))


def test_position_is_lagged_no_lookahead():
    c = _close()
    pos = pd.Series(1.0, index=c.index)
    led = backtest.run_strategy(c, pos, cost_bps=0.0)
    # first bar's position is shifted in as 0 -> no return earned on day 0.
    assert led["pos"].iloc[0] == 0.0
    assert led["gross"].iloc[0] == 0.0


def test_turnover_cost_charged_on_changes():
    c = _close()
    pos = pd.Series(0.0, index=c.index)
    pos.iloc[100:200] = 1.0                     # one flat->long->flat round trip
    led = backtest.run_strategy(c, pos, cost_bps=10.0)
    # two position changes (in and out), each pays 10 bps of turnover.
    assert abs(led["turnover"].sum() - 2.0) < 1e-9
    assert led["net"].sum() < led["gross"].sum()


def test_trades_segment_holding_spells():
    idx = pd.bdate_range("2020-01-01", periods=10)
    led = pd.DataFrame({
        "pos": [0, 1, 1, 0, 0, -1, -1, -1, 0, 1],
        "net": [0, 0.01, 0.02, 0, 0, -0.01, 0.0, 0.03, 0, 0.05],
    }, index=idx).astype(float)
    tr = backtest.trades_from_position(led)
    assert len(tr) == 3                         # long, short, long
    assert list(tr["side"]) == [1.0, -1.0, 1.0]
    assert tr["bars"].tolist() == [2, 3, 1]


def test_stats_keys_present():
    c = _close()
    led = backtest.run_strategy(c, backtest.tsi_position(c), cost_bps=10.0)
    st = backtest.strategy_stats(led)
    for k in ["n_trades", "cagr", "sharpe", "exposure", "max_drawdown", "profit_factor", "win_rate"]:
        assert k in st
    assert -1.0 <= st["max_drawdown"] <= 0.0
    assert 0.0 <= st["exposure"] <= 1.0

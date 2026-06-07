"""Exit resolution, the event-driven backtest, and cost monotonicity."""

import numpy as np
import pandas as pd

from fear_gauge import backtest, exits, triggers


def test_stop_fills_before_target_on_a_both_touch_bar():
    # Entry at 100; next bar low 90 (hits -5% stop) and high 106 (hits +3% target).
    idx = pd.bdate_range("2021-01-01", periods=2)
    market = pd.DataFrame(
        {"Open": [100, 100], "High": [100, 106], "Low": [100, 90], "Close": [100, 100.0]},
        index=idx,
    )
    rule = exits.ExitRule(max_hold=1, target=0.03, stop=0.05)
    tr = exits.resolve_trade(market, 0, 100.0, rule)
    assert tr.reason == "stop"  # conservative: the stop wins a both-touch bar


def test_time_exit_when_no_levels_hit():
    idx = pd.bdate_range("2021-01-01", periods=4)
    close = pd.Series([100, 101, 102, 103.0], index=idx)
    market = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close})
    tr = exits.resolve_trade(market, 0, 100.0, exits.ExitRule(max_hold=2))
    assert tr.reason == "time" and tr.holding_days == 2


def test_backtest_runs_and_reports(synth_market, synth_gauge):
    sig = triggers.first_crossings(triggers.level(synth_gauge, 30), cooldown=21)
    res = backtest.run(synth_market, sig, exits.ExitRule(max_hold=10), backtest.CostModel())
    assert res.stats["n_trades"] >= 1
    assert res.equity.iloc[-1] > 0
    # exposure is a fraction of days in-position
    assert 0.0 <= res.stats["exposure"] <= 1.0


def test_higher_costs_never_improve_return(synth_market, synth_gauge):
    sig = triggers.first_crossings(triggers.level(synth_gauge, 30), cooldown=21)
    rule = exits.ExitRule(max_hold=10)
    cheap = backtest.run(synth_market, sig, rule, backtest.CostModel(panic_slippage_bps=0))
    dear = backtest.run(synth_market, sig, rule, backtest.CostModel(panic_slippage_bps=50))
    assert dear.stats["total_return"] <= cheap.stats["total_return"] + 1e-12


def test_family_scan_sorted_by_sharpe(synth_market, synth_gauge):
    fam = {"V1_30": triggers.level(synth_gauge, 30), "V2": triggers.spike(synth_gauge)}
    scan = backtest.family_scan(synth_market, fam, exits.default_grid())
    s = scan["sharpe"].dropna().to_numpy()
    assert (np.diff(s) <= 1e-9).all()  # non-increasing

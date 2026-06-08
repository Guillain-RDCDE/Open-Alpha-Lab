"""Robustness: decay-by-year, the wait rule, neutrality, bootstrap, capacity."""

import numpy as np

from pairs_trading import backtest, data, robustness


def test_decay_by_year_columns(panel):
    decay = robustness.decay_by_year(panel, top_n=6)
    assert {"monthly_net", "sharpe", "n_days"} <= set(decay.columns)
    assert len(decay) >= 2                         # spans multiple calendar years


def test_wait_rule_runs_all_lags(panel):
    fx = robustness.wait_rule_effect(panel, top_n=6, waits=(1, 2, 3))
    assert list(fx.index) == [1, 2, 3]
    assert (fx["n_trades"] > 0).all()


def test_market_neutrality_keys(panel):
    res = backtest.run(panel, top_n=6)
    out = robustness.market_neutrality(res.daily, data.market_return(panel))
    assert {"alpha_daily_bps", "alpha_ann_pct", "beta", "r_squared"} <= set(out)


def test_bootstrap_sharpe_ci_brackets_point(panel):
    res = backtest.run(panel, top_n=6)
    boot = robustness.bootstrap_sharpe(res.daily, n_boot=500)
    assert boot["ci_low"] <= boot["sharpe"] <= boot["ci_high"]


def test_selection_recall_full_on_truth(true_pairs):
    # passing the truth itself back in must score 1.0
    from pairs_trading.pairs import Pair
    fake = [Pair(p.a, p.b, ssd=0.0, sigma=1.0) for p in true_pairs]
    assert robustness.selection_recall(fake, true_pairs) == 1.0


def test_capacity_scales_with_edge(frames):
    res = backtest.run(data.close_panel(frames), top_n=6)
    dvol = data.dollar_volume_panel(frames)
    small = robustness.capacity(dvol, res.trades, edge_bps=20)
    big = robustness.capacity(dvol, res.trades, edge_bps=80)
    assert big["capacity_usd_per_leg"] > small["capacity_usd_per_leg"] > 0

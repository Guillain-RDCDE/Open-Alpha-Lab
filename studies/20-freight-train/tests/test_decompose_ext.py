"""The trend is HAC-significant and carries a basket alpha on the trend tape, a statistical zero on the
null; the crisis-convexity complement builds and splits months sensibly."""

import numpy as np

from trend_follow import strategy, decompose, extension


def test_hac_significant_on_trend(trend_panel):
    panel, _ = trend_panel
    t = decompose.mean_tstat_hac(strategy.tsmom_returns(panel, cost_bps=2.0))
    assert t["t_stat"] > 2.5 and t["mean_ann_pct"] > 0


def test_hac_zero_on_null(null_panel):
    panel, _ = null_panel
    t = decompose.mean_tstat_hac(strategy.tsmom_returns(panel, cost_bps=2.0))
    assert abs(t["t_stat"]) < 2.0


def test_basket_alpha_positive_on_trend(trend_panel):
    panel, _ = trend_panel
    a = decompose.basket_alpha(panel, cost_bps=2.0)
    assert a["alpha_ann_pct"] > 0 and a["alpha_t"] > 2.0
    assert a["beta"] < 0.5                            # not just disguised long beta


def test_subsample_and_bootstrap_run(trend_panel):
    panel, _ = trend_panel
    sub = decompose.subsample_sharpe(panel, cost_bps=2.0, n_chunks=3)
    assert len(sub) == 3 and "sharpe" in sub.columns
    bs = decompose.sharpe_bootstrap(panel, n_boot=300, cost_bps=2.0)
    assert bs["ci_low"] < bs["sharpe_gain"] < bs["ci_high"]


def test_crisis_convexity_builds(trend_panel):
    panel, _ = trend_panel
    cc = extension.crisis_convexity(panel, cost_bps=2.0)
    assert cc["n_crisis_months"] > 0
    # the basket's worst months really are worse than its calm months (a sanity check on the split)
    assert cc["basket_crisis_mean_pct"] < cc["basket_calm_mean_pct"]
    dm = extension.down_market_capture(panel, cost_bps=2.0)
    assert dm["basket_in_down_months_pct"] < 0

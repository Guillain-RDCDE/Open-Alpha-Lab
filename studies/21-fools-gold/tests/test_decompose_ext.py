"""The golden-minus-death spread is HAC-significant on the trend tape and a zero on the null; the timing
book runs sub-1 beta; and the parameter grid generalises on a trend but not on a random walk."""

from fools_gold import decompose, extension


def test_spread_tstat_significant_on_trend(trend_close):
    sp = decompose.spread_tstat(trend_close)
    assert sp["t_stat"] > 2.0 and sp["spread_ann_pct"] > 0


def test_spread_tstat_zero_on_null(null_close):
    sp = decompose.spread_tstat(null_close)
    assert abs(sp["t_stat"]) < 2.0


def test_timing_is_sub_one_beta(trend_close):
    vb = decompose.vs_buy_hold(trend_close, cost_bps=2.0)
    assert 0.0 < vb["beta"] < 1.0       # out of the market a lot -> less exposure


def test_risk_matched_builds(trend_close):
    rm = decompose.risk_matched(trend_close, cost_bps=2.0)
    assert 0.0 < rm["avg_exposure"] < 1.0
    assert "sharpe_edge_over_blend" in rm


def test_param_grid_trend_beats_null(trend_close, null_close):
    """A real trend gives a higher average crossover payoff than a (single, possibly spuriously-
    trending) random walk. We test the robust *direction* — a single null path can drift by chance,
    which is itself the data-mining lesson, not a clean zero."""
    pg_t = extension.param_grid(trend_close, cost_bps=2.0)
    pg_n = extension.param_grid(null_close, cost_bps=2.0)
    assert pg_t["frac_beat_buy_hold"] > 0.5          # a real trend: most pairs help
    assert pg_t["mean_gain"] > pg_n["mean_gain"]     # and it out-earns the driftless tape on average

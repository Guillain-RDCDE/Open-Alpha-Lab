"""Inverting the noisy sample covariance does not decisively beat the naive book, the in-sample tangency
Sharpe overstates the causal one (the sand castle), and shrinkage converges the optimizer to the naive."""

from sand_castle import decompose, extension, strategy


def test_optimizer_does_not_beat_naive_much(rev_panel, rev_market):
    ov = decompose.optimizer_vs_naive(rev_panel, rev_market, cost_bps=5.0)
    # the C^{-1} step buys essentially nothing net -- it never decisively beats the naive signal weighting
    assert ov["opt_minus_naive_net"] < 0.5


def test_gross_beats_net(rev_panel, rev_market):
    gv = decompose.gross_vs_net(rev_panel, rev_market, cost_bps=5.0)
    assert gv["gross_sharpe"] > gv["net_sharpe"]   # turnover cost always erodes the gross edge
    assert gv["cost_gap"] >= 0.0


def test_weight_instability_reports(rev_panel, rev_market):
    wi = decompose.weight_instability(rev_panel, rev_market)
    assert wi["condition_number"] > 1.0
    assert wi["max_abs_weight_optimized"] > 0.0


def test_shrink_converges_to_naive(rev_panel, rev_market):
    sw = extension.shrink_sweep(rev_panel, rev_market, cost_bps=5.0, shrinks=(0.0, 1.0))
    # at full shrink the optimized book ~= the naive book (diagonal C -> C^{-1}E proportional to E)
    naive = sw["naive_net_sharpe"].iloc[0]
    assert abs(sw.loc[1.0, "optimized_net_sharpe"] - naive) < 0.4

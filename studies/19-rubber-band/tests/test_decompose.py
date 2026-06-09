"""The bounce is HAC-significant on the reversal tape and a statistical zero on the null; the
break-even cost is finite and positive when there's an edge, zero when there isn't."""

from rubber_band import decompose, strategy, extension


def test_hac_tstat_significant_on_reversal(reversal_ohlc):
    t = decompose.mean_tstat_hac(strategy.timing_returns(reversal_ohlc, cost_bps=0.0))
    assert t["t_stat"] > 3.0
    assert t["mean_ann_pct"] > 0.0


def test_hac_tstat_zero_on_null(null_ohlc):
    t = decompose.mean_tstat_hac(strategy.timing_returns(null_ohlc, cost_bps=0.0))
    assert abs(t["t_stat"]) < 2.0


def test_breakeven_positive_with_edge(reversal_ohlc):
    be = decompose.breakeven_cost({"A": reversal_ohlc})
    assert be["breakeven_bps"] > 0.5
    assert be["gross_sharpe"] > 0.0


def test_breakeven_zero_on_null(null_ohlc):
    be = decompose.breakeven_cost({"A": null_ohlc})
    assert be["breakeven_bps"] == 0.0      # no gross edge -> nothing to give back to costs


def test_bootstrap_ci_clears_zero_on_reversal(reversal_ohlc):
    bs = decompose.sharpe_bootstrap(strategy.timing_returns(reversal_ohlc, cost_bps=0.0), n_boot=500)
    assert bs["ci_low"] > 0.0


def test_extension_gross_beats_net(reversal_ohlc):
    basket = {"A": reversal_ohlc}
    r = extension.basket_net_at_spread(basket, spread_bps=3.0)
    assert r["gross_sharpe"] > r["net_sharpe"]
    pn = extension.per_name_breakeven(basket)
    assert "gross_sharpe" in pn.columns and "breakeven_bps" in pn.columns

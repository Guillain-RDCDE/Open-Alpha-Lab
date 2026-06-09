"""Beat-7 worked complement: the edge splits into a no-shorting defensive slice and a short-the-wild
slice, the dollar-neutral book has a finite borrow break-even on the anomaly tape, and the null leaves
nothing to split."""

from dull_roar import extension


def test_shorting_decomposition_splits_the_edge(anomaly_panel, anomaly_market):
    sd = extension.shorting_decomposition(anomaly_panel, market=anomaly_market)
    # full alpha is the sum of the defensive (no-short) and short-the-wild slices, by construction
    assert abs((sd["alpha_defensive"] + sd["alpha_short"]) - sd["alpha_full"]) < 1e-6
    assert sd["alpha_full"] > 2.0
    assert sd["defensive_sharpe_gain"] > 0.0


def test_borrow_breakeven_finite_on_anomaly(anomaly_panel, anomaly_market):
    be = extension.borrow_breakeven(anomaly_panel, market=anomaly_market)
    # the naive dollar-neutral book pays at zero borrow but only up to a finite fee (the short leg bites)
    assert 0.0 <= be["breakeven_bps"] <= 5000.0
    assert be["crossed"] is True


def test_null_leaves_no_alpha_to_harvest(null_panel, null_market):
    sd = extension.shorting_decomposition(null_panel, market=null_market)
    assert abs(sd["alpha_full"]) < 3.0
    assert abs(sd["alpha_defensive_t"]) < 2.0

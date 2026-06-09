"""The sort separates calm from wild out-of-sample, recovers the flat SML on the anomaly tape, and
finds no gradient on the null."""

import numpy as np

from dull_roar import sort


def test_realized_vol_shape(anomaly_panel):
    rv = sort.realized_vol(anomaly_panel, window=126)
    assert rv.shape == anomaly_panel.shape
    assert rv.iloc[:125].isna().all().all()          # warm-up is NaN until the window fills
    assert rv.iloc[126:].notna().any().any()


def test_low_leg_is_actually_calmer(anomaly_panel):
    """Out-of-sample check that the sort works: the realized vol of the low-vol book is below the
    high-vol book's (sorting on past vol does separate future vol)."""
    book = sort.quantile_portfolios(anomaly_panel)
    assert book.low.std() < book.high.std()
    assert book.rebalances > 100


def test_sml_slope_recovers_anomaly(anomaly_panel):
    sml = sort.security_market_line(anomaly_panel)
    assert sml["sharpe_vol_slope"] < -0.3           # clear negative gradient: calm earns more per risk
    assert sml["low_bucket_sharpe"] > sml["high_bucket_sharpe"]
    assert sml["anomaly_present"]


def test_sml_slope_flat_on_null(null_panel):
    sml = sort.security_market_line(null_panel)
    # null: no real gradient. allow a small in-sample artefact but it must be far milder than the anomaly
    assert abs(sml["sharpe_vol_slope"]) < 0.3
    assert abs(sml["low_bucket_sharpe"] - sml["high_bucket_sharpe"]) < 0.2


def test_no_lookahead_weights_piecewise_constant(anomaly_panel):
    """Weights may only change at rebalances (every 21 days), never intra-period — a basic guard that
    the book is held as described and not re-fit each day."""
    book = sort.quantile_portfolios(anomaly_panel, rebal=21)
    changes = (book.w_low.diff().abs().sum(axis=1) > 1e-12).sum()
    assert changes <= book.rebalances + 1          # at most one change per rebalance

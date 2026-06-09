"""The momentum engine recovers the baked winners>losers spread (and finds nothing on the null), and
the WML factor makes money with momentum and adds nothing without it."""

import numpy as np

from stampede import data, momentum, strategy


def test_deterministic_and_momentum(momentum_tape):
    panel, _, truth = momentum_tape
    panel2, _, _ = data.synthetic_panel(mom_strength=0.0015, seed=24)
    assert np.allclose(panel.to_numpy(), panel2.to_numpy())
    assert truth.has_momentum


def test_score_shape(momentum_panel):
    sc = momentum.momentum_score(momentum_panel, lookback=252, skip=21)
    assert sc.shape == momentum_panel.shape
    assert sc.iloc[:200].isna().all().all()       # warm-up until the 12-month window fills
    assert sc.iloc[252:].notna().any().any()


def test_spread_recovers_momentum(momentum_panel):
    sp = momentum.momentum_spread(momentum_panel)
    assert sp["wml_ann_pct"] > 5.0           # winners clearly out-earn losers
    assert sp["winners_ann_pct"] > sp["losers_ann_pct"]
    assert sp["momentum_present"]


def test_spread_flat_on_null(null_panel):
    sp = momentum.momentum_spread(null_panel)
    assert abs(sp["wml_ann_pct"]) < 8.0      # no persistent relative drift -> ~flat


def test_wml_beats_market_on_momentum(momentum_panel):
    cmp = strategy.compare(momentum_panel, cost_bps=5.0)
    assert cmp["wml"]["sharpe"] > 0.8
    assert cmp["turnover_ann"] < 15.0        # monthly 12-1 sort -> modest turnover


def test_wml_no_edge_on_null(null_panel):
    cmp = strategy.compare(null_panel, cost_bps=5.0)
    assert abs(cmp["wml"]["sharpe"]) < 0.5

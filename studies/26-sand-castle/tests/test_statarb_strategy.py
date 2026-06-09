"""The reversion signal is real on the reversion tape and absent on the null; the weights are
dollar-neutral and gross-normalised; the book makes money gross with reversion and nothing without it."""

import numpy as np

from sand_castle import data, statarb, strategy


def test_deterministic_and_reversion(rev):
    panel, _, truth = rev
    panel2, _, _ = data.synthetic_panel(n_stocks=40, n_bars=1512, revert=0.20, seed=26)
    assert np.allclose(panel.to_numpy(), panel2.to_numpy())
    assert truth.has_reversion


def test_signal_quality_recovers_reversion(rev_panel, rev_market):
    sq = statarb.signal_quality(rev_panel, rev_market)
    assert sq["mean_ic"] > 0.0 and sq["ic_t"] > 3.0
    assert sq["reversion_present"]


def test_signal_quality_flat_on_null(null_panel, null_market):
    sq = statarb.signal_quality(null_panel, null_market)
    assert abs(sq["ic_t"]) < 3.0


def test_weights_dollar_neutral_gross_one():
    E = np.array([0.4, -0.1, 0.3, -0.6])
    C = np.eye(4) + 0.1
    for w in (statarb.optimal_weights(E, C), statarb.naive_weights(E)):
        assert abs(w.sum()) < 1e-9              # dollar-neutral
        assert abs(np.abs(w).sum() - 1.0) < 1e-9  # gross-normalised


def test_gross_edge_on_reversion(rev_panel, rev_market):
    g = strategy.summary(strategy.statarb_returns(rev_panel, rev_market, optimized=False, cost_bps=0.0))
    assert g["sharpe"] > 0.5                     # gross of cost, the reversion pays


def test_no_gross_edge_on_null(null_panel, null_market):
    g = strategy.summary(strategy.statarb_returns(null_panel, null_market, optimized=False, cost_bps=0.0))
    assert abs(g["sharpe"]) < 0.6

"""The falsification battery must reproduce the verdict's qualitative facts on a small sample."""

import numpy as np

from markov_mint import robustness
from markov_mint.markov import MarkovMintSystem

# A lean system keeps the test suite fast without changing any qualitative result.
LEAN = MarkovMintSystem(n_sims=1200)


def test_no_directional_edge_on_null(efficient):
    """On a fair market the machine captures no positive edge; the oracle has nothing to find."""
    df = robustness.analyze_markets(efficient, LEAN, seed=0)
    head = robustness.headline_test(df)
    assert head["oracle_edge_pp"] == 0.0          # no exploitable side exists
    assert head["machine_t"] < 2.0                # not a positive signal


def test_raw_edge_is_shrinking_noise():
    """The raw Monte-Carlo edge is zero-mean and its spread collapses as history grows."""
    hist = robustness.edge_vs_history(hist_lens=(40, 250), n_markets=300, seed=0, system=LEAN)
    assert abs(hist.loc[40, "mean_raw_edge_pp"]) < 2.5
    assert hist.loc[250, "std_raw_edge_pp"] < hist.loc[40, "std_raw_edge_pp"]


def test_chain_adds_noise_not_signal(efficient):
    """Deleting the chain shrinks the bet count and the system edge is mostly Monte-Carlo noise."""
    inert = robustness.inertness(efficient, LEAN, seed=0)
    assert inert["active_frac_full"] > inert["active_frac_ablated"]   # chain manufactures trades
    assert 0.0 < inert["edge_corr_full_vs_ablated"] < 0.9            # mostly noise, not the price


def test_calibration_ceiling_forces_short_favorites(efficient):
    df = robustness.analyze_markets(efficient, LEAN, seed=0)
    ceil = robustness.calibration_ceiling_effect(df)
    assert ceil["n_above_ceiling"] > 0
    assert ceil["frac_above_that_buy_no"] > 0.5     # rich favorites are reflexively shorted


def test_null_is_a_coin_flip(efficient):
    """No positive edge on the null: the win rate is a coin flip, never 'every trade'.

    (Per-trade returns on the null are zero-mean but heavily skewed — single-sample means and
    bankrolls are high-variance on 400 markets; the headline run in docs/results.md uses
    2,000. Here we assert the robust facts: a coin-flip win rate and no positive Sharpe.)
    """
    df = robustness.analyze_markets(efficient, LEAN, seed=0)
    pnl = robustness.pnl_sim(df, spread=0.02, seed=0)
    assert 0.4 < pnl["win_rate"] < 0.6               # a coin flip, never "every trade"
    assert pnl["per_trade_sharpe"] < 0.1             # not a positive-edge book


def test_cost_sweep_monotone(efficient):
    df = robustness.analyze_markets(efficient, LEAN, seed=0)
    sweep = robustness.cost_sweep(df, spreads=(0.0, 0.02, 0.08), seed=0)
    net = sweep["mean_ret"].to_numpy()
    assert net[0] >= net[1] >= net[2]


def test_planted_edge_is_found_and_costs_gate_it(biased):
    """Machinery sanity on the planted wedge: the edge is real, found, and toll-gated.

    With the Thaler-Ziemba wedge planted in the correct direction (longshots rich, favorites
    cheap), the calibration table points the machine the right way, so its gross directional
    edge must be positive. The cost-blind ('forced') oracle is the gross ceiling; the honest
    net reference is the cost-aware oracle, which only trades where the wedge clears the
    entry toll — it must net at least as much as the forced one, and keep a positive mean.
    """
    rec = robustness.recover_planted(biased, LEAN, spread=0.02, seed=1)
    assert rec["oracle_forced_edge_pp_gross"] > 0.0     # the planted edge is real and findable
    assert rec["machine_edge_pp_gross"] > 0.0           # the machine now points the right way
    # The cost-aware oracle passes on toll-dominated markets...
    assert rec["oracle_aware_frac"] < 1.0
    # ...and that selectivity is exactly what survives the spread.
    assert rec["oracle_aware_mean_ret_net"] > rec["oracle_forced_mean_ret_net"]
    assert rec["oracle_aware_mean_ret_net"] > 0.0

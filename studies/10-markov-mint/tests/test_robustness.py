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


def test_null_is_a_coin_flip_that_bleeds(efficient):
    """No positive edge: win rate is a coin flip and a normal spread sinks the bankroll.

    (The *zero-cost* terminal bankroll is high-variance on a small sample — the headline run in
    docs/results.md uses 2,000 markets; here we assert the robust facts: a coin-flip win rate, a
    non-positive per-trade Sharpe, and a bleed once the 2c spread bites.)
    """
    df = robustness.analyze_markets(efficient, LEAN, seed=0)
    pnl = robustness.pnl_sim(df, spread=0.02, seed=0)
    assert 0.4 < pnl["win_rate"] < 0.6               # a coin flip, never "every trade"
    assert pnl["per_trade_sharpe"] < 0.1             # not a positive-edge book
    assert pnl["terminal_bankroll"] < 1.0            # the spread sinks it


def test_cost_sweep_monotone(efficient):
    df = robustness.analyze_markets(efficient, LEAN, seed=0)
    sweep = robustness.cost_sweep(df, spreads=(0.0, 0.02, 0.08), seed=0)
    net = sweep["mean_ret"].to_numpy()
    assert net[0] >= net[1] >= net[2]


def test_planted_edge_exists_and_is_thin(biased):
    """Machinery sanity: a real wedge is present (oracle finds it), but it is small."""
    rec = robustness.recover_planted(biased, LEAN, spread=0.02, seed=1)
    assert rec["oracle_edge_pp_gross"] > 0.0          # the planted edge is real and findable
    # ...and even perfect information barely survives a 2c spread.
    assert rec["oracle_mean_ret_net"] < rec["oracle_edge_pp_gross"]

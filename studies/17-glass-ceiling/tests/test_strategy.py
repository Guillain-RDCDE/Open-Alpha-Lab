"""The load-bearing claims: a symmetric ±1R bracket is a coin flip on the null tape, the win rate
tracks the baked-in continuation drift, and the break-even arithmetic (0.5 + cost_R/2) holds — so the
spread, paid twice, is what kills a 1:1 strategy whose true win rate is ~50%."""

import numpy as np

from glass_ceiling import strategy


def test_null_tape_is_a_coin_flip(null_tape):
    """No continuation -> win rate's Wilson interval straddles 0.5, gross expectancy ~ 0."""
    bars, _ = null_tape
    trades = strategy.run(bars)
    s = strategy.summary(trades)
    assert s["n_trades"] > 300                      # enough trades to mean something
    lo, hi = strategy.win_rate_ci(trades)
    assert lo <= 0.5 <= hi                           # indistinguishable from a coin flip
    assert abs(s["expectancy_R_gross"]) < 0.08


def test_continuation_lifts_win_rate(cont_tape):
    """Genuine follow-through pushes the long bracket above 0.5 and gross expectancy positive."""
    bars, _ = cont_tape
    trades = strategy.run(bars)
    s = strategy.summary(trades)
    assert s["win_rate"] > 0.5
    assert s["expectancy_R_gross"] > 0.0


def test_exhaustion_drops_win_rate(fade_tape):
    """Fading breakouts pull the win rate below 0.5 — the buy-the-high trap is detectable."""
    bars, _ = fade_tape
    trades = strategy.run(bars)
    s = strategy.summary(trades)
    assert s["win_rate"] < 0.5


def test_breakeven_arithmetic_holds(null_tape):
    """break-even win rate == 0.5 + cost_R/2, and net == gross − cost_R, by construction."""
    bars, _ = null_tape
    trades = strategy.run(bars)
    s = strategy.summary(trades, roundtrip_bps=4.0)
    assert np.isclose(s["breakeven_win_rate"], 0.5 + s["cost_R"] / 2.0)
    assert np.isclose(s["expectancy_R_net"], s["expectancy_R_gross"] - s["cost_R"])
    assert s["cost_R"] > 0.0                          # a real cost in R, since R is a ~1% stop


def test_costs_make_the_coin_flip_negative(null_tape):
    """The whole verdict in one assert: a ~50% strategy is net-negative once it pays a spread."""
    bars, _ = null_tape
    trades = strategy.run(bars)
    sweep = strategy.cost_sweep(trades, roundtrip_bps=(0, 2, 5, 10))
    assert sweep.loc[0, "expectancy_R_net"] >= sweep.loc[10, "expectancy_R_net"]
    assert sweep.loc[10, "expectancy_R_net"] < 0.0


def test_one_position_at_a_time(null_tape):
    """Trades never overlap: each entry comes strictly after the previous exit."""
    bars, _ = null_tape
    trades = strategy.run(bars)
    gaps = trades["entry_idx"].to_numpy()[1:] - trades["exit_idx"].to_numpy()[:-1]
    assert (gaps >= 1).all()

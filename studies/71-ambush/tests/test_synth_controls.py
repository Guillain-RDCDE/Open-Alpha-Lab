"""The positive control and the null — the harness must pass both or prove nothing."""

import pytest

from ambush import strategy, synth
from quantlab.analytics import mean_tstat_hac


def test_planted_confluence_premium_lights_up():
    spy, vix = synth.synthetic_tape(plant_bps_per_signal=8.0, seed=0)
    lift = strategy.lift_table(spy, vix)
    # broad lift: zero-signal days flat-ish, K>=3 days carry the plant
    assert lift.loc[3, "next_bps"] > lift.loc[0, "next_bps"] + 10
    t = mean_tstat_hac(strategy.armed_stream(spy, vix, k=3))
    assert t["tstat"] > 2.0
    led = strategy.book(spy, vix, synth.flat_rf(spy.index), k=3)
    assert strategy.summary(led["net_excess"])["sharpe"] > 0.5  # banked through the overlay


def test_random_walk_stays_dark():
    spy, vix = synth.synthetic_tape(plant_bps_per_signal=0.0, seed=1)
    t = mean_tstat_hac(strategy.armed_stream(spy, vix, k=3))
    assert abs(t["tstat"]) < 2.0
    led = strategy.book(spy, vix, synth.flat_rf(spy.index), k=3)
    assert abs(strategy.summary(led["net_excess"])["sharpe"]) < 0.6


def test_tape_is_deterministic_per_seed():
    a, va = synth.synthetic_tape(n_days=300, seed=7)
    b, vb = synth.synthetic_tape(n_days=300, seed=7)
    assert a.equals(b) and va.equals(vb)
    c, _ = synth.synthetic_tape(n_days=300, seed=8)
    assert not a.equals(c)

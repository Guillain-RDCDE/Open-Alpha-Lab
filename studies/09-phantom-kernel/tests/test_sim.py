"""The simulator must produce the worlds it claims: an exponential kernel in A, a heavy tail
in B, and a clean exogenous order stream."""

import numpy as np

from phantom_kernel import sim


def test_reach_is_nonnegative():
    rng = np.random.default_rng(0)
    for w in (sim.WORLD_A, sim.WORLD_B):
        r = sim.sample_reach(w, 10_000, rng)
        assert (r >= 0).all()


def test_world_a_reach_is_exponential(deltas):
    """Exponential reach => survival e^{-k delta}; fitting log-counts recovers k."""
    counts = sim.fill_counts(sim.WORLD_A, deltas, n_orders=200_000, seed=0)
    # log(count) should be ~ linear in delta with slope -k over the populated buckets.
    m = counts > 50
    slope = np.polyfit(deltas[m], np.log(counts[m]), 1)[0]
    assert abs(-slope - sim.WORLD_A.reach_k) < 0.05


def test_world_b_tail_is_heavier_than_exponential(deltas):
    """Pareto reach must keep delivering fills far in the tail where an exponential is dead."""
    ca = sim.fill_counts(sim.WORLD_A, deltas, n_orders=200_000, seed=0)
    cb = sim.fill_counts(sim.WORLD_B, deltas, n_orders=200_000, seed=0)
    far = deltas >= 20.0
    assert cb[far].sum() > 5 * ca[far].sum()


def test_fill_counts_monotone_decreasing(deltas):
    counts = sim.fill_counts(sim.WORLD_B, deltas, n_orders=100_000, seed=1)
    assert np.all(np.diff(counts) <= 0)


def test_flow_shapes_and_textbook_world_is_clean():
    flow = sim.simulate_flow(sim.WORLD_A, n_steps=3_000, dt=0.01, seed=0)
    assert flow.s.shape == (3_001,)
    assert flow.buy_arr.shape == (3_000,)
    # World A has no information and no jumps: no informed orders at all.
    assert flow.buy_informed.sum() == 0 and flow.sell_informed.sum() == 0
    assert np.isfinite(flow.s).all()


def test_friction_world_has_informed_flow():
    flow = sim.simulate_flow(sim.WORLD_B, n_steps=5_000, dt=0.01, seed=0)
    assert flow.buy_informed.sum() + flow.sell_informed.sum() > 0

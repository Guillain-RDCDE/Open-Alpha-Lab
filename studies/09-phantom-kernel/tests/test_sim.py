"""The simulator must produce the worlds it claims: an exponential kernel in A, a heavy tail
in B, and a clean exogenous order stream."""

import numpy as np

from phantom_kernel import sim


def test_reach_is_nonnegative():
    rng = np.random.default_rng(0)
    for w in (sim.WORLD_A, sim.WORLD_B, sim.WORLD_B_STRESS):
        r = sim.sample_reach(w, 10_000, rng)
        assert (r >= 0).all()


def test_world_b_mean_reach_matches_world_a():
    """The two worlds (and the stress variant) trade comparable volume: only the tail shape
    differs, so the kernel test isn't confounded by a different mean reach."""
    for w in (sim.WORLD_B, sim.WORLD_B_STRESS):
        assert abs(w.mean_reach() - sim.WORLD_A.mean_reach()) < 0.02


def test_world_b_tail_is_inside_the_measured_band():
    """The central friction world must not be heavier than every tail the study measured on
    real books (survival exponents ~1.4-3.2, docs/results_real.md); the heavier case is the
    explicitly labelled stress world."""
    assert 1.4 <= sim.WORLD_B.pareto_alpha <= 3.2
    assert sim.WORLD_B_STRESS.pareto_alpha < 1.4
    assert "stress" in sim.WORLD_B_STRESS.name.lower()


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

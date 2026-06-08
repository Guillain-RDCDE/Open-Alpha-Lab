"""The estimator must recover k where it exists, prefer the right shape, and price the cost
of a static k when the truth drifts."""

import numpy as np

from phantom_kernel import estimator, sim


def test_wls_recovers_a_known_line():
    x = np.linspace(0, 5, 50)
    y = 3.0 - 2.0 * x
    a, b, r2 = estimator._wls(x, y, np.ones_like(x))
    assert abs(a - 3.0) < 1e-9 and abs(b + 2.0) < 1e-9 and r2 > 0.999999


def test_fit_exponential_recovers_k(deltas):
    counts = sim.fill_counts(sim.WORLD_A, deltas, n_orders=300_000, seed=0)
    fit = estimator.fit_exponential(deltas, counts)
    assert abs(fit["k"] - sim.WORLD_A.reach_k) < 0.05
    assert fit["r2"] > 0.99


def test_gof_picks_exponential_in_world_a(deltas):
    counts = sim.fill_counts(sim.WORLD_A, deltas, n_orders=300_000, seed=0)
    g = estimator.goodness_of_fit(deltas, counts)
    assert g["winner"] == "exponential"
    assert g["aic_gap"] < 0          # exponential preferred


def test_gof_picks_powerlaw_in_world_b(deltas):
    """The headline H1 falsification: heavy-tailed reach => the AS form is the wrong shape."""
    counts = sim.fill_counts(sim.WORLD_B, deltas, n_orders=300_000, seed=0)
    g = estimator.goodness_of_fit(deltas, counts)
    assert g["winner"] == "power-law"
    assert g["aic_gap"] > 0
    assert g["r2_pow"] > g["r2_exp"]


def test_static_k_misprices_the_spread():
    deltas = np.linspace(0.25, 40.0, 80)
    out = estimator.static_k_spread_error(np.array([0.3, 0.6, 0.9, 1.2]), deltas, n_orders=120_000, seed=0)
    # Each regime's own k is recovered; the single pooled k mis-prices the spread materially.
    assert np.allclose(out["k_recovered_per_regime"], [0.3, 0.6, 0.9, 1.2], atol=0.05)
    assert out["max_abs_spread_pct_error"] > 50.0

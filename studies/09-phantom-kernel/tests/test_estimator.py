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
    rng = np.random.default_rng(0)
    reach = sim.sample_reach(sim.WORLD_A, 300_000, rng)
    counts = sim.fill_counts(sim.WORLD_A, deltas, n_orders=300_000, seed=0)
    g = estimator.goodness_of_fit(reach, deltas, counts)
    assert g["winner"] == "exponential"
    assert g["aic_gap"] < 0          # exponential preferred
    assert g["V"] < 0


def test_gof_picks_powerlaw_in_world_b(deltas):
    """The headline H1 falsification: heavy-tailed reach => the AS form is the wrong shape."""
    rng = np.random.default_rng(0)
    reach = sim.sample_reach(sim.WORLD_B, 300_000, rng)
    counts = sim.fill_counts(sim.WORLD_B, deltas, n_orders=300_000, seed=0)
    g = estimator.goodness_of_fit(reach, deltas, counts)
    assert g["winner"] == "power-law"
    assert g["aic_gap"] > 0
    assert g["V"] > 0
    assert g["r2_pow"] > g["r2_exp"]


def test_reach_likelihood_test_on_exponential_data():
    """Per-observation test, exponential truth: the gap must favour the exponential."""
    rng = np.random.default_rng(1)
    x = rng.exponential(1.0 / 0.6, 100_000)
    out = estimator.reach_likelihood_test(x)
    assert out["winner"] == "exponential"
    assert out["aic_gap"] < 0 and out["bic_gap"] < 0
    assert out["V"] < 0 and out["p"] < 0.05
    assert abs(out["lam"] - 0.6) < 0.02          # the MLE recovers the planted rate


def test_reach_likelihood_test_on_powerlaw_data():
    """Per-observation test, power-law truth: the gap must flip to the power law."""
    rng = np.random.default_rng(1)
    x = 0.686 * (1.0 + rng.pareto(1.7, 100_000))  # Pareto I, the World-B reach law
    out = estimator.reach_likelihood_test(x)
    assert out["winner"] == "power-law"
    assert out["aic_gap"] > 0 and out["bic_gap"] > 0
    assert out["V"] > 0 and out["p"] < 0.05
    assert abs(out["alpha_mle"] - 1.7) < 0.05    # the MLE recovers the planted exponent


def test_reach_likelihood_gap_scales_per_observation():
    """The honest magnitude: the per-order log-likelihood edge is sample-size invariant
    (the old cumulative-count AIC gap doubled with n_orders — the audit's tell)."""
    rng = np.random.default_rng(2)
    x = 0.686 * (1.0 + rng.pareto(1.7, 200_000))
    small = estimator.reach_likelihood_test(x[:100_000])
    big = estimator.reach_likelihood_test(x)
    assert abs(small["ll_per_obs"] - big["ll_per_obs"]) < 0.05


def test_tail_test_distinguishes_powerlaw_from_exponential():
    """The Clauset/Vuong tail test (used on real data) must call clean samples correctly."""
    rng = np.random.default_rng(0)
    pareto = 1.0 + rng.pareto(2.0, 60_000)        # genuine power-law tail
    expo = rng.exponential(1.0, 60_000)           # genuine exponential tail
    assert estimator.tail_test(pareto)["winner"] == "power-law"
    assert estimator.tail_test(expo)["winner"] == "exponential"


def test_static_k_misprices_the_spread():
    deltas = np.linspace(0.25, 40.0, 80)
    out = estimator.static_k_spread_error(np.array([0.3, 0.6, 0.9, 1.2]), deltas, n_orders=120_000, seed=0)
    # Each regime's own k is recovered; the single pooled k mis-prices the spread materially.
    assert np.allclose(out["k_recovered_per_regime"], [0.3, 0.6, 0.9, 1.2], atol=0.05)
    assert out["max_abs_spread_pct_error"] > 50.0


def test_spread_error_shrinks_at_the_trading_horizon():
    """T=1 (all k-term) is the worst case by construction; at the tournament horizon the same
    mis-calibration must cost materially less — the reason the headline is quoted at T=600."""
    deltas = np.linspace(0.25, 40.0, 80)
    ks = np.array([0.3, 0.6, 0.9, 1.2])
    bound = estimator.static_k_spread_error(ks, deltas, n_orders=120_000, seed=0,
                                            horizon=1.0, eval_t=0.0)
    headline = estimator.static_k_spread_error(ks, deltas, n_orders=120_000, seed=0,
                                               horizon=600.0, eval_t=300.0)
    assert headline["max_abs_spread_pct_error"] < bound["max_abs_spread_pct_error"]
    assert headline["max_abs_spread_pct_error"] > 20.0   # still a material mispricing

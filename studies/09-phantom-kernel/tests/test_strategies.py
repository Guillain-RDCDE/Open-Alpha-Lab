"""The AS equations and the four quoters must behave as specified, and the market loop must
respect its own accounting."""

import numpy as np

from phantom_kernel import sim, strategies as st


def test_reservation_price_skews_against_inventory():
    s, gamma, sigma, T = 100.0, 0.1, 0.4, 1.0
    assert st.reservation_price(s, 0.0, 0.0, gamma, sigma, T) == s          # flat -> mid
    assert st.reservation_price(s, 5.0, 0.0, gamma, sigma, T) < s           # long -> below mid
    assert st.reservation_price(s, -5.0, 0.0, gamma, sigma, T) > s          # short -> above mid


def test_skew_is_independent_of_k():
    """The study's punchline, as an invariant: the reservation price contains no k."""
    a = st.reservation_price(100.0, 5.0, 0.0, 0.1, 0.4, 1.0)
    # optimal_spread does depend on k; the reservation price (skew) must not.
    assert a == 100.0 - 5.0 * 0.1 * 0.4**2 * 1.0


def test_optimal_spread_positive_and_increasing_in_sigma():
    lo = st.optimal_spread(0.0, 0.1, 0.2, 0.6, 1.0)
    hi = st.optimal_spread(0.0, 0.1, 0.8, 0.6, 1.0)
    assert lo > 0 and hi > lo


def test_clamp_stops_the_inventory_growing_side():
    q = st.ClampQuoter(half_spread=1.0, max_inventory=10.0)
    bid, ask = q.quote(100.0, 10.0, 0.0)     # already at +max -> stop buying
    assert bid is None and ask == 101.0
    bid, ask = q.quote(100.0, -10.0, 0.0)    # at -max -> stop selling
    assert ask is None and bid == 99.0


def test_run_market_accounting():
    flow = sim.simulate_flow(sim.WORLD_A, n_steps=5_000, dt=0.01, seed=0)
    led = st.run_market(flow, st.SymmetricQuoter(2.0))
    assert led.pnl.shape == (5_001,)
    assert led.inventory.shape == (5_001,)
    assert led.n_fills >= 0


def test_as_controls_inventory_better_than_symmetric():
    """In the textbook world AS must hold far less inventory than a skew-free quoter."""
    flow = sim.simulate_flow(sim.WORLD_A, n_steps=20_000, dt=0.01, seed=0)
    T = 20_000 * 0.01
    led_as = st.run_market(flow, st.ASQuoter(0.1, sim.WORLD_A.sigma, sim.WORLD_A.reach_k, T), T=T)
    led_sym = st.run_market(flow, st.SymmetricQuoter(4.0), T=T)
    assert led_as.inventory.std() < led_sym.inventory.std()

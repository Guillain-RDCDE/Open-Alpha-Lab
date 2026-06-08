"""The faithful port must behave like the article's code — including its quirks."""

import numpy as np

from markov_mint.markov import (
    build_transition_matrix,
    calibrate,
    kelly,
    monte_carlo,
)


def test_transition_rows_normalised():
    prices = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.2, 0.35, 0.4, 0.3, 0.45])
    T = build_transition_matrix(prices, n_states=10)
    sums = T.sum(axis=1)
    # Visited rows sum to 1; unvisited rows are all-zero (sum 0). Nothing in between.
    assert np.all(np.isclose(sums, 1.0) | np.isclose(sums, 0.0))


def test_monte_carlo_is_a_probability():
    rng = np.random.default_rng(0)
    T = build_transition_matrix(np.linspace(0.2, 0.8, 40), n_states=10)
    p = monte_carlo(T, start_state=5, days=20, n_sims=1000, rng=rng)
    assert 0.0 <= p <= 1.0


def test_calibration_is_capped():
    """The table's ceiling (0.958) and floor (0.0043) are the source of the forced-NO bug."""
    assert calibrate(0.999) == 0.958
    assert calibrate(0.5) == 0.5
    assert calibrate(0.0) == 0.0043
    # A near-certain contract is handed a probability strictly below any rich price.
    assert calibrate(0.99) < 0.98


def test_kelly_nonnegative_and_scales():
    # No edge -> no bet; bigger edge -> bigger bet.
    assert kelly(0.5, 50.0) == 0.0
    assert kelly(0.6, 50.0) > 0.0
    assert kelly(0.7, 50.0) > kelly(0.6, 50.0)

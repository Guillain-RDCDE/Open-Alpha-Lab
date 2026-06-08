"""The synthetic generators must produce what they advertise: a martingale null and a real wedge."""

import numpy as np

from markov_mint import data


def test_efficient_market_is_calibrated(efficient):
    """A martingale ends at its outcome, so E[outcome] == E[price]: prices are honest."""
    price = np.array([m.current_price for m in efficient])
    outcome = np.array([m.outcome for m in efficient])
    # The fair price is, on average, the resolution rate — the defining test of "no bias".
    assert abs(outcome.mean() - price.mean()) < 0.05


def test_efficient_true_prob_equals_price(efficient):
    """On the null the traded price IS the fair probability — nothing to arbitrage."""
    gap = np.array([abs(m.current_price - m.true_prob) for m in efficient])
    assert gap.max() < 1e-9


def test_biased_market_has_a_wedge(biased):
    """The planted favorite-longshot distortion pushes the traded price off the fair prob."""
    gap = np.array([m.current_price - m.true_prob for m in biased])
    # Low (longshot) prices trade richer than fair; the mean absolute wedge is non-trivial.
    assert np.abs(gap).mean() > 0.01


def test_prices_within_bounds(efficient):
    for m in efficient:
        assert (m.prices > 0).all() and (m.prices < 1).all()
        assert m.outcome in (0, 1)


def test_deterministic(efficient):
    again = data.efficient_markets(n_markets=400, seed=0)
    assert np.allclose(again[0].prices, efficient[0].prices)
    assert again[-1].outcome == efficient[-1].outcome

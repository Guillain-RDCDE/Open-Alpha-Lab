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
    # The mean absolute wedge is non-trivial — there is a real edge to recover.
    assert np.abs(gap).mean() > 0.01


def test_wedge_sign_is_thaler_ziemba(biased):
    """The SIGN of the planted wedge, market by market — the bias the article quotes.

    Favorite-longshot bias means longshots are over-bet, i.e. priced ABOVE their true
    probability (rich), and favorites priced BELOW theirs (cheap). Log-odds compression
    (gamma < 1) guarantees this strictly for every market: for true_prob < 0.5 the traded
    price must exceed true_prob, and vice versa. An amplifying gamma > 1 would plant the
    opposite wedge — this test is the regression guard against that sign flip.
    """
    longshots = [m for m in biased if m.true_prob < 0.35]
    favorites = [m for m in biased if m.true_prob > 0.65]
    assert len(longshots) > 20 and len(favorites) > 20   # both tails are populated
    assert all(m.current_price > m.true_prob for m in longshots)   # longshots trade RICH
    assert all(m.current_price < m.true_prob for m in favorites)   # favorites trade CHEAP


def test_wedge_sign_worked_example():
    """Algebra check on a synthetic 5¢ longshot: compression prices it above fair."""
    fair = 0.05
    traded = float(data.sigmoid(0.85 * np.log(fair / (1 - fair))))
    assert traded > fair                       # 7.6¢ for a 5.0¢-fair contract: rich
    fav = 0.95
    traded_fav = float(data.sigmoid(0.85 * np.log(fav / (1 - fav))))
    assert traded_fav < fav                    # 92.4¢ for a 95.0¢-fair contract: cheap


def test_prices_within_bounds(efficient):
    for m in efficient:
        assert (m.prices > 0).all() and (m.prices < 1).all()
        assert m.outcome in (0, 1)


def test_no_price_atoms(efficient, biased):
    """No probability mass parked at a clip bound (the 0.99-pin artefact of an old version).

    Prices are sigmoid of a continuous log-odds walk, so every market's current price should
    be distinct — an atom at any bound means a clip is again creating markets whose recorded
    price understates their true resolution rate.
    """
    for basket in (efficient, biased):
        prices = [m.current_price for m in basket]
        assert len(set(prices)) == len(prices)


def test_deterministic(efficient):
    again = data.efficient_markets(n_markets=400, seed=0)
    assert np.allclose(again[0].prices, efficient[0].prices)
    assert again[-1].outcome == efficient[-1].outcome

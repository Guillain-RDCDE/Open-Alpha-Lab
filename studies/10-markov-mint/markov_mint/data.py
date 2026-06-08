"""Synthetic prediction-market generator for the Markov-Mint study — fully offline.

There is no cached Polymarket tape here (and the viral article's headline statistic —
"72.1 M trades, $18.26 bn, Becker 2026" — is an *unverifiable* citation; we treat it as
folklore, not data). That is not a limitation for *this* study: the claim under test is a
**method** claim — *"a Markov chain on price history finds mispricings the price itself
misses"* — and the cleanest way to falsify a method claim is to feed it data whose truth we
control. So the desk builds the market.

Two generators, two ground truths:

    * :func:`efficient_markets` — the **null**. Each market's price is the *correct Bayesian
      posterior* P(resolves YES | information so far), which is a **martingale** by
      construction (tower property): ``E[price_{t+1} | history_t] = price_t``. The price is
      already the best possible estimate of the outcome. A method that "finds an edge" here
      is finding noise — there is, provably, nothing to find. This is where the claim must
      die if it is empty.
    * :func:`biased_markets` — a **planted favorite-longshot wedge**. The *fair* posterior is
      simulated exactly as above, but the *traded* price is pushed toward the centre
      (``traded = sigmoid(gamma * logit(fair))`` with ``gamma > 1``): cheap longshots trade
      *richer* than they resolve, expensive favorites trade *cheaper* — the classic bias the
      article quotes. Here a real edge **does** exist, of a known size, so we can ask the
      honest follow-up: how big is it, where does it live, and does *the Markov chain* (as
      opposed to a one-line price→truth lookup) recover any of it?

Each :class:`Market` carries its full pre-resolution price history, the days left to expiry,
and — the thing a real trader never gets but a simulator does — the **realized outcome**, so
every trade can be scored against ground truth rather than against the price it was lifted
from. Everything is deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Clip prices into (EPS, 1-EPS) so log-odds stay finite and discretisation has room at the
# tails (a contract never trades at a literal 0 or 100 on a live book either).
EPS = 0.01


def logit(p: np.ndarray | float) -> np.ndarray | float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class Market:
    """One binary prediction market, simulated end to end.

    ``prices`` is the YES-price history *up to now* (what a trader can see and what the
    article's machine is fed); ``current_price`` is the last point; ``days_to_expiry`` is the
    Monte-Carlo horizon; ``outcome`` ∈ {0,1} is the **realized resolution** (ground truth,
    used only to score trades); ``true_prob`` is the fair P(YES) now (== ``current_price`` for
    an efficient market, and the un-distorted posterior for a biased one).
    """

    prices: np.ndarray
    days_to_expiry: int
    outcome: int
    true_prob: float

    @property
    def current_price(self) -> float:
        return float(self.prices[-1])


def _posterior_path(outcome: int, p0: float, n_steps: int, delta: float,
                    rng: np.random.Generator) -> np.ndarray:
    """The exact Bayesian posterior P(YES | signals so far) — a martingale by construction.

    Prior odds are ``p0/(1-p0)``. Each step delivers a Gaussian signal ``s ~ N(±delta, 1)``
    whose mean's sign is the (hidden) outcome; the log-likelihood-ratio increment is exactly
    ``2*delta*s``, so the posterior log-odds is a random walk ``logit(p0) + Σ 2*delta*s_k``.
    Because it is a genuine posterior under the prior the outcome was drawn from, the
    *probability* ``sigmoid(L_t)`` satisfies ``E[p_{t+1} | history_t] = p_t`` — the defining
    property of a fair price. ``delta`` sets how fast information accrues (how far the price
    has wandered toward its eventual resolution by "now").
    """
    L = float(logit(p0))
    out = np.empty(n_steps + 1)
    out[0] = p0
    mu = delta if outcome == 1 else -delta
    s = rng.normal(mu, 1.0, size=n_steps)
    L = L + np.cumsum(2.0 * delta * s)
    out[1:] = sigmoid(L)
    return np.clip(out, EPS, 1.0 - EPS)


def _draw_markets(n_markets: int, hist_len: int, days_to_expiry: int,
                  delta: float, prior_scale: float, gamma: float,
                  seed: int) -> list[Market]:
    """Shared engine: draw ``n_markets`` posterior-martingale markets, optionally distorted.

    ``prior_scale`` spreads the initial fair prices across (0,1) including the tails
    (``p0 = sigmoid(N(0, prior_scale))``); ``gamma == 1`` leaves the price efficient, while
    ``gamma > 1`` applies the favorite-longshot wedge to the *traded* price only.
    """
    rng = np.random.default_rng(seed)
    markets: list[Market] = []
    for _ in range(n_markets):
        p0 = float(sigmoid(rng.normal(0.0, prior_scale)))
        outcome = int(rng.random() < p0)
        fair = _posterior_path(outcome, p0, hist_len, delta, rng)
        if gamma == 1.0:
            traded = fair
        else:
            traded = np.clip(sigmoid(gamma * logit(fair)), EPS, 1.0 - EPS)
        markets.append(
            Market(
                prices=traded,
                days_to_expiry=days_to_expiry,
                outcome=outcome,
                true_prob=float(fair[-1]),
            )
        )
    return markets


def efficient_markets(n_markets: int = 2000, hist_len: int = 60,
                      days_to_expiry: int = 30, delta: float = 0.15,
                      prior_scale: float = 1.6, seed: int = 0) -> list[Market]:
    """The null: markets whose price is the fair posterior (a martingale). No edge exists."""
    return _draw_markets(n_markets, hist_len, days_to_expiry, delta,
                         prior_scale, gamma=1.0, seed=seed)


def biased_markets(n_markets: int = 2000, hist_len: int = 60,
                   days_to_expiry: int = 30, delta: float = 0.15,
                   prior_scale: float = 1.6, gamma: float = 1.18,
                   seed: int = 1) -> list[Market]:
    """A planted favorite-longshot wedge: traded price ≠ fair prob by a known amount.

    ``gamma > 1`` pushes the fair posterior to a more-extreme *traded* price, so low prices
    are over-stated (longshots trade rich) and high prices under-stated (favorites trade
    cheap) — a real, exploitable edge of controlled size, used for the machinery-sanity and
    "could you trade it" beats.
    """
    return _draw_markets(n_markets, hist_len, days_to_expiry, delta,
                         prior_scale, gamma=gamma, seed=seed)


def markets_frame(markets: list[Market]):
    """A tidy frame (one row per market) for fingerprinting and quick inspection.

    Columns: current price, days to expiry, realized outcome, fair P(YES). Deterministic, so
    its content hash pins the exact synthetic sample behind a published number.
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "price": [m.current_price for m in markets],
            "days": [m.days_to_expiry for m in markets],
            "outcome": [m.outcome for m in markets],
            "true_prob": [m.true_prob for m in markets],
        }
    )

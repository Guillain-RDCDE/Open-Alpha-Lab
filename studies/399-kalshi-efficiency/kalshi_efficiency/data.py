"""Data layer for Study 399 — Kalshi-Efficiency (event-contract calibration).

Two sources, both offline-friendly. **There is no free real Kalshi tape**, so the "real"
path here is an explicit, clearly-labelled *illustrative* book of resolved binary
event-contracts (a methods demo, NOT a backtest of live Kalshi prices). The synthetic path is
the deterministic positive control with a planted favourite–longshot bias knob.

A binary event-contract is a row with:
  * ``price``   — the market price in dollars, ``0<price<1`` (i.e. the crowd's implied prob);
  * ``outcome`` — the realized binary, ``1`` if the event resolved YES else ``0``;
  * ``ts``      — a decorative resolution timestamp (period index; not used for inference).

**Calibration** asks: among contracts priced near ``p``, did YES occur about ``p`` of the time?
The believer's edge would be a *systematic* gap between ``price`` and realized frequency — the
classic **favourite–longshot bias** (longshots over-bet ⇒ over-priced; favourites under-bet ⇒
under-priced). We make that gap a single tunable knob so the engine can be validated end-to-end.

Pure numpy + pandas + stdlib for the offline path. ``fetch_real`` builds the cached illustrative
tape and is the only function that touches anything outside the package; it is never imported by
the notebooks' offline cells (which read the cache or fall back to the frozen ``R`` dict).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "kalshi_contracts.csv")

# Size of the illustrative resolved-contract book. Large enough that a calibration curve is
# legible, small enough that finite-sample wiggle is the visible (and honest) story.
N_CONTRACTS = 4_000

# The favourite–longshot bias baked into the *illustrative real tape*. This is a transparent
# modelling choice, NOT a measurement of live Kalshi: a mild racetrack-style distortion so the
# calibration curve has something to show. The verdict does not hinge on its exact size — it
# hinges on the fact that a few-cent tilt is fee-and-spread-eaten and statistically slippery.
ILLUSTRATIVE_EDGE = 0.04


# --------------------------------------------------------------------------- #
# "Real" tape — an explicit, clearly-labelled ILLUSTRATIVE book (no free Kalshi feed)
# --------------------------------------------------------------------------- #
def fetch_real(path: str = DEFAULT_CACHE, n: int = N_CONTRACTS,
               edge: float = ILLUSTRATIVE_EDGE, seed: int = 1399) -> pd.DataFrame:
    """Build and cache the illustrative resolved-contract book.

    NOT a network call and NOT live Kalshi data (Kalshi's resolved history is not free). This
    constructs a transparent, seeded book of binary event-contracts whose prices are
    well-calibrated *except* for a mild favourite–longshot tilt of size ``edge`` — labelled an
    illustration everywhere. Used once to populate ``_cache/kalshi_contracts.csv``; never
    imported by the offline notebook cells.
    """
    df = _make_book(n=n, edge=edge, seed=seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached illustrative resolved-contract book (price, outcome, ts)."""
    df = pd.read_csv(path)
    return df


# --------------------------------------------------------------------------- #
# Shared book generator (used by both the illustrative tape and the synthetic control)
# --------------------------------------------------------------------------- #
def _make_book(n: int, edge: float, seed: int) -> pd.DataFrame:
    """Deterministic book of ``n`` binary contracts with a favourite–longshot bias of ``edge``.

    A market price ``p`` is drawn (a realistic spread of binaries — many near-coin-flips, fewer
    extreme favourites/longshots). The *true* YES probability is warped away from ``p`` by the
    favourite–longshot map: longshots (low ``p``) resolve YES *less* often than priced
    (over-priced), favourites (high ``p``) resolve YES *more* often (under-priced). ``edge=0``
    ⇒ true prob == price ⇒ perfectly calibrated by construction.
    """
    rng = np.random.default_rng(seed)
    # Prices: a Beta(1.4,1.4) gives a believable spread of binaries across [0.02, 0.98].
    p = rng.beta(1.4, 1.4, size=n)
    p = np.clip(p, 0.02, 0.98)
    # Favourite–longshot warp on the LOG-ODDS scale (monotone, fixed-point at p=0.5):
    #   true logit = (1 + edge_strength) * logit(price)
    # edge>0 steepens the odds: longshots get truer-longer, favourites truer-shorter.
    # Translate the headline cent-edge into a log-odds slope so edge=0 is exactly calibrated.
    k = 1.0 + 6.0 * edge
    logit = np.log(p / (1.0 - p))
    true_p = 1.0 / (1.0 + np.exp(-k * logit))
    outcome = (rng.random(n) < true_p).astype(int)
    # decorative resolution stamps (period index -> never OutOfBounds; not used for inference)
    ts = pd.period_range("2021-07", periods=n, freq="D").astype(str)
    return pd.DataFrame({"price": p, "outcome": outcome, "ts": ts})


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_book(n: int = N_CONTRACTS, edge: float = 0.0, seed: int = 399) -> pd.DataFrame:
    """Deterministic positive control: a book with a *known* favourite–longshot edge.

    ``edge=0`` must be perfectly calibrated — the reliability curve hugs the diagonal and no
    test manufactures a longshot/favourite return spread out of finite-sample wiggle. A large
    ``edge`` must light up: the calibration curve bows, the Brier reliability term grows, and
    the long-favourite / short-longshot spread becomes significant. This is the engine's
    end-to-end validation, on data where we *know* the truth.
    """
    return _make_book(n=n, edge=edge, seed=seed)

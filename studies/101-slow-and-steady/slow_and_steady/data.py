"""Data layer for Study 101 (Slow-and-Steady).

Two tapes, one shape (a daily frame with a ``close`` column):

- ``synthetic_daily`` — a *deterministic, offline* generator with a single ``drift``
  knob, the study's positive/negative control in a bottle. The race is dollar-cost
  averaging (DCA) vs lump-sum (LS): DCA only out-earns LS when the market *falls* over
  the deployment window, so the sign of ``drift`` is the planted truth.
    - ``drift < 0`` → a **down-drifting** tape. Buying in slowly means buying cheaper on
      average, so **DCA should win** terminal wealth most of the time.
    - ``drift > 0`` → an **up-drifting** tape (the empirical norm for equities). Money
      put to work sooner compounds longer, so **lump-sum should win**.
  This is exactly the asymmetry the literature predicts (Vanguard 2012/2023): LS wins
  whenever expected returns are positive, which they are on average.
- ``load_real`` — the real daily tape via :mod:`quantlab.data` (Yahoo, cached parquet),
  total-return adjusted so dividends compound inside both the lump-sum and the
  drip-fed dollars — a fair, apples-to-apples wealth comparison.

No look-ahead is baked in here: the rolling-window wealth engine in ``strategy.py``
only ever uses prices from *within* each window, deploying cash forward in time.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 6000,
    drift: float = 0.0,
    daily_vol: float = 0.010,
    start: str = "2000-01-03",
    seed: int = 101,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close series with a tunable annual drift.

    ``drift`` is the *annualised* log-drift of a geometric random walk:

    - ``drift < 0`` → a down-drifting tape. Spreading the buy-in (DCA) lowers the average
      purchase price, so **DCA should out-earn lump-sum** on most windows — the positive
      control for "DCA wins when the market falls".
    - ``drift > 0`` → an up-drifting tape (equities' empirical norm). Deploying sooner
      compounds longer, so **lump-sum should out-earn DCA** — the study's real-world null.
    - ``drift == 0`` → a flat, zero-drift walk; LS and DCA are near-tied in expectation,
      with LS's slight edge coming only from its longer average time in the market.

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column and ``truth``
    records the planted parameters (incl. the sign that names the expected winner).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    mu = drift / 252.0
    rets = rng.normal(mu, daily_vol, n_days)
    close = 100.0 * np.exp(np.cumsum(rets))
    frame = pd.DataFrame({"close": close}, index=idx)
    frame.index.name = "Date"
    truth = {
        "drift": drift,
        "n_days": n_days,
        "daily_vol": daily_vol,
        "seed": seed,
        "expected_winner": ("dca" if drift < 0 else "lump_sum"),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return close via quantlab, cache-first
# ---------------------------------------------------------------------------
def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-29",
    end: str | None = None,
    mode: str = "total_return",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily close for ``ticker`` as a one-column ``close`` frame.

    Delegates to :func:`quantlab.data.fetch` (cache-first parquet under the repo
    ``_cache/``). ``mode="total_return"`` folds dividends back in so the dollars compound
    identically whether they were invested all at once (lump-sum) or fed in over a year
    (DCA) — the only fair way to score terminal wealth. Returns a frame with a single
    lower-case ``close`` column, indexed by date.
    """
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    from quantlab import data as qdata

    raw = qdata.fetch(ticker, start=start, end=end, mode=mode, use_cache=use_cache)
    close = raw["Close"].rename("close")
    return close.to_frame()


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(frame["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]

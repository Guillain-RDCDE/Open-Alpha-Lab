"""Data layer for Study 98 (High-Noon).

Two tapes, one shape (a daily frame with a ``close`` column):

- ``synthetic_daily`` — a *deterministic, offline* generator. A single knob
  (``ath_reversal``) plants the only thing the "buying at a high is dangerous" claim
  needs to be true: a mean-reversion-*at-highs* drag. ``ath_reversal=0`` is a driftful
  geometric random walk — all-time highs are simply where an uptrend has been, so they
  cannot predict *worse* forward returns. ``ath_reversal>0`` subtracts drift on the days
  the tape is sitting at (or just under) its running peak, so an ATH genuinely forecasts
  a reversal. This is the study's null (and its positive control) in a bottle: the
  harness MUST detect "ATH is bearish" when it is planted, and MUST NOT when it isn't.
- ``load_real`` — the real daily tape via :mod:`quantlab.data` (Yahoo, cached parquet),
  total-return adjusted so forward returns are the dividend-inclusive returns an investor
  actually earns. Cache-first; the network is touched only on a cache miss.

No look-ahead is baked in here — the ATH flag and the forward-return alignment live in
``strategy.py`` (the at-ATH state is known at close *t*; the forward return is the return
*after* *t*).
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
    n_days: int = 8000,
    ath_reversal: float = 0.0,
    daily_vol: float = 0.010,
    near_pct: float = 0.01,
    start: str = "1994-01-03",
    seed: int = 98,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close series with a tunable mean-reversion-at-highs drag.

    - ``ath_reversal = 0`` → a single constant-drift geometric random walk. All-time
      highs are just where the uptrend currently sits, so being *at* a high carries no
      information about the *forward* return. This is the study's null: the harness must
      find no bearishness in highs.
    - ``ath_reversal > 0`` → on every day whose close is within ``near_pct`` of the
      running all-time high, the *next* day's drift is reduced by ``ath_reversal`` of a
      reference annual drift (i.e. highs genuinely forecast a reversal). This is the
      positive control: the harness must now flag highs as bearish.

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column and ``truth``
    records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    mu = 0.09 / 252.0  # ~9%/yr baseline drift
    drag = ath_reversal * 1.20 / 252.0  # drift removed the day after an at-high close

    close = np.empty(n_days)
    close[0] = 100.0
    running_max = close[0]
    near_high = np.zeros(n_days, dtype=bool)
    for t in range(1, n_days):
        # The at-high state is evaluated on the PREVIOUS close (known before today's move).
        at_high = close[t - 1] >= running_max * (1.0 - near_pct)
        near_high[t - 1] = at_high
        mu_t = mu - (drag if at_high else 0.0)
        r = rng.normal(mu_t, daily_vol)
        close[t] = close[t - 1] * np.exp(r)
        running_max = max(running_max, close[t])
    # Flag for the final day too (relative to its own running max).
    near_high[n_days - 1] = close[n_days - 1] >= running_max * (1.0 - near_pct)

    frame = pd.DataFrame({"close": close}, index=idx)
    frame.index.name = "Date"
    truth = {
        "ath_reversal": ath_reversal,
        "n_days": n_days,
        "daily_vol": daily_vol,
        "near_pct": near_pct,
        "seed": seed,
        "frac_near_high": float(near_high.mean()),
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
    ``_cache/``). ``mode="total_return"`` folds dividends back in so the forward returns
    are the ones an investor actually earns and the running all-time high is a *total
    return* high (the only kind that compounds). Returns a frame with a single lower-case
    ``close`` column, indexed by date.
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

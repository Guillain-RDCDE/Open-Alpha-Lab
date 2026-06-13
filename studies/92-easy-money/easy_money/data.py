"""Data layer for Study 92 (Easy-Money).

Two tapes, one shape (a daily frame with a ``close`` column):

- ``synthetic_vol_etp`` — a *deterministic, offline* generator of a volatility-ETP
  tape. A ``carry`` knob plants the only thing the short-vol trade can possibly
  harvest: a relentless daily roll-decay (contango bleed). With ``carry=0`` the
  tape is a no-drift random walk — no roll to harvest, so shorting it earns
  nothing risk-adjusted. With ``carry>0`` the tape bleeds steadily lower *and*
  occasionally suffers an upward jump-spike (a vol explosion), so the short-vol
  strategy earns a quiet drip and then takes a catastrophic, fat-tailed hit — the
  study's null AND its positive control in a bottle.
- ``load_real`` — the real daily tapes via :mod:`quantlab.data` (Yahoo, cached
  parquet): ``VIXY`` the long-vol ETP (since 2011), total-return adjusted so the
  reverse-splits are folded in and the series is a continuous NAV-equivalent, and
  ``^VIX`` the spot index (since 1990). Cache-first; the network is touched only on
  a cache miss.

No look-ahead is baked in here — the one-day execution lag lives in
``strategy.py`` (the short is sized at close *t*, it earns the return of *t+1*).
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
def synthetic_vol_etp(
    n_days: int = 4000,
    carry: float = 0.0,
    daily_vol: float = 0.018,
    jump_prob: float = 0.004,
    jump_mean: float = 0.30,
    start: str = "2011-01-03",
    seed: int = 92,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily vol-ETP NAV with a tunable roll-decay and jump-spikes.

    The generator works in **simple daily returns** so the backtest's ``pct_change``
    sees exactly the planted moves, and so a short of leverage 1 cannot lose more
    than the largest single up-move (jumps are capped to a realistic magnitude, as
    the real VIXY's worst up-day is about +43%).

    - ``carry = 0`` → a zero-drift, no-jump tape: simple returns are mean-zero, so
      shorting it earns ~0 with no fat tail. There is no contango bleed to harvest
      and no spike to fear — a risk-adjusted Sharpe near zero. This is the null.
    - ``carry > 0`` → the tape bleeds lower every day by a planted simple rate
      (``-carry`` per day, the contango roll the long-ETP pays the short) AND it
      jumps *upward* on rare days (a vol explosion that hammers the short). The
      short-vol carry should then bank a steady positive drip with a clearly
      positive Sharpe AND carry a fat LEFT tail (negative skew, high kurtosis, a
      deep drawdown). This is the positive control: the harness must detect both
      the carry AND its tail.

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    if carry <= 0:
        rets = rng.normal(0.0, daily_vol, n_days)
        n_jumps = 0
    else:
        # Steady negative simple-return drift (the daily contango bleed) + noise.
        rets = rng.normal(-carry, daily_vol, n_days)
        # Rare UPWARD jumps — the vol-spike tail that hammers the short. Capped at
        # +60% so a 1x short loses at most ~60% on the worst day (realistic scale).
        jumps = rng.random(n_days) < jump_prob
        n_jumps = int(jumps.sum())
        jump_size = np.minimum(rng.exponential(jump_mean, n_days), 0.60)
        rets = np.where(jumps, rets + jump_size, rets)

    close = 100.0 * np.cumprod(1.0 + rets)
    frame = pd.DataFrame({"close": close}, index=idx)
    frame.index.name = "Date"
    truth = {
        "carry": carry,
        "n_days": n_days,
        "daily_vol": daily_vol,
        "jump_prob": jump_prob,
        "jump_mean": jump_mean,
        "seed": seed,
        "n_jumps": n_jumps,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily total-return close via quantlab, cache-first
# ---------------------------------------------------------------------------
def load_real(
    ticker: str = "VIXY",
    start: str = "2011-01-03",
    end: str | None = None,
    mode: str = "total_return",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily close for ``ticker`` as a one-column ``close`` frame.

    Delegates to :func:`quantlab.data.fetch` (cache-first parquet under the repo
    ``_cache/``). For ``VIXY``, ``mode="total_return"`` folds the reverse-splits
    (and any distributions) back in, so the series is a single continuous
    NAV-equivalent rather than a tape that jumps at every 1:4 reverse split. For
    ``^VIX`` the mode is a no-op (an index has no splits/dividends). Returns a
    frame with a single lower-case ``close`` column, indexed by date.
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

"""Data layer for Study 247 (Bond-Seasonality).

Three things live here, all sharing one shape (a daily frame with a ``close`` column):

- ``tom_mask`` - derives the turn-of-month (TOM) window: the last ``n_end`` and first
  ``n_start`` trading days of every calendar month.  No external calendar required; the
  window is fixed by the trading index itself.

- ``load_real`` - the real daily total-return tape for TLT (20+ yr Treasury ETF) and IEF
  (7-10 yr Treasury ETF) via :mod:`quantlab.data` (Yahoo, cached parquet), cached
  study-locally under ``_cache/`` to avoid races in parallel batch runs.

- ``synthetic_daily`` - a deterministic, offline business-day tape that optionally PLANTS
  a TOM return bump on the derived TOM days.  ``planted=True`` is the positive control
  (the harness must detect it); ``planted=False`` is the negative control (it must not).

No look-ahead is baked in: a TOM day is fixed by the calendar in advance and needs
**no execution lag** (the rule is: you know well before month-end which day is the last
trading day; enter at that close, exit at the close of the first trading day of the next
month).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Study-local cache - keeps each agent's parquet files isolated.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Turn-of-month mask
# ---------------------------------------------------------------------------
def tom_mask(
    idx: pd.DatetimeIndex,
    n_end: int = 2,
    n_start: int = 2,
) -> pd.Series:
    """Boolean Series aligned to ``idx``: True on turn-of-month days, False otherwise.

    The TOM window covers the last ``n_end`` and first ``n_start`` trading days of each
    calendar month.  The default (2 + 2 = 4 days per month) captures the well-documented
    month-end rebalancing and institutional bid window without expanding the definition to
    the point of vacuity.

    Derivation is purely from the trading index; no external calendar is consulted.
    """
    mask = pd.Series(False, index=idx)
    periods = idx.to_period("M")
    for p in periods.unique():
        days = idx[periods == p].sort_values()
        if len(days) >= n_end:
            for d in days[-n_end:]:
                mask.loc[d] = True
        if len(days) >= n_start:
            for d in days[:n_start]:
                mask.loc[d] = True
    return mask


# ---------------------------------------------------------------------------
# Synthetic tape - the deterministic offline control
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 20,
    bump: float = 0.0035,
    planted: bool = True,
    daily_vol: float = 0.007,
    mu: float = 0.00015,
    start: str = "2000-01-01",
    seed: int = 247,
    n_end: int = 2,
    n_start: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible business-day close series with an optional TOM bump planted.

    Builds a pure business-day (Mon-Fri) index, then derives the TOM window via
    :func:`tom_mask`, and optionally adds ``bump`` to the returns on every TOM day.

    - ``planted=True``  -> bump on TOM days.  The harness must detect a significant gap.
    - ``planted=False`` -> flat i.i.d. tape.  The harness must NOT find a gap.

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column and ``truth``
    records the planted parameters.

    Note: the synthetic tape uses a plain business-day index with no engineered holidays.
    This is intentional: the TOM rule is derived purely from the trading index (first/last
    N days of each calendar month) and does not depend on holiday gaps.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_years * 252)

    is_tom = tom_mask(idx, n_end=n_end, n_start=n_start).to_numpy()

    rets = rng.normal(mu, daily_vol, len(idx))
    if planted:
        rets[is_tom] = rets[is_tom] + bump

    close = 100.0 * np.exp(np.cumsum(rets))
    frame = pd.DataFrame({"close": close}, index=idx)
    frame.index.name = "Date"
    truth = {
        "n_years": n_years,
        "bump": bump if planted else 0.0,
        "planted": planted,
        "n_days": len(idx),
        "n_tom": int(is_tom.sum()),
        "seed": seed,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape - daily total-return via quantlab, cache-first
# ---------------------------------------------------------------------------
def load_real(
    ticker: str = "TLT",
    start: str = "2002-07-30",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Real daily close for ``ticker`` as a one-column ``close`` frame (total-return).

    Caches to the study-local ``_cache/`` directory (gitignored) to avoid clashes in
    parallel batch runs.  Both TLT and IEF are intended callers:

    - ``load_real("TLT")`` - 20+ yr Treasury ETF, from 2002-07-30.
    - ``load_real("IEF")`` - 7-10 yr Treasury ETF, from 2002-07-30.

    Returns a frame with a single lower-case ``close`` column, indexed by date.
    """
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    from quantlab import data as qdata

    cache_dir = DEFAULT_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker.replace('^', '')}.parquet")

    if use_cache and os.path.exists(cache_path):
        raw_close = pd.read_parquet(cache_path)
    else:
        raw = qdata.fetch(ticker, start=start, end=end, mode="total_return", use_cache=True)
        raw_close = raw[["Close"]].rename(columns={"Close": "close"})
        raw_close.to_parquet(cache_path)

    if "close" not in raw_close.columns and "Close" in raw_close.columns:
        raw_close = raw_close.rename(columns={"Close": "close"})

    frame = raw_close[["close"]].copy()
    if start:
        frame = frame.loc[start:]
    if end:
        frame = frame.loc[:end]
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(frame["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]

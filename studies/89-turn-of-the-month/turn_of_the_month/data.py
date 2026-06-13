"""Data layer for Study 89 (Turn-of-the-Month).

Two tapes, one shape (a daily frame with a ``close`` column):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``bump`` knob plants
  the only thing a turn-of-month (TOM) test can possibly harvest: a calendar-clustered
  excess return on the last trading day of each month plus the first three of the next.
  ``bump=0`` is a flat i.i.d. geometric random walk — there is no TOM cluster, so the
  harness must *not* find one. ``bump>0`` adds a per-day excess return only on the TOM
  days, so the harness must detect it. This is the study's null (and its positive
  control) in a bottle.
- ``load_real`` — the real daily tape via :mod:`quantlab.data` (Yahoo, cached parquet).
  ``SPY`` in ``total_return`` mode is the headline (dividends folded in so the
  buy-and-hold benchmark is fair). ``^GSPC`` in ``split_only`` mode is available as a
  longer, **price-only** sample (no dividends — stated as such; never call it total
  return).

The TOM window is **calendar-known** — it is defined purely by the trading-day-of-month
position, which is knowable in advance — so there is NO execution lag to apply (a
calendar rule is not a forecast). The day-labelling itself lives in ``strategy.py``.
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
def _tom_mask_from_index(idx: pd.DatetimeIndex, last: int = 1, first: int = 3) -> np.ndarray:
    """Boolean mask: True on the TOM window (last ``last`` trading days of month t-1
    through trading day +``first`` of month t).

    Built purely from the calendar position within each month — the same labelling the
    strategy uses, duplicated here only so the synthetic generator can plant the bump on
    exactly the days the harness will later test (a self-contained positive control).
    """
    s = pd.Series(np.arange(len(idx)), index=idx)
    ym = idx.to_period("M")
    # Trading-day-of-month, 1-based, counting from the start of each month.
    tdom = s.groupby(ym).cumcount() + 1
    # Trading-days-from-month-end, 1-based (1 == last trading day of the month).
    n_in_month = s.groupby(ym).transform("size")
    tdfe = n_in_month - tdom + 1
    is_first = tdom.to_numpy() <= first
    is_last = tdfe.to_numpy() <= last
    return is_first | is_last


def synthetic_daily(
    n_days: int = 6000,
    bump: float = 0.0,
    daily_vol: float = 0.010,
    base_drift: float = 0.05 / 252.0,
    start: str = "2000-01-03",
    seed: int = 89,
    last: int = 1,
    first: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close series with a tunable turn-of-month bump.

    - ``bump = 0`` → a single flat-drift geometric random walk. Returns are i.i.d.; there
      is no calendar cluster, so a TOM test must find nothing significant. This is the
      study's null.
    - ``bump > 0`` → the same walk plus a constant excess daily return of ``bump`` added
      only on the TOM days (last trading day of month t-1 .. trading day +``first`` of
      month t). A correct harness must detect a TOM premium here — the positive control.

    Returns ``(frame, truth)`` where ``frame`` has a single ``close`` column and ``truth``
    records the planted parameters (including ``frac_tom`` and the planted ``bump``).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    rets = rng.normal(base_drift, daily_vol, n_days)
    tom = _tom_mask_from_index(idx, last=last, first=first)
    if bump != 0.0:
        rets = rets + bump * tom.astype(float)
    close = 100.0 * np.exp(np.cumsum(rets))
    frame = pd.DataFrame({"close": close}, index=idx)
    frame.index.name = "Date"
    truth = {
        "bump": bump,
        "n_days": n_days,
        "daily_vol": daily_vol,
        "base_drift": base_drift,
        "seed": seed,
        "last": last,
        "first": first,
        "frac_tom": float(tom.mean()),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — daily close via quantlab, cache-first
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
    ``_cache/``). The headline run uses ``SPY`` with ``mode="total_return"`` (dividends
    folded in) so buy-and-hold is a fair benchmark for a strategy that sits in cash part
    of the time. ``^GSPC`` with ``mode="split_only"`` gives a longer **price-only**
    sample (no dividends — call it price-only, never total return). Returns a frame with
    a single lower-case ``close`` column, indexed by date.
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

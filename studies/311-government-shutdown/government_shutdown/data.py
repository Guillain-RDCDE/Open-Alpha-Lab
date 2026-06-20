"""Data layer for Study 311 (Government-Shutdown).

Two tapes, one shape (a tz-naive daily OHLCV frame of *total-return* SPY prices):

- ``synthetic_daily`` — a *deterministic, offline* generator. A geometric random walk
  with an optional **planted event effect**: on a list of "event" dates we add a
  positive drift bump over the days *after* the event (the positive control), or nothing
  at all (the null). This lets a test assert the event-study engine recovers a planted
  buy-the-dip bounce only when one is really there.
- ``load_real`` — the real daily **total-return** SPY tape, cache-first via the shared
  ``_cache/SPY_total_return.parquet`` panel (yfinance ``auto_adjust=True`` ⇒ dividends +
  splits folded in ⇒ total return), with a graceful ``quantlab.data`` fallback and, only
  on an explicit ``fetch=True``, a network pull. The test-suite and reproducible core
  never touch the network.

THE SHUTDOWN TABLE is hardcoded below: US federal *funding-gap* shutdowns that actually
furloughed workers, with the trading-relevant START date (the first session on or after
the funding lapse). Brief technical lapses with no real furlough are excluded and the
exclusion is stated, in the open.

No look-ahead is baked in here — the event-window bookkeeping lives in ``strategy.py``.
The shutdown start date is *calendar-known* (announced when funding lapses), so the
canonical event window needs no extra execution lag; a ``+1`` session entry lag is
offered as a robustness check in the engine.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
PANEL = os.path.join(DEFAULT_CACHE, "SPY_total_return.parquet")

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# The hardcoded shutdown table
# ---------------------------------------------------------------------------
# US federal government *funding-gap* shutdowns that furloughed workers, since SPY's
# 1993 inception (so every event has a real SPY tape around it). The date is the first
# CALENDAR day of the funding gap; the engine snaps it to the first trading session on
# or after it. The "days" column is the duration in calendar days, for context only.
#
# Sources: Congressional Research Service, "Shutdown of the Federal Government: Causes,
# Processes, and Effects" (RL34680); Wikipedia "Government shutdowns in the United States".
# Brief sub-day lapses with no furlough (e.g. the Feb-2018 overnight lapse) are excluded;
# the canonical list keeps the shutdowns markets actually noticed.
SHUTDOWNS = pd.DataFrame(
    [
        # name,                       start,        days
        ("1995 (Nov, Clinton/Gingrich)", "1995-11-14", 5),
        ("1995-96 (Dec-Jan, Clinton)",   "1995-12-16", 21),
        ("2013 (Oct, ACA/Obama)",        "2013-10-01", 16),
        ("2018 (Jan, Schumer)",          "2018-01-20", 3),
        ("2018-19 (Dec-Jan, border wall)", "2018-12-22", 35),
    ],
    columns=["name", "start", "days"],
)
SHUTDOWNS["start"] = pd.to_datetime(SHUTDOWNS["start"])


def shutdown_starts() -> pd.DatetimeIndex:
    """The funding-gap start dates as a tz-naive DatetimeIndex."""
    return pd.DatetimeIndex(SHUTDOWNS["start"].to_numpy(), name="start")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 8400,
    event_effect: float = 0.0,
    event_window: int = 20,
    n_events: int = 40,
    drift: float = 0.07,
    daily_vol: float = 0.011,
    start: str = "1993-01-29",
    seed: int = 311,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict]:
    """A reproducible daily total-return OHLCV tape with optional planted event bumps.

    The close follows a geometric random walk (drift + i.i.d. normal shocks). On
    ``n_events`` evenly-spaced "event" sessions we add an extra per-day drift of
    ``event_effect`` over the ``event_window`` sessions *after* the event — the planted
    "buy-the-dip" bounce.

    - ``event_effect = 0`` → a pure walk; events carry no information. This is the null:
      the engine must NOT find a significant post-event drift.
    - ``event_effect > 0`` → a real positive bounce after each event. This is the
      positive control: the engine must recover it.

    Returns ``(bars, events, truth)`` where ``events`` is the DatetimeIndex of the planted
    event sessions and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    mu = drift / TRADING_DAYS_PER_YEAR
    log_ret = mu + rng.normal(0.0, daily_vol, n_days)

    # Plant evenly-spaced events, leaving room for the window on both sides.
    lo, hi = event_window + 5, n_days - event_window - 5
    if hi > lo and n_events > 0:
        event_locs = np.linspace(lo, hi, n_events).round().astype(int)
        event_locs = np.unique(event_locs)
        for e in event_locs:
            log_ret[e : e + event_window] += event_effect
    else:
        event_locs = np.array([], dtype=int)

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, daily_vol * 0.3, n_days - 1))
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
    hi_px = np.maximum(open_, close) + wick
    lo_px = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi_px, "low": lo_px, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    events = pd.DatetimeIndex(bars.index[event_locs], name="event") if event_locs.size else \
        pd.DatetimeIndex([], name="event")
    truth = {
        "event_effect": event_effect,
        "event_window": event_window,
        "n_events": int(events.size),
        "drift": drift,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, events, truth


def synthetic_null_events(
    bars: pd.DataFrame,
    n_events: int = 40,
    window: int = 20,
    seed: int = 7,
) -> pd.DatetimeIndex:
    """Draw ``n_events`` random session dates from ``bars`` — the placebo/event-null control.

    These are *random* dates with no special meaning, used to ask: does the post-event
    drift the engine measures around the real shutdowns differ from what you'd measure
    around any random date at all? (The synthetic-event null in the brief.)
    """
    rng = np.random.default_rng(seed)
    lo, hi = window + 5, len(bars) - window - 5
    locs = rng.choice(np.arange(lo, hi), size=min(n_events, hi - lo), replace=False)
    return pd.DatetimeIndex(sorted(bars.index[locs]), name="null_event")


# ---------------------------------------------------------------------------
# Real tape — daily total-return SPY, cache-first
# ---------------------------------------------------------------------------
def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-01",
    end: str | None = None,
    use_cache: bool = True,
    fetch: bool = False,
) -> pd.DataFrame:
    """Real daily **total-return** OHLCV for ``ticker``; cache-first, network only on ``fetch``.

    Primary source is the shared total-return panel
    (``_cache/SPY_total_return.parquet``, yfinance ``auto_adjust=True``). On a cache miss
    it falls back to :func:`quantlab.data.fetch` with ``mode="total_return"``; only with
    ``fetch=True`` does it go to the network (and cache the result). Columns are
    lower-cased ``open/high/low/close/volume``; the index is a tz-naive ``date``.
    """
    cache_path = os.path.join(DEFAULT_CACHE, f"{ticker}_total_return.parquet")
    bars = None
    if use_cache and os.path.exists(cache_path):
        bars = pd.read_parquet(cache_path)
    elif not fetch:
        try:
            import sys
            sys.path.insert(0, REPO_ROOT)
            from quantlab import data as qdata

            raw = qdata.fetch(ticker, start=start, end=end, mode="total_return",
                              use_cache=use_cache)
            bars = raw.rename(columns=str.lower)
        except Exception as exc:  # graceful offline fallback
            raise FileNotFoundError(
                f"No cached total-return tape for {ticker} at {cache_path}, and the "
                f"quantlab fallback failed offline ({exc!r}). Call "
                f"load_real(fetch=True) once to populate the cache."
            ) from exc
    else:
        import yfinance as yf

        raw = yf.download(ticker, start=start, interval="1d", auto_adjust=True,
                          progress=False)
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw
        os.makedirs(DEFAULT_CACHE, exist_ok=True)
        bars.to_parquet(cache_path)

    bars = bars.rename(columns=str.lower)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in bars.columns]
    bars = bars[keep]
    bars.index = pd.DatetimeIndex(bars.index)
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    if end is not None:
        bars = bars.loc[:end]
    return bars.loc[start:]


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]

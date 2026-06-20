"""Data layer for Study 315 (Sovereign-Downgrade) — the *directional equity* angle.

Two tapes, one shape (a tz-naive daily total-return price frame). This study asks a
question about **stock returns** around the days the United States lost a sovereign
credit-rating notch — distinct from Study 312 (Debt-Ceiling), which works on the **VIX**
and the long-vol-into / short-vol-out *volatility* round-trip around the X-dates. Here the
event is the **rating action itself** (the agency's announcement), and the apparatus is a
classic abnormal-return event study with a **synthetic control** (the same window measured
around matched non-event dates).

- ``synthetic_spy`` — a *deterministic, offline* generator of a daily total-return price
  series (a drifting random walk) with an optional **planted event dip** on a list of
  "downgrade" sessions: a negative abnormal return concentrated on the announcement bar and
  the day after (the panic the folklore predicts), or nothing at all (the null). A test can
  then assert the event-study engine recovers a planted dip only when one is really there.

- ``load_real`` — the real daily SPY **total-return** tape, cache-first via the shared
  ``_cache/SPY_total_return.parquet`` panel, with a graceful ``quantlab.data`` fallback and,
  only on an explicit ``fetch=True``, a network pull. The test-suite and reproducible core
  never touch the network.

THE DOWNGRADE TABLE is hardcoded below: the three times a major agency cut the US
**long-term sovereign** rating one notch from the top. ``announce`` is the date the action
hit the tape (after the US cash close where the agency announced post-close, so the first
*tradeable* reaction is the next session — see ``strategy.snap_to_sessions(lag=1)``). The
canonical event bar is the **first full session after the announcement**.

No look-ahead is baked in here — the event-window bookkeeping lives in ``strategy.py``. The
announcement is calendar-*surprising* (not pre-scheduled like a debt-ceiling X-date), so the
canonical window enters at the first session that could actually trade on the news (lag=1);
a same-session (lag=0) variant is offered as a robustness check.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# The hardcoded US sovereign-downgrade table
# ---------------------------------------------------------------------------
# The three times a major agency cut the United States' long-term sovereign credit rating
# one notch below the top (AAA / Aaa). ``announce`` is the calendar date the action was
# published; all three were announced after the US equity close, so the first *tradeable*
# session is the next business day (the engine's canonical lag=1). The "2025 Moody-style"
# entry in the brief refers to Moody's May 2025 cut of the last AAA the US still held.
#
# Sources: S&P Global Ratings press release "United States of America Long-Term Rating
# Lowered To 'AA+'" (Aug 5, 2011); Fitch Ratings "Fitch Downgrades the United States'
# Long-Term Ratings to 'AA+' from 'AAA'" (Aug 1, 2023); Moody's Ratings "Moody's Ratings
# downgrades United States ratings to Aa1 from Aaa" (May 16, 2025). Contemporaneous
# Reuters/WSJ/FT wrap-ups corroborate the announcement timing (all post-close).
DOWNGRADES = pd.DataFrame(
    [
        # name,                       agency,    announce,      from_,  to_
        ("2011 S&P (AAA→AA+)",        "S&P",     "2011-08-05", "AAA",  "AA+"),
        ("2023 Fitch (AAA→AA+)",      "Fitch",   "2023-08-01", "AAA",  "AA+"),
        ("2025 Moody's (Aaa→Aa1)",    "Moody's", "2025-05-16", "Aaa",  "Aa1"),
    ],
    columns=["name", "agency", "announce", "from_", "to_"],
)
DOWNGRADES["announce"] = pd.to_datetime(DOWNGRADES["announce"])


def announcements() -> pd.DatetimeIndex:
    """The downgrade-announcement dates as a tz-naive DatetimeIndex."""
    return pd.DatetimeIndex(DOWNGRADES["announce"].to_numpy(), name="announce")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_spy(
    n_days: int = 8400,
    event_effect: float = 0.0,
    n_events: int = 40,
    daily_drift: float = 0.0003,
    daily_vol: float = 0.011,
    start: str = "1993-01-29",
    seed: int = 315,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict]:
    """A reproducible daily total-return price tape with optional planted event dips.

    Log returns are an i.i.d. normal ``N(daily_drift, daily_vol)`` random walk (an index
    total-return series, drift up, no autocorrelation). On ``n_events`` evenly-spaced
    "downgrade" sessions we subtract ``event_effect`` (a negative abnormal return) split
    across the announcement bar and the bar after it — the panic the folklore predicts.

    - ``event_effect = 0`` → a plain drifting walk; downgrade dates carry no information.
      This is the null: the engine must NOT find a significant abnormal return.
    - ``event_effect > 0`` → a real dip around each event (a drop of ``event_effect`` in
      total over the two announcement bars). This is the positive control.

    Returns ``(bars, events, truth)`` where ``bars`` carries OHLCV (``close`` is the
    total-return level), ``events`` is the DatetimeIndex of the planted event sessions, and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start=start, periods=n_days)

    log_ret = rng.normal(daily_drift, daily_vol, n_days)

    # Plant evenly-spaced events, leaving room for the window on both sides.
    pad = 60
    lo, hi = pad, n_days - pad
    if hi > lo and n_events > 0:
        event_locs = np.unique(np.linspace(lo, hi, n_events).round().astype(int))
        for e in event_locs:
            # The dip lands on the announcement bar and the day after (split 60/40).
            if event_effect != 0.0:
                log_ret[e] -= 0.6 * event_effect
                if e + 1 < n_days:
                    log_ret[e + 1] -= 0.4 * event_effect
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
        {
            "open": open_,
            "high": hi_px,
            "low": lo_px,
            "close": close,
            "volume": vol,
        },
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    events = (
        pd.DatetimeIndex(bars.index[event_locs], name="event")
        if event_locs.size
        else pd.DatetimeIndex([], name="event")
    )
    truth = {
        "event_effect": event_effect,
        "n_events": int(events.size),
        "daily_drift": daily_drift,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
    }
    return bars, events, truth


def synthetic_null_events(
    bars: pd.DataFrame,
    n_events: int = 200,
    window: int = 20,
    seed: int = 7,
) -> pd.DatetimeIndex:
    """Draw ``n_events`` random session dates from ``bars`` — the synthetic-control pool.

    These are *random* dates with no special meaning. The event study compares the abnormal
    return measured around the real downgrades to the distribution of the same statistic
    measured around these matched non-event dates — the synthetic control / placebo.
    """
    rng = np.random.default_rng(seed)
    lo, hi = window + 5, len(bars) - window - 5
    locs = rng.choice(np.arange(lo, hi), size=min(n_events, hi - lo), replace=False)
    return pd.DatetimeIndex(sorted(bars.index[locs]), name="null_event")


# ---------------------------------------------------------------------------
# Real tape — daily SPY total return, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str = "SPY") -> str:
    """Path to the shared cache parquet for ``ticker`` total-return bars."""
    safe = ticker.replace("/", "")
    return os.path.join(DEFAULT_CACHE, f"{safe}_total_return.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "1993-01-01",
    end: str | None = None,
    use_cache: bool = True,
    fetch: bool = False,
) -> pd.DataFrame:
    """Real daily total-return OHLCV for ``ticker``; cache-first, network only on ``fetch``.

    Primary source is the shared cache parquet (``_cache/SPY_total_return.parquet``). On a
    cache miss it falls back to :func:`quantlab.data.fetch` in ``total_return`` mode; only
    with ``fetch=True`` does it go to the network (and cache the result). Columns are
    lower-cased ``open/high/low/close/volume``; the index is a tz-naive ``date``.

    SPY total-return (dividends reinvested) is the right tape for a directional equity
    event study — we want the all-in shareholder reaction, not a price-only proxy.
    """
    cache_path = _cache_path(ticker)
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
                f"No cached tape for {ticker} at {cache_path}, and the quantlab "
                f"fallback failed offline ({exc!r}). Call load_real(fetch=True) once "
                f"to populate the cache."
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

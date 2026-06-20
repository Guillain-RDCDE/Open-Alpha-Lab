"""Data layer for Study 320 (Russell-Reconstitution).

Two tapes, one shape (a tz-naive daily OHLCV frame), plus a hardcoded table of annual
Russell reconstitution *effective* dates.

THE EVENT. Once a year FTSE Russell rebuilds its US indices (Russell 1000 / 2000 / 3000).
Membership is recomputed from a late-May "rank day" snapshot, the preliminary lists are
published in early-to-mid June, and the new membership becomes **effective after the close
of the reconstitution Friday** — traditionally the *last Friday of June* (FTSE Russell
shifts to the prior Friday when the last Friday would fall on the 29th/30th, to keep a
clean settlement runway). Because every index fund tracking these benchmarks must hold the
new weights *by that close*, an enormous, mechanical, one-direction buy/sell order lands in
the closing auction of that Friday — one of the highest-volume sessions of the US year.

THE FOLKLORE. The reconstitution date is known years ahead, the flow is forced and
one-directional, so (the story goes) you can *front-run* it: buy the small-cap index in the
days before reconstitution Friday to ride the index-fund buying pressure, then sell into or
just after the event. This study tests that on the small-cap proxy IWM (iShares Russell
2000) and, where available, IWO (iShares Russell 2000 Growth).

DISTINCT FROM STUDY 249 (Index-Inclusion). Study 249 is a *cross-section of single names*
being added to the S&P 500 (the "inclusion pop" on individual stocks). Study 320 is a
*calendar* event on the *whole index ETF* — there is no name-picking and no survivorship
panel; the trade is IWM/IWO itself on a date fixed by the FTSE Russell calendar.

  - ``synthetic_daily`` — a *deterministic, offline* generator. A ``premium_bps`` knob
    plants a pre-reconstitution drift into the run-up window only; ``premium_bps=0`` is the
    null (a pure drift+noise tape with no event effect). This is the positive control / null
    in a bottle, and it never backs a stamp.
  - ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
    the test-suite and reproducible core never touch the network.

No look-ahead: the reconstitution date is on the published FTSE Russell calendar years in
advance, so a "buy N sessions before, sell on the event close" rule needs *no* execution
lag (the rule is calendar-known, like a turn-of-month window). The discipline lives in
``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Hardcoded Russell reconstitution EFFECTIVE dates (the reconstitution Friday).
# The new membership is effective after the close of this session. Source: FTSE
# Russell US index reconstitution calendars / annual recon announcements. The rule
# of thumb is the last Friday of June, moved to the prior Friday when the last
# Friday is the 29th/30th (to leave a settlement runway), which is why a few years
# land on the 4th-Friday date.
# ---------------------------------------------------------------------------
_RECON_FRIDAYS = [
    "2000-06-30",
    "2001-06-29",  # last Friday was the 29th
    "2002-06-28",
    "2003-06-27",
    "2004-06-25",
    "2005-06-24",
    "2006-06-30",
    "2007-06-29",
    "2008-06-27",
    "2009-06-26",
    "2010-06-25",
    "2011-06-24",
    "2012-06-22",  # moved earlier (last Friday 29th)
    "2013-06-28",
    "2014-06-27",
    "2015-06-26",
    "2016-06-24",
    "2017-06-23",
    "2018-06-22",
    "2019-06-28",
    "2020-06-26",
    "2021-06-25",
    "2022-06-24",
    "2023-06-23",
    "2024-06-28",
    "2025-06-27",
]

RECON_DATES: list[pd.Timestamp] = [pd.Timestamp(d) for d in _RECON_FRIDAYS]


def recon_table() -> pd.DataFrame:
    """The reconstitution calendar as a DataFrame (year, effective_date, weekday)."""
    return pd.DataFrame(
        {
            "year": [d.year for d in RECON_DATES],
            "effective_date": RECON_DATES,
            "weekday": [d.weekday() for d in RECON_DATES],  # 4 = Friday
        }
    )


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    start_year: int = 2000,
    end_year: int = 2025,
    premium_bps: float = 0.0,
    runup_days: int = 5,
    drift_bps: float = 3.0,
    daily_vol: float = 0.013,
    seed: int = 320,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with an optional planted pre-reconstitution drift.

    Returns are i.i.d. normal with mean ``drift_bps`` and std ``daily_vol``, PLUS, on each
    of the ``runup_days`` sessions immediately *before* a reconstitution Friday, an extra
    ``premium_bps`` of daily return (the planted "front-run" edge).

    - ``premium_bps = 0`` → the null: no event effect, just drift + noise. A test asserts the
      engine finds *nothing* here.
    - ``premium_bps > 0`` → a plantable run-up edge the engine must detect (positive control).

    The calendar uses the real ``RECON_DATES`` Fridays for the requested years, and a plain
    business-day index for everything else. Returns ``(bars, truth)``.

    NOTE: we build the index with ``pd.bdate_range`` over a bounded span (≤ a few decades),
    never ``pd.date_range(periods=BIG, freq="M")`` — no ns-timestamp overflow risk.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{start_year}-01-03")
    end = pd.Timestamp(f"{end_year}-12-31")
    sessions = pd.bdate_range(start=start, end=end)
    n = len(sessions)

    drift = drift_bps * 1e-4
    prem = premium_bps * 1e-4
    log_ret = rng.normal(drift, daily_vol, n)

    # Which sessions fall in a [event - runup_days, event) run-up window?
    recon = [d for d in RECON_DATES if start_year <= d.year <= end_year]
    is_runup = np.zeros(n, dtype=bool)
    is_event = np.zeros(n, dtype=bool)
    pos = {ts: i for i, ts in enumerate(sessions)}
    for ev in recon:
        # snap the event to the nearest session at or before the Friday
        loc = sessions.searchsorted(ev, side="right") - 1
        if loc < 0 or loc >= n:
            continue
        is_event[loc] = True
        lo = max(0, loc - runup_days)
        is_runup[lo:loc] = True  # the run-up window EXCLUDES the event session itself

    log_ret[is_runup] += prem

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1] * np.exp(rng.normal(0.0, daily_vol * 0.3, n - 1))
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 60_000_000, n).astype(float)
    # event sessions carry the famous volume spike (decorative; not used by stats)
    vol[is_event] *= 4.0

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {
        "premium_bps": premium_bps,
        "runup_days": runup_days,
        "drift_bps": drift_bps,
        "daily_vol": daily_vol,
        "n_events": len(recon),
        "n_days": n,
        "seed": seed,
    }
    return bars, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_daily.parquet")


def fetch_daily(
    ticker: str = "IWM",
    start: str = "1999-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``). Default ticker IWM (iShares Russell 2000); IWO (Russell
    2000 Growth) is the secondary tape used for breadth.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]

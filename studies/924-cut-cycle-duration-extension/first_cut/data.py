"""Data layer for Study 924 — First Cut (buying duration at the first FOMC cut of a cycle).

Two tapes and one hardcoded calendar:

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the duration legs (TLT ≈ 20y+ Treasuries,
  IEF ≈ 7-10y), the front-end leg (SHY ≈ 1-3y), the cash leg (BIL, 1-3 month T-bills)
  and ``^IRX`` (the 13-week bill discount rate, used only to build a *proxy* cash
  accrual index for the pre-BIL era). ``fetch`` touches the network and caches parquet
  into the **shared** ``studies/_cache`` (retry up to 4×); ``load_prices`` reads that
  cache **offline** and never imports yfinance.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators. A
  duration tape, a front-end tape and a cash accrual leg, plus a handful of planted
  "first cut" event dates after which duration earns an extra drift. The
  ``signal_strength`` knob scales that planted post-event drift: at ``0`` there is no
  event effect at all (the null); at ``1`` the effect is large enough that the event
  study must find it. Seeds are fixed → the whole test-suite is deterministic.

**The event calendar is a hardcoded ASSUMPTION, not tape.** ``FIRST_CUTS`` is the list of
FOMC meetings/actions that began an easing cycle; ``ALL_CUTS`` is every target-rate cut.
Both are typed by hand from the Fed's published policy actions and are the study's single
non-market input — the README labels them as such, and the ALL-cuts control exists partly
to show how much of the story depends on that hand-picked "first" label.

No look-ahead is baked in here — the one execution lag lives in ``strategy.py``: the cut
is announced on day ``t``, the position is opened at the **close of day t+1**.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# TLT/IEF = the duration legs; SHY = the front end; BIL = cash (the excess-of-cash
# benchmark); ^IRX = the 13-week bill rate, a *proxy* cash accrual for the pre-BIL era.
TICKERS = ("TLT", "IEF", "SHY", "BIL", "^IRX")

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# The hardcoded FOMC calendar (ASSUMPTION — hand-typed, not market data)
# --------------------------------------------------------------------------- #
# The first target-rate cut of each distinct easing cycle. The 2020-03-03 inter-meeting
# emergency cut is listed as its own cycle start (the pandemic easing) even though the
# 2019 "mid-cycle adjustment" had ended only five months earlier — a judgement call that
# the ALL-cuts control deliberately sidesteps.
FIRST_CUTS = (
    "2001-01-03",   # inter-meeting, start of the dot-com easing
    "2007-09-18",   # start of the GFC easing
    "2019-07-31",   # the "mid-cycle adjustment"
    "2020-03-03",   # inter-meeting emergency cut (pandemic)
    "2024-09-18",   # start of the post-inflation easing
)

# Every FOMC target-rate cut. Truncated at 2024-12-18 on purpose: a 12-month forward
# window must close on or before the as-of (2026-06-30), so any cut after 2025-06-30
# could not be measured at the longest horizon anyway. There were no cuts between
# 2024-12-18 and 2025-06-30.
ALL_CUTS = (
    "2001-01-03", "2001-01-31", "2001-03-20", "2001-04-18", "2001-05-15",
    "2001-06-27", "2001-08-21", "2001-09-17", "2001-10-02", "2001-11-06",
    "2001-12-11",
    "2002-11-06",
    "2003-06-25",
    "2007-09-18", "2007-10-31", "2007-12-11",
    "2008-01-22", "2008-01-30", "2008-03-18", "2008-04-30", "2008-10-08",
    "2008-10-29", "2008-12-16",
    "2019-07-31", "2019-09-18", "2019-10-30",
    "2020-03-03", "2020-03-15",
    "2024-09-18", "2024-11-07", "2024-12-18",
)


def first_cuts() -> pd.DatetimeIndex:
    """The hardcoded first-cut-of-cycle dates as a DatetimeIndex."""
    return pd.DatetimeIndex(pd.to_datetime(list(FIRST_CUTS)), name="event")


def all_cuts() -> pd.DatetimeIndex:
    """The hardcoded full cut calendar (the control) as a DatetimeIndex."""
    return pd.DatetimeIndex(pd.to_datetime(list(ALL_CUTS)), name="event")


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "1999-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is dividend-adjusted total return — non-negotiable for bond ETFs,
    where the coupon is most of the holding-period return over a 12-month window.
    ``^IRX`` is a *rate* index, not a price, and is handled as such downstream.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    tk, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the shared cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily closes OFFLINE into one aligned frame, sliced to ``asof``.

    One column per ticker. TLT/IEF/SHY/BIL are total-return closes; ``^IRX`` is the
    13-week bill *discount rate in percent* and must be converted with
    :func:`cash_index_from_irx` before it can be compounded like a price.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call first_cut.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def cash_index_from_irx(irx: pd.Series) -> pd.Series:
    """Build a PROXY cash accrual index from ``^IRX`` (13-week bill discount rate, %).

    BIL only lists from 2007-05-30, which would silently drop the 2001 and (nearly) the
    2007 events from an excess-of-cash race. This proxy accrues the quoted 13-week
    discount rate daily, ``(1 + irx/100) ** (1/252)``, so the pre-BIL era gets a cash
    leg at all. It is a **PROXY**: a discount rate is not a bond-equivalent yield, there
    is no fund fee, and no reinvestment timing. It is used for cross-checks only; every
    headline number races against BIL's actual total return.
    """
    r = pd.Series(irx).astype(float).dropna() / 100.0
    r = r.clip(lower=0.0)
    daily = (1.0 + r) ** (1.0 / TRADING_DAYS_PER_YEAR)
    return daily.cumprod().rename("cash_proxy")


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted post-cut rally)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 24,
    n_events: int = 5,
    drift_dur_ann: float = 0.02,       # baseline annualised duration excess-of-cash drift
    vol_dur_ann: float = 0.14,         # annualised duration vol (TLT-ish)
    vol_front_ann: float = 0.02,       # annualised front-end vol (SHY-ish)
    cash_rate_ann: float = 0.025,
    event_drift_ann: float = 0.18,     # planted post-event excess drift, annualised
    event_days: int = 126,             # how long the planted effect lasts (~6 months)
    signal_strength: float = 1.0,      # 0 = pure null, 1 = full planted effect
    start: str = "2002-01-02",
    seed: int = 924,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict]:
    """A daily duration / front-end / cash tape with planted "first cut" events.

    ``signal_strength`` scales the planted post-event drift only:

    - ``1`` → for ``event_days`` after each event the duration leg earns an extra
      ``event_drift_ann`` annualised excess return; the event study must find it.
    - ``0`` → no event effect whatsoever (the null); the event study must stay quiet.

    Returns ``(prices, events, truth)`` where ``prices`` carries ``duration``, ``front``
    and ``cash`` total-return levels, ``events`` are the planted event dates, and
    ``truth`` records the planted parameters. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a few thousand business days stays far inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    # Space the events out so their windows do not overlap.
    margin = max(event_days + 30, 260)
    slots = np.linspace(margin, n_days - margin - 1, n_events).astype(int)
    event_idx = np.unique(slots)
    events = pd.DatetimeIndex(dates[event_idx], name="event")

    boost = np.zeros(n_days)
    for i in event_idx:
        j = min(i + 1 + event_days, n_days)
        boost[i + 1: j] += signal_strength * event_drift_ann / TRADING_DAYS_PER_YEAR

    d_dur = drift_dur_ann / TRADING_DAYS_PER_YEAR
    s_dur = vol_dur_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_front = vol_front_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    cash_daily_rate = cash_rate_ann / TRADING_DAYS_PER_YEAR

    shock = rng.normal(0.0, 1.0, n_days)
    r_cash = np.full(n_days, cash_daily_rate)
    r_dur = r_cash + d_dur + boost + s_dur * shock
    # The front end shares the duration shock but with ~1/6 the sensitivity.
    r_front = r_cash + 0.25 * d_dur + 0.15 * boost + s_front * shock + \
        0.2 * s_front * rng.normal(0.0, 1.0, n_days)

    prices = pd.DataFrame(
        {
            "duration": 100.0 * np.cumprod(1.0 + r_dur),
            "front": 100.0 * np.cumprod(1.0 + r_front),
            "cash": 100.0 * np.cumprod(1.0 + r_cash),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "event_drift_ann": event_drift_ann,
        "event_days": event_days,
        "n_events": int(len(events)),
        "n_days": n_days,
        "n_years": n_years,
        "seed": seed,
        "cash_rate_ann": cash_rate_ann,
        "planted_6m_excess": signal_strength * event_drift_ann * (event_days / TRADING_DAYS_PER_YEAR),
    }
    return prices, events, truth


def synthetic_panel(
    n_worlds: int = 8,
    signal_strength: float = 0.0,
    base_seed: int = 924,
    **kwargs,
) -> list[tuple[pd.DataFrame, pd.DatetimeIndex, dict]]:
    """``n_worlds`` independent synthetic tapes (seeds ``base_seed + k``).

    A panel of worlds is how the null is calibrated honestly: with only a handful of
    events per world, a single world's *t* is nearly meaningless, so the tests check the
    **distribution** of the event-study statistic across worlds instead.
    """
    return [
        synthetic_daily(signal_strength=signal_strength, seed=base_seed + k, **kwargs)
        for k in range(n_worlds)
    ]

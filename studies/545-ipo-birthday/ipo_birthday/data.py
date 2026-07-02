"""Data layer for Study 545 (IPO-Birthday) — the anniversary 'birthday effect'.

The folklore: a listed company gets an **attention bump** around the anniversary of its IPO each
year (press retrospectives, "one year on the market" coverage, index/analyst calendar effects), so
its stock should show a small positive **cumulative abnormal return (CAR)** in a window straddling
the anniversary date. This is the calendar/attention cousin of the birthday effect studied in
behavioural finance (Kliger & Qadan on Rosh Hashanah/Yom Kippur; the general attention-drives-return
literature of Barber & Odean, Da-Engelberg-Gao).

We measure it as an **event study**: for each firm-year, take the anniversary of its IPO date, build
an event window of trading days around it, and accumulate each name's return *net of a benchmark*
(SPY) — the market-adjusted abnormal return. The headline statistic is the mean CAR across all
firm-year events and its *t*-stat; the honest null is a **random-anchor placebo** (re-run the same
window on random calendar dates that are *not* IPO anniversaries).

Two tapes, one schema:

- ``synthetic_panel`` — a deterministic, offline generator. A single knob, ``bump_bps``, plants an
  **anniversary bump** (a small excess drift in the event window around each firm's anniversary);
  ``bump_bps = 0`` is the null (returns are pure benchmark + idiosyncratic noise, no calendar
  effect). The study's positive control and null in one bottle. Tests never touch the network.
- ``fetch_prices`` — the real yfinance tape (daily adjusted close for a curated IPO basket + SPY),
  cache-first into this study's own ``_cache/`` so the reproducible core never needs the network.

Curated basket: a hardcoded table of well-known US IPOs with their IPO dates (from public records).
Survivorship is named on the SIGNAL axis — the basket is names *still trading in 2026*, so failed
IPOs are absent; a birthday effect is a small symmetric window return, so survivorship is a milder
bias here than for a long-run drift study, but it is stated openly.

No look-ahead: each event's CAR is accumulated forward from the window start; the window is defined
purely by the (known, historical) IPO calendar date. The single documented convention is that the
event window is centred on the first trading day on/after the anniversary (the ``[-pre, +post]``
window in trading days).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Curated IPO table — notable US IPOs with their IPO (first-trade) dates.
# Sources: SEC S-1 filings / Nasdaq/NYSE listing records / Renaissance Capital.
# Only tickers still queryable on yfinance (survivorship NAMED on the Signal axis).
# We favour names with a long post-IPO history so many firm-year anniversaries exist.
# ---------------------------------------------------------------------------
IPO_TABLE: list[dict] = [
    {"ticker": "AMZN", "ipo_date": "1997-05-15"},
    {"ticker": "EBAY", "ipo_date": "1998-09-24"},
    {"ticker": "NFLX", "ipo_date": "2002-05-23"},
    {"ticker": "GOOGL", "ipo_date": "2004-08-19"},
    {"ticker": "CRM", "ipo_date": "2004-06-23"},
    {"ticker": "MA", "ipo_date": "2006-05-25"},
    {"ticker": "V", "ipo_date": "2008-03-19"},
    {"ticker": "TSLA", "ipo_date": "2010-06-29"},
    {"ticker": "LNKD", "ipo_date": "2011-05-19"},   # LinkedIn (delisted 2016; yf may still serve)
    {"ticker": "META", "ipo_date": "2012-05-18"},   # Facebook
    {"ticker": "BABA", "ipo_date": "2014-09-19"},
    {"ticker": "SHOP", "ipo_date": "2015-05-21"},
    {"ticker": "SQ", "ipo_date": "2015-11-19"},     # Block
    {"ticker": "SNAP", "ipo_date": "2017-03-02"},
    {"ticker": "SPOT", "ipo_date": "2018-04-03"},
    {"ticker": "DDOG", "ipo_date": "2019-09-19"},
    {"ticker": "UBER", "ipo_date": "2019-05-10"},
    {"ticker": "LYFT", "ipo_date": "2019-03-29"},
    {"ticker": "ZM", "ipo_date": "2019-04-18"},     # Zoom
    {"ticker": "SNOW", "ipo_date": "2020-09-16"},
    {"ticker": "PLTR", "ipo_date": "2020-09-30"},
    {"ticker": "ABNB", "ipo_date": "2020-12-10"},
    {"ticker": "DASH", "ipo_date": "2020-12-09"},
    {"ticker": "COIN", "ipo_date": "2021-04-14"},
    {"ticker": "RBLX", "ipo_date": "2021-03-10"},
    {"ticker": "HOOD", "ipo_date": "2021-07-29"},
    {"ticker": "RIVN", "ipo_date": "2021-11-10"},
    {"ticker": "SOFI", "ipo_date": "2021-06-01"},
    {"ticker": "ARM", "ipo_date": "2023-09-14"},
    {"ticker": "CART", "ipo_date": "2023-09-19"},   # Instacart
]

# Tickers renamed since IPO (yfinance symbol).
TICKER_ALIAS = {"SQ": "XYZ"}

BENCHMARK = "SPY"


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic panel."""

    bump_bps: float  # excess return (bps, total over the window) planted around each anniversary

    @property
    def has_effect(self) -> bool:
        return self.bump_bps != 0.0


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 30,
    years: int = 8,
    bump_bps: float = 120.0,
    pre: int = 5,
    post: int = 5,
    idio_vol_ann: float = 0.45,
    seed: int = 545,
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp], WorldTruth]:
    """A reproducible **wide price panel** with a tunable IPO-anniversary bump.

    Builds ``n_stocks`` synthetic firms each with a random IPO date and ``years`` of daily returns.
    Every firm's return is ``alpha + beta*benchmark + idio_noise``; when ``bump_bps != 0`` an extra
    excess return of ``bump_bps`` (total, spread across the ``[-pre, +post]`` trading-day event
    window) is added *only* in the days around each anniversary. A benchmark (SPY-like) column is
    included so the strategy computes market-adjusted abnormal returns exactly as on the real tape.

    ``bump_bps = 0`` is the null (no calendar effect). Returns ``(prices, ipo_dates, truth)`` where
    ``prices`` is a wide frame (index=date, columns=tickers + ``BENCHMARK``, values=price levels)
    and ``ipo_dates`` maps each synthetic ticker to its planted IPO date.

    Deterministic in ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = years * TRADING_DAYS
    dates = pd.bdate_range("2014-01-02", periods=n_days, name="date")

    mkt_daily = 0.0003  # benchmark drift ~7.8%/yr
    mkt_vol = 0.010
    bench_ret = mkt_daily + mkt_vol * rng.standard_normal(n_days)

    win = pre + post + 1
    per_day_bump = (bump_bps * 1e-4) / win  # spread the total bump across the window days

    cols: dict[str, np.ndarray] = {BENCHMARK: 100.0 * np.cumprod(1.0 + bench_ret)}
    ipo_map: dict[str, pd.Timestamp] = {}

    for j in range(n_stocks):
        beta = 0.8 + 0.6 * rng.random()
        idio = (idio_vol_ann / np.sqrt(TRADING_DAYS)) * rng.standard_normal(n_days)
        alpha = -0.00002 + 0.00004 * rng.random()
        ret = alpha + beta * bench_ret + idio

        # IPO date somewhere in the first year so many anniversaries exist
        ipo_offset = int(rng.integers(0, TRADING_DAYS))
        ipo_date = dates[ipo_offset]
        ipo_map[f"S{j:03d}"] = ipo_date

        # plant the bump in each anniversary window
        if bump_bps != 0.0:
            for yr in range(1, years + 1):
                anniv = ipo_date + pd.DateOffset(years=yr)
                pos = int(np.searchsorted(dates.values, np.datetime64(anniv)))
                lo, hi = pos - pre, pos + post
                if lo >= 0 and hi < n_days:
                    ret[lo : hi + 1] += per_day_bump

        cols[f"S{j:03d}"] = 100.0 * np.cumprod(1.0 + ret)

    prices = pd.DataFrame(cols, index=dates)
    return prices, ipo_map, WorldTruth(bump_bps=bump_bps)


# ---------------------------------------------------------------------------
# Real tape — yfinance daily prices, cache-first
# ---------------------------------------------------------------------------
def _price_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "ipo_birthday_prices.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def ipo_dates() -> dict[str, pd.Timestamp]:
    """Curated {yfinance-ticker: IPO date} for the real basket (aliases applied)."""
    out = {}
    for r in IPO_TABLE:
        tk = TICKER_ALIAS.get(r["ticker"], r["ticker"])
        out[tk] = pd.Timestamp(r["ipo_date"])
    return out


def fetch_prices(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "1997-01-01",
    end: str = "2026-06-27",
) -> pd.DataFrame:
    """Daily adjusted-close prices for the curated IPO basket + SPY, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    frame. Network is touched only on an explicit ``fetch=True`` (then cached). Returns a wide
    frame indexed by date, one column per ticker (adjusted close).
    """
    path = _price_cache()
    if os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    tickers = list(ipo_dates().keys()) + [BENCHMARK]
    raw = _retry(
        lambda: yf.download(
            tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True
        )["Close"]
    )
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw = raw.dropna(how="all")
    os.makedirs(cache_dir, exist_ok=True)
    raw.to_parquet(path)
    return raw


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0.0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        import json

        blob = json.dumps({k: str(v) for k, v in sorted(obj.items())}).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]

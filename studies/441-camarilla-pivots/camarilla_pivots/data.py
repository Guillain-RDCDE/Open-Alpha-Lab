"""Data layer for Study 441 — Camarilla Pivots.

Two tapes, one shape: a tz-naive **intraday** OHLC frame with a ``session`` (calendar date)
column so we can group bars into trading days and attach each day's prior-day Camarilla ladder.

* **Real tape.** yfinance intraday bars (``5m`` ~60 calendar days, ``1m`` ~7 days) for a small
  set of very liquid names (SPY/QQQ + a couple of single names). Cache-first into ``_cache/``
  as parquet so the reproducible core and notebooks run offline once populated. The **short
  span** of intraday history (Yahoo caps 5m at ~60d, 1m at ~7d) is the loud caveat — there are
  only ~40 sessions per name, so the power budget is thin and the verdict must lean on pooling.

* **Synthetic.** A deterministic, fixed-seed intraday generator with a **planted edge knob**
  (``respect``): at ``respect=0`` the path is a pure random walk inside the day (a level is just
  a price the tape happens to cross — no special behaviour); at ``respect>0`` we plant genuine
  mean-reversion *toward the central pivot whenever price touches an outer Camarilla level* —
  exactly the behaviour the folklore claims. It is the positive control: zero edge must NOT
  manufacture significance; a large planted edge must light up.

Pure numpy + pandas + stdlib on the offline path. ``fetch_intraday`` (network) is only used to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A small, transparent set of the most liquid intraday names (deep books, tight spreads — the
# *best case* for an intraday-level edge to survive costs).
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


# --------------------------------------------------------------------------- #
# Camarilla ladder (shared identity — also used by strategy.py)
# --------------------------------------------------------------------------- #
def camarilla_levels(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """The Camarilla pivot ladder from a prior session's H/L/C.

    Range ``R = high - low``. The classic Camarilla multipliers build four levels on each side
    of the close:

        H4 = close + R * 1.1/2      L4 = close - R * 1.1/2     (breakout)
        H3 = close + R * 1.1/4      L3 = close - R * 1.1/4     (reversal — the headline levels)
        H2 = close + R * 1.1/6      L2 = close - R * 1.1/6
        H1 = close + R * 1.1/12     L1 = close - R * 1.1/12

    plus the central pivot P = (high + low + close) / 3 (the classic pivot, the reversion
    target). Returns a dict of all nine levels.
    """
    R = prev_high - prev_low
    return {
        "P": (prev_high + prev_low + prev_close) / 3.0,
        "H4": prev_close + R * 1.1 / 2.0,
        "H3": prev_close + R * 1.1 / 4.0,
        "H2": prev_close + R * 1.1 / 6.0,
        "H1": prev_close + R * 1.1 / 12.0,
        "L1": prev_close - R * 1.1 / 12.0,
        "L2": prev_close - R * 1.1 / 6.0,
        "L3": prev_close - R * 1.1 / 4.0,
        "L4": prev_close - R * 1.1 / 2.0,
    }


# --------------------------------------------------------------------------- #
# Real tape — yfinance intraday, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, interval: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_{interval}.parquet")


def _tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise a yfinance intraday frame: tz-naive index, lower-case OHLC, session col."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    bars = raw.rename(columns=str.lower)[["open", "high", "low", "close"]].copy()
    if bars.index.tz is not None:
        # convert to US/Eastern wall clock then drop tz so sessions group by NY trading day
        bars.index = bars.index.tz_convert("America/New_York").tz_localize(None)
    bars.index.name = "ts"
    bars["session"] = bars.index.normalize()
    return bars.dropna()


def fetch_intraday(ticker: str, interval: str = "5m", period: str = "60d",
                   fetch: bool = True, cache_dir: str = DEFAULT_CACHE,
                   retries: int = 3) -> pd.DataFrame:
    """Real intraday OHLC for ``ticker``; cache-first.

    On a cache miss (or ``fetch=True``) hits yfinance with a small backoff, then caches the
    parquet so re-runs are offline. ``interval='5m'`` allows ``period`` up to ~60d; ``'1m'`` up
    to ~7d (Yahoo's hard caps). Network is touched only here.
    """
    path = _cache_path(ticker, interval, cache_dir)
    if not fetch and os.path.exists(path):
        return pd.read_parquet(path)
    if os.path.exists(path) and not fetch:
        return pd.read_parquet(path)

    import yfinance as yf  # lazy

    last_err = None
    for attempt in range(retries):
        try:
            raw = yf.download(ticker, interval=interval, period=period,
                              auto_adjust=False, progress=False)
            if raw is not None and not raw.empty:
                bars = _tidy(raw)
                os.makedirs(cache_dir, exist_ok=True)
                bars.to_parquet(path)
                return bars
        except Exception as e:  # pragma: no cover - network path
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    if os.path.exists(path):
        return pd.read_parquet(path)
    raise RuntimeError(f"yfinance returned no {interval} bars for {ticker}: {last_err}")


def have_real(tickers=TICKERS, interval: str = "5m",
              cache_dir: str = DEFAULT_CACHE) -> bool:
    return all(os.path.exists(_cache_path(t, interval, cache_dir)) for t in tickers)


def load_real(cache_dir: str = DEFAULT_CACHE, tickers=TICKERS,
              interval: str = "5m") -> dict:
    """Cached intraday frames keyed by ticker (offline). Skips any name without a cache file."""
    out = {}
    for t in tickers:
        p = _cache_path(t, interval, cache_dir)
        if os.path.exists(p):
            out[t] = pd.read_parquet(p)
    return out


def fingerprint(frames) -> str:
    """Short sha1 content fingerprint of one or many intraday close series, for the as-of stamp."""
    h = hashlib.sha1()
    if isinstance(frames, dict):
        items = sorted(frames.items())
        for k, v in items:
            h.update(k.encode())
            h.update(np.ascontiguousarray(v["close"].to_numpy(dtype=float)).tobytes())
    else:
        h.update(np.ascontiguousarray(frames["close"].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_panel(n_sessions: int = 40, bars_per_day: int = 78, respect: float = 0.0,
                    seed: int = 441, sigma_bar: float = 0.0009,
                    overnight_sd: float = 0.006) -> tuple[pd.DataFrame, dict]:
    """Deterministic intraday OHLC panel with a planted *respect-the-level* knob.

    Each session is a random walk of ``bars_per_day`` log-return bars (5-minute-ish: 78 bars =
    a 6.5-hour RTH session). The prior session's H/L/C set today's Camarilla ladder. If
    ``respect`` > 0, whenever the running intraday price sits **beyond an outer level**
    (above H3 or below L3) we add a restoring drift of ``respect`` × (distance past the level),
    i.e. price is genuinely pulled **back toward the pivot** after touching L3/H3 — the exact
    behaviour the folklore claims. With ``respect = 0`` the path is a pure random walk: a level
    is just a price the tape crosses, and the touch→reversion test must NOT manufacture an edge.

    Returns ``(bars, truth)`` where ``bars`` has columns open/high/low/close + ``session`` and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    rows_ts, rows_o, rows_h, rows_l, rows_c, rows_s = [], [], [], [], [], []
    sessions = pd.bdate_range("2024-01-02", periods=n_sessions)

    log_p = np.log(100.0)
    prev_h = prev_l = prev_c = None
    for d in sessions:
        # overnight gap
        log_p += rng.normal(0.0, overnight_sd)
        # today's ladder from yesterday's H/L/C (after the first day)
        lev = None
        if prev_c is not None:
            lev = camarilla_levels(prev_h, prev_l, prev_c)
        day_hi = day_lo = day_close = None
        ts0 = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=30)
        day_o = None
        for b in range(bars_per_day):
            p0 = np.exp(log_p)
            drift = 0.0
            if lev is not None and respect != 0.0:
                if p0 > lev["H3"]:
                    drift = -respect * (p0 - lev["H3"]) / p0
                elif p0 < lev["L3"]:
                    drift = respect * (lev["L3"] - p0) / p0
            log_p += drift + rng.normal(0.0, sigma_bar)
            c = np.exp(log_p)
            o = p0
            wick = abs(rng.normal(0.0, sigma_bar)) * c
            hi = max(o, c) + wick
            lo = min(o, c) - wick
            ts = ts0 + pd.Timedelta(minutes=5 * b)
            rows_ts.append(ts); rows_o.append(o); rows_h.append(hi)
            rows_l.append(lo); rows_c.append(c); rows_s.append(pd.Timestamp(d))
            if day_o is None:
                day_o = o
            day_hi = hi if day_hi is None else max(day_hi, hi)
            day_lo = lo if day_lo is None else min(day_lo, lo)
            day_close = c
        prev_h, prev_l, prev_c = day_hi, day_lo, day_close

    bars = pd.DataFrame({"open": rows_o, "high": rows_h, "low": rows_l,
                         "close": rows_c, "session": rows_s},
                        index=pd.DatetimeIndex(rows_ts, name="ts"))
    truth = {"respect": respect, "n_sessions": n_sessions,
             "bars_per_day": bars_per_day, "seed": seed}
    return bars, truth

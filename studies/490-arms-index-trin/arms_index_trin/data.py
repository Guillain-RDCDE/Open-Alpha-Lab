"""Data layer for Study 490 (Arms Index / TRIN).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The TRIN rule fires on *high-TRIN panic days* and bets on a next-day bounce. We plant exactly
  that: with ``edge > 0`` the basket gets occasional synchronised down-days (a panic that drives
  TRIN high) that are *followed* by a real upward bounce on SPY; with ``edge = 0`` every series
  is an independent random walk, so a high-TRIN day carries no information and the entry is a
  fair coin. This is the positive control — a harness that cannot bank the planted bounce proves
  nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a cached
  parquet if present and only touches the network on an explicit cache miss (short back-off +
  retry), then caches the parquet so re-runs are offline.

**Breadth caveat.** TRIN is properly built from *exchange-wide* advancing/declining issue counts
and their volumes. That feed is not available offline, so we build a **breadth proxy** from a
small basket of liquid ETFs: each ETF is treated as one "issue"; it *advances* if its daily
return is positive; the *move magnitude* |return| stands in for issue volume. This is a coarse
proxy and it caps the test — but the Signal question is unchanged: does the breadth signal beat
**random-day** entries on SPY?

No look-ahead is baked in here — that discipline lives in ``strategy.py``: TRIN is read on the
close of *t*, and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The 5 liquid ETFs reused from the desk's standard cache so the study runs fully offline.
# They double as the breadth-proxy basket (one "issue" each); SPY is also the traded instrument.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

# A wider sector-ETF basket for a richer breadth proxy *when a network fetch is allowed*. These
# are cached on first fetch and then served offline; if absent and offline, the loaders fall back
# to DEFAULT_TICKERS so the gate/CI never need the network.
SECTOR_BASKET = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "SPY"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    n_names: int = 5,
    start: str = "2010-01-04",
    seed: int = 490,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """A reproducible daily OHLC **basket** with a *known* amount of TRIN-bounce structure.

    Returns ``(panel, truth)`` where ``panel`` maps a name -> OHLC frame (the first name, ``A``,
    is the traded instrument, the SPY stand-in). Each name's price path is a random walk in log
    returns with daily sigma ``annual_vol/sqrt(252)``. With ``edge > 0`` we occasionally inject a
    synchronised **panic day** (every name dumps together on heavy magnitude -> TRIN spikes high)
    and then add a real **bounce** to name ``A`` on the *following* day, proportional to ``edge``.
    At ``edge = 0`` the names are independent walks and a high-TRIN day is a fair coin.

    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)
    names = [chr(ord("A") + i) for i in range(n_names)]

    # zero-mean returns so that at edge=0 a high-TRIN day carries NO directional information
    rets = rng.normal(0.0, daily_vol, size=(n_days, n_names))

    if edge > 0.0:
        # schedule synchronised panic days (deterministic given the seed). The panic drives TRIN
        # high on day i; we enter at close i+1; so the planted bounce must land on i+2..i+2+H to
        # be bankable (forward_returns measures p[i+1+h]/p[i+1]).
        panic = rng.random(n_days) < 0.04
        for i in range(n_days):
            if panic[i] and i + 3 < n_days:
                shock = -abs(rng.normal(2.5 * daily_vol, 0.5 * daily_vol))
                rets[i, :] += shock                       # every name dumps together -> TRIN high
                rets[i + 2, 0] += edge * abs(shock)       # name A bounces AFTER entry (day i+2)
                rets[i + 3, 0] += 0.5 * edge * abs(shock)

    closes = 100.0 * np.exp(np.cumsum(rets, axis=0))
    panel: dict[str, pd.DataFrame] = {}
    for j, nm in enumerate(names):
        cl = closes[:, j]
        op = np.empty_like(cl)
        op[0] = 100.0
        op[1:] = cl[:-1]
        wick = np.abs(rng.normal(0.0, daily_vol * 0.5, cl.size)) * cl
        hi = np.maximum(op, cl) + wick
        lo = np.minimum(op, cl) - wick
        panel[nm] = pd.DataFrame(
            {"open": op, "high": hi, "low": lo, "close": cl},
            index=pd.DatetimeIndex(sessions, name="date"),
        )
    truth = {"edge": edge, "annual_vol": annual_vol, "n_names": n_names,
             "n_days": n_days, "seed": seed, "traded": names[0]}
    return panel, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def load_real(
    ticker: str = "SPY",
    start: str = "2005-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = True,
) -> pd.DataFrame:
    """Real daily OHLC for ``ticker``; **cache-first** (network only on a cache miss).

    Reads a cached parquet if present. Otherwise — and only if ``allow_fetch`` — downloads from
    yfinance (with a couple of retries + back-off on rate limits) and caches the parquet, so every
    subsequent call is fully offline.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
    elif allow_fetch:
        bars = _download(ticker, start, end)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)
    else:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}) once (network) to populate the cache."
        )

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars[["open", "high", "low", "close"]]


def _download(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf  # lazy: only on a real cache miss

    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=start, end=end, interval="1d",
                              auto_adjust=True, progress=False)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                bars = raw.rename(columns=str.lower)[["open", "high", "low", "close"]]
                bars.index.name = "date"
                return bars
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yfinance returned no daily bars for {ticker}: {last_err}")


def load_basket(
    tickers: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load every basket member, cache-first. Members missing from the cache are skipped (when
    ``allow_fetch`` is False) so the breadth proxy degrades gracefully to whatever is cached.

    Returns a name -> OHLC frame dict. Falls back to the cached DEFAULT_TICKERS when *none* of the
    requested members are available (keeps the gate/CI fully offline).
    """
    tickers = tickers or DEFAULT_TICKERS
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        if os.path.exists(_cache_path(t, cache_dir)) or allow_fetch:
            try:
                out[t] = load_real(t, cache_dir=cache_dir, allow_fetch=allow_fetch)
            except Exception:  # noqa: BLE001
                continue
    if not out:
        for t in DEFAULT_TICKERS:
            if os.path.exists(_cache_path(t, cache_dir)):
                out[t] = load_real(t, cache_dir=cache_dir, allow_fetch=False)
    return out


def have_real(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every cached parquet for ``tickers`` is present (offline-safe check)."""
    tickers = tickers or DEFAULT_TICKERS
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]

"""Data layer for Study 326 (Stablecoin-Depeg).

The question: when a stablecoin **breaks the buck** — the algorithmic UST/Luna death
spiral of May-2022, or the SVB-weekend USDC depeg of March-2023 — is that a crypto-wide
*sell* signal? We answer it as an **event study**: the abnormal return of a crypto basket
(BTC, ETH) in a window around each depeg date, benchmarked against a **synthetic control**
of randomly-placed non-event windows on the same tape.

Three tapes, all offline for the reproducible core:

- ``DEPEG_EVENTS`` — a tiny hardcoded table of the two undisputed, market-wide depeg
  events. This is the universe of *real* events, and its size (n = 2) is the whole story:
  no honest inference clears the bar off two observations. Two near-misses (Tether's
  Oct-2022 wobble, the FTX-week USDT scare) are listed but flagged ``major=False`` so a
  reader can see we did not cherry-pick the count.

- ``synthetic_tape`` — a deterministic, offline daily crypto-return generator with a
  *planted* depeg ``shock`` knob: on each planted event day the basket takes an extra
  negative jump that decays over the following days. ``shock = 0`` is the null — depegs
  fall on ordinary days and carry no abnormal return. This is the positive control / null
  in a bottle, and it is the **only** tape allowed to back the engine's machinery (never a
  Signal stamp).

- ``load_real`` — the real Yahoo! BTC-USD / ETH-USD daily close tapes (via ``yfinance``),
  cache-only by default (``_cache/<TICKER>_daily.parquet``) so the test-suite and the
  reproducible core never touch the network. Graceful fallback: a missing cache raises
  ``FileNotFoundError`` and callers guard with ``os.path.exists`` (CI is offline).

No look-ahead. A depeg is a *dated public event*; the abnormal-return window is centred on
the known event day and needs no execution lag. The **tradable** "sell on the depeg" rule
in ``strategy.py`` does carry exactly one documented lag (act at the close of day t+1,
since the depeg is only confirmed intraday on day t).

Adjustment mode: BTC/ETH spot has no dividend, so price-only == total-return here.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# Tickers of the crypto basket (Yahoo spot, USD).
TICKERS = ("BTC-USD", "ETH-USD")


# ---------------------------------------------------------------------------
# The real depeg events — the deterministic, offline core (and the whole point)
# ---------------------------------------------------------------------------
# Each entry: date the peg break became undeniable + a one-line label. ``major``
# flags the two market-wide breaks everyone agrees on; the False rows are the
# near-misses we *deliberately* exclude from the headline count (showing the count
# is a decision, not a cherry-pick). Dates are the day the depeg was the market's
# top story; the event window in strategy.py brackets them.
DEPEG_EVENTS: list[dict] = [
    {"date": "2022-05-09", "coin": "UST",  "label": "UST/Luna death spiral (algorithmic)", "major": True},
    {"date": "2023-03-11", "coin": "USDC", "label": "USDC SVB-weekend depeg to $0.88",     "major": True},
    # Near-misses, EXCLUDED from the headline n=2 (shown for honesty about the count):
    {"date": "2022-10-12", "coin": "USDT", "label": "USDT wobble to ~$0.997 (intraday)",   "major": False},
    {"date": "2022-11-10", "coin": "USDT", "label": "FTX-week USDT scare to ~$0.985",       "major": False},
]


def depeg_events(major_only: bool = True) -> pd.DatetimeIndex:
    """Return the depeg event dates as a ``DatetimeIndex`` (tz-naive, sorted).

    ``major_only=True`` (the default and the headline) returns just the two
    market-wide breaks (UST, USDC); ``False`` adds the two near-misses.
    """
    rows = [e for e in DEPEG_EVENTS if (e["major"] or not major_only)]
    return pd.DatetimeIndex(sorted(pd.Timestamp(e["date"]) for e in rows))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (positive control / null)
# ---------------------------------------------------------------------------
def synthetic_tape(
    n_days: int = 1500,
    n_events: int = 8,
    shock: float = -0.10,
    decay: float = 0.5,
    daily_vol: float = 0.04,
    base_drift: float = 0.0005,
    start: str = "2015-01-02",
    seed: int = 326,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict]:
    """A reproducible daily crypto-basket tape with planted depeg shocks.

    Daily simple returns are ``base_drift + eps_t`` with ``eps_t`` i.i.d. normal of
    standard deviation ``daily_vol`` (crypto-like). On each of ``n_events`` planted event
    days the basket takes an **extra** return of ``shock`` (e.g. -0.10 = a 10% crash on the
    day), and the next days inherit a geometrically decaying echo ``shock * decay**k``.

    - ``shock = 0`` → events fall on ordinary days; there is **no** abnormal return (the
      null). The engine must read ~zero here.
    - ``shock < 0`` → a genuine crypto-wide sell-off planted on each event (the positive
      control). The engine must recover it.

    Returns ``(bars, event_dates, truth)``. ``bars`` is a daily OHLC-ish frame with a
    ``ret`` column (the basket's simple daily return) and a ``close`` index level; events
    are spaced so their windows do not overlap.
    """
    rng = np.random.default_rng(seed)
    # Decorative business-day index — bdate_range keeps the span tiny (no ns overflow).
    sessions = pd.bdate_range(start=start, periods=n_days)

    ret = base_drift + rng.normal(0.0, daily_vol, n_days)

    # Place events on an evenly-spaced, non-overlapping grid (deterministic),
    # leaving a margin at both ends so a ±window fits.
    margin = 40
    if n_events > 0:
        locs = np.linspace(margin, n_days - margin - 1, n_events).round().astype(int)
        locs = np.unique(locs)
    else:
        locs = np.array([], dtype=int)

    for loc in locs:
        k = 0
        while loc + k < n_days and abs(shock) * (decay ** k) > 1e-4:
            ret[loc + k] += shock * (decay ** k)
            k += 1

    close = 100.0 * np.cumprod(1.0 + ret)
    bars = pd.DataFrame(
        {"close": close, "ret": ret},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    event_dates = pd.DatetimeIndex(bars.index[locs]) if len(locs) else pd.DatetimeIndex([])
    truth = {
        "n_days": n_days,
        "n_events": int(len(locs)),
        "shock": shock,
        "decay": decay,
        "daily_vol": daily_vol,
        "base_drift": base_drift,
        "seed": seed,
    }
    return bars, event_dates, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily closes, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str = DEFAULT_CACHE) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "").replace("-", "")
    return os.path.join(cache_dir, f"{safe}_daily.parquet")


def load_real(
    ticker: str,
    start: str = "2017-01-01",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily close + simple return for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the pull is cached as a
    parquet under ``_cache/``). Returns a tz-naive daily frame with ``close`` and ``ret``
    (simple daily return). BTC/ETH spot has no dividend: price-only == total-return.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False`` — callers
    guard with ``os.path.exists(_cache_path(ticker))`` (the offline-CI rule).
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: network only on fetch=True

        raw = yf.download(
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw.rename(columns=str.lower)["close"].dropna()
        df = pd.DataFrame({"close": close})
        df["ret"] = df["close"].pct_change()
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    if "ret" not in df.columns:
        df["ret"] = df["close"].pct_change()
    return df


def have_real_cache(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's cache parquet exists (the offline-CI guard)."""
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def real_basket(
    tickers=TICKERS, start: str = "2017-01-01", cache_dir: str = DEFAULT_CACHE
) -> pd.DataFrame:
    """Equal-weight daily return of the crypto basket from cached real tapes.

    Returns a frame with ``ret`` (equal-weight mean daily return across ``tickers``)
    and ``close`` (a synthetic basket level, base 100). Cache-only; the caller must have
    checked ``have_real_cache()`` first.
    """
    rets = []
    for t in tickers:
        df = load_real(t, start=start, cache_dir=cache_dir)
        rets.append(df["ret"].rename(t))
    panel = pd.concat(rets, axis=1).dropna(how="all")
    basket = panel.mean(axis=1).rename("ret")
    out = pd.DataFrame({"ret": basket})
    out["close"] = 100.0 * (1.0 + out["ret"].fillna(0.0)).cumprod()
    out.index.name = "date"
    return out


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(obj) -> str:
    """A short content fingerprint of a Series/DataFrame return column, for the as-of."""
    if isinstance(obj, pd.DataFrame):
        col = obj["ret"] if "ret" in obj.columns else obj.iloc[:, 0]
        arr = col.to_numpy(dtype=float)
    else:
        arr = pd.Series(obj).to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

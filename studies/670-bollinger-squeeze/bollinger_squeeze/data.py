"""Data layer for Study 670 — Bollinger-Squeeze.

Two tapes, one shape (a tz-naive daily OHLCV frame, calendar-date indexed):

- ``fetch_daily`` — the real Yahoo! daily tape (``yfinance``), cache-only by default so
  the reproducible core never touches the network once the cache exists. Adjusted
  (total-return) OHLC, ``auto_adjust=True`` — split/dividend events never fake a band
  pierce.

- ``synthetic_daily`` — a deterministic, offline generator. A mean-reverting log-vol
  regime alternates "quiet" (low daily vol, band-contraction) and "loud" (high daily
  vol) episodes; a TUNABLE knob ``continuation`` controls how much the FIRST loud-regime
  day's sign persists into the next few days (the planted directional-continuation
  effect the TTM Squeeze claims to exploit). ``continuation = 0`` is the null world —
  the sign of the breakout day carries no forward information; the Welch detector must
  NOT manufacture significance from it.

Basket: **SPY, QQQ, IWM, DIA, GLD** — the same five liquid, 20+-year ETF tapes used by
sibling studies 128-keltner-channel / 485-starc-bands, so results line up across the
family.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

BASKET = ["SPY", "QQQ", "IWM", "DIA", "GLD"]
START = "2005-01-01"
AS_OF = "2026-06-30"          # last complete calendar month at publication (2026-07-10)


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"bars_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    start: str = START,
    end: str | None = None,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached
    as a parquet under ``_cache/``). Adjusted OHLC (``auto_adjust=True``) — total-return
    consistent bands, no split-driven fake pierces.
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
            ticker, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False,
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
    bars.index.name = "date"
    return bars


def have_cache(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    tickers = tickers or BASKET
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def fetch_all(tickers: list[str] | None = None, start: str = START, end: str | None = None,
              cache_dir: str = DEFAULT_CACHE) -> None:
    """Fetch + cache every ticker in the basket. Network; run once."""
    for t in tickers or BASKET:
        fetch_daily(t, start=start, end=end, fetch=True, cache_dir=cache_dir)


def load_basket(tickers: list[str] | None = None, asof: str = AS_OF,
                 cache_dir: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Cached daily bars for the basket, sliced to [START, asof]."""
    out = {}
    for t in tickers or BASKET:
        bars = fetch_daily(t, cache_dir=cache_dir)
        out[t] = bars.loc[bars.index <= pd.Timestamp(asof)].copy()
    return out


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column)."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]


def basket_fingerprint(baskets: dict[str, pd.DataFrame]) -> str:
    """One fingerprint over the whole basket (ticker order matters, sorted for stability)."""
    h = hashlib.sha1()
    for t in sorted(baskets):
        h.update(t.encode())
        h.update(np.ascontiguousarray(baskets[t]["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Synthetic world — a squeeze/expansion regime with a TUNABLE planted
# directional-continuation effect (the positive control)
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 4000,
    continuation: float = 0.0,
    quiet_vol: float = 0.006,
    loud_vol: float = 0.020,
    p_to_loud: float = 1.0 / 70.0,
    p_to_quiet: float = 1.0 / 15.0,
    persist_days: int = 15,
    kappa_quiet: float = 0.9,
    wick_quiet_mult: float = 1.3,
    wick_loud_mult: float = 0.15,
    start: str = "2005-01-03",
    seed: int = 670,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape that alternates range-bound / trending regimes.

    A 2-state Markov chain drives the regime: **quiet** (long runs, mean ~70 bars) is
    **mean-reverting** around a "center" anchor with pull ``kappa_quiet`` and small vol
    ``quiet_vol`` — real range-bound consolidation, so the Bollinger Band genuinely
    contracts INSIDE the Keltner Channel (a true squeeze — the intraday wick multiplier
    ``wick_quiet_mult`` is calibrated so ATR stays comfortably wider than the
    reverted close's rolling std). **Loud** (short runs, mean ~15 bars) turns the
    mean-reversion off, raises vol to ``loud_vol`` and shrinks the wick multiplier
    (``wick_loud_mult``, so ATR lags the sudden close-to-close vol) — the range breaks
    and the bands re-expand, firing the "release" a few bars in.

    The anchor **resets to the live price the instant reversion resumes** (the first
    quiet bar after a loud run) — so a loud excursion is never artificially erased by
    a stale pre-excursion anchor snapping back (that WAS a bug here: it manufactured a
    spurious reversal in the continuation=0 null before the reset was made same-bar).

    On the FIRST loud day after a quiet run an extra drift of ``continuation *
    loud_vol`` is added in a random +-1 direction, and that SAME sign persists for the
    next ``persist_days`` loud sessions — the planted "breakout direction predicts the
    next several days" effect. ``continuation = 0`` is the null world: the first loud
    day's sign carries no forward information, and the detector must not manufacture
    significance from it.

    Business-day index, ~16 years for the default ``n_days`` — far below the
    ~250-year pandas ns-timestamp trap. Returns ``(bars, truth)``; ``truth`` carries
    the regime and first-loud-day arrays for diagnostics.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    loud = np.zeros(n_days, dtype=bool)
    state = False
    for t in range(n_days):
        p = p_to_loud if not state else p_to_quiet
        if rng.random() < p:
            state = not state
        loud[t] = state

    log_price = np.empty(n_days)
    center = np.empty(n_days)
    direction = 0.0
    persist = 0
    for t in range(n_days):
        sigma = loud_vol if loud[t] else quiet_vol
        first_loud = loud[t] and (t == 0 or not loud[t - 1])
        first_quiet = (not loud[t]) and (t > 0 and loud[t - 1])
        if first_loud:
            direction = 1.0 if rng.random() < 0.5 else -1.0
            persist = persist_days
        drift = 0.0
        if persist > 0 and loud[t]:
            drift = continuation * loud_vol * direction
            persist -= 1
        if t == 0:
            log_price[0] = 0.0
            center[0] = 0.0
            continue
        # Reversion targets the LIVE anchor; on the reset bar itself the anchor IS the
        # prior price, so revert is exactly zero — no snap-back onto a stale pre-move
        # center (see docstring).
        ref = log_price[t - 1] if first_quiet else center[t - 1]
        revert = kappa_quiet * (ref - log_price[t - 1]) if not loud[t] else 0.0
        eps = rng.normal(0.0, sigma)
        log_price[t] = log_price[t - 1] + revert + drift + eps
        center[t] = log_price[t] if first_quiet else center[t - 1]

    log_ret = np.empty(n_days)
    log_ret[0] = log_price[0]
    log_ret[1:] = np.diff(log_price)

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick_sigma = np.where(loud, loud_vol * wick_loud_mult, quiet_vol * wick_quiet_mult)
    wick = np.abs(rng.normal(0.0, 1.0, n_days)) * wick_sigma * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    vol = rng.integers(1_000_000, 5_000_000, n_days).astype(float)

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(idx, name="date"),
    )
    truth = {"continuation": continuation, "quiet_vol": quiet_vol, "loud_vol": loud_vol,
             "p_to_loud": p_to_loud, "p_to_quiet": p_to_quiet, "persist_days": persist_days,
             "n_days": n_days, "seed": seed,
             "loud": loud, "first_loud": np.array([t for t in range(n_days)
                                                     if loud[t] and (t == 0 or not loud[t - 1])])}
    return bars, truth

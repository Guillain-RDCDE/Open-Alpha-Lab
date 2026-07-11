"""Data layer for Study 686 — Stick Sandwich.

The **bullish stick sandwich** (Nison, *Japanese Candlestick Charting Techniques*) is a
three-candle "failed rally" pattern:

* candle **t-2** — a bearish (down) bar, the first slice of "bread";
* candle **t-1** — a bullish (up) bar that rallies *above* candle t-2's close, the "filling"
  (buyers push back, briefly);
* candle **t**   — a bearish (down) bar whose close lands back at **~the same price** as
  candle t-2's close, the second slice of "bread" — the rally is completely given back and
  the tape prints the *same* closing level twice, with an up day sandwiched in between.

The folklore: two closes meeting at the same level, tested twice within three days, marks a
support level the market keeps "bouncing off" — a bullish reversal, especially after a
downtrend. This is the structural cousin of the two-candle **counterattack / meeting line**
(sibling study [460](../460-counterattack-lines/)) with a third bar and a full round trip
(down, up, back down to the same close) instead of a single gap-and-meet.

This module produces two tapes, both a tz-naive daily OHLC frame indexed by date:

* ``load_real(cache_dir=DEFAULT_CACHE)`` — the real Yahoo! daily tape (``yfinance``),
  **cache-first**: reads cached parquets and only touches the network on a cache miss (with
  retries + backoff), then caches so re-runs are offline. SPY plus a broad basket of
  long-listed US large-caps (the desk's standard "large basket" also used by sibling study
  [685-tri-star-doji](../685-tri-star-doji/)), so a fairly rare exact-3-bar geometry still
  yields a usable sample.

* ``synthetic_panel(...)`` — a deterministic, offline generator returning ``(data, truth)``.
  A single ``edge`` knob plants the only structure a stick-sandwich rule could possibly
  harvest: forced bearish/bullish/bearish 3-bar blocks whose outer two closes match within
  tolerance, each optionally followed by a genuine planted upward kick. ``edge=0`` is a pure
  random walk whose planted sandwiches carry no forward information — the null in a bottle.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the sandwich is
read on bars *t-2, t-1, t*'s OHLC (all known at the close of *t*), and the trade enters at
*t+1*'s close (one documented lag).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# SPY plus a broad, transparent, fixed basket of long-listed US large-caps spanning every
# major sector (a 30-name slice of the desk's standard large-cap basket, also used by
# 685-tri-star-doji) — chosen for long, clean yfinance history + sector spread, so a fairly
# rare exact 3-bar geometry has a genuine chance at a usable pooled sample. This is a
# *survivors* basket — long-listed names that did not delist; survivorship is named on the
# Signal axis, and the drift-matched base-rate control (not this basket choice) is what
# neutralizes it for the reversal claim.
BASKET = [
    "SPY",
    "AAPL", "MSFT", "XOM", "JNJ", "PG", "KO", "JPM", "WMT", "IBM", "CVX",
    "PFE", "MRK", "INTC", "CSCO", "HD", "MCD", "DIS", "BA", "CAT", "MMM",
    "HON", "UNH", "ORCL", "PEP", "ABT", "TXN", "COST", "GS", "AXP",
]

AS_OF = "2026-06-30"  # last complete calendar month at publication (2026-07-10)


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"sts_{safe}_1d.parquet")


def fetch_one(ticker: str, period: str = "25y", cache_dir: str = DEFAULT_CACHE,
              fetch: bool = False, retries: int = 3) -> pd.DataFrame:
    """Daily OHLC for one ``ticker``; cache-first.

    With ``fetch=False`` (default) this reads the cached parquet and never touches the
    network — on a cache miss it raises so the offline core stays offline. With
    ``fetch=True`` it downloads (retrying a couple of times with backoff on a transient
    failure) and writes the parquet cache. Returns a tz-naive daily OHLC frame sliced to
    :data:`AS_OF` (no partial-month drift).
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call load_real(fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = None
        for attempt in range(retries):
            try:
                raw = yf.download(ticker, period=period, interval="1d",
                                  auto_adjust=True, progress=False)
                if raw is not None and not raw.empty:
                    break
            except Exception:
                raw = None
            time.sleep(1.5 * (attempt + 1))
        if raw is None or raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close"]]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    bars = bars.dropna(how="any")
    return bars.loc[bars.index <= pd.Timestamp(AS_OF)]


def load_real(cache_dir: str = DEFAULT_CACHE, fetch: bool = False,
              tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Cache-first dict ``{ticker: daily OHLC frame}`` for the basket."""
    tickers = tickers or BASKET
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            out[t] = fetch_one(t, cache_dir=cache_dir, fetch=fetch)
        except (FileNotFoundError, RuntimeError):
            if not fetch:
                raise
    return out


def have_real(cache_dir: str = DEFAULT_CACHE, tickers: list[str] | None = None) -> bool:
    tickers = tickers or BASKET
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


# --------------------------------------------------------------------------- #
# Synthetic world — planted stick-sandwich reversal (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_names: int = 20, n_days: int = 3000, edge: float = 0.0,
    annual_vol: float = 0.16, trend_lookback: int = 10,
    plant_rate: float = 1.0 / 40.0, start: str = "2010-01-04", seed: int = 686,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Deterministic OHLC panel with a PLANTED post-sandwich reversal knob.

    Each name is an independent daily random walk in log-returns. At scheduled positions
    (rate ``plant_rate`` of eligible windows, gated on the preceding leg genuinely being
    down — the reversal-context the folklore needs) we overwrite three consecutive bars to
    form an exact bearish/bullish/bearish stick sandwich: bar 1 bearish, bar 2 bullish and
    closing above bar 1, bar 3 bearish and closing back within tolerance of bar 1's close.
    When ``edge > 0`` the sessions *following* a planted sandwich get an extra upward kick
    (the claimed bounce); at ``edge = 0`` the planted geometry carries no forward
    information and an entry is a fair coin — the null a harness must not manufacture
    significance from.

    Returns ``(data, truth)`` where ``data`` is ``{name: OHLC frame}`` (same shape as
    :func:`load_real`) and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)
    tol = 0.0015
    names = [f"N{i:02d}" for i in range(n_names)]

    data: dict[str, pd.DataFrame] = {}
    n_planted_total = 0
    for name in names:
        close = np.empty(n_days)
        open_ = np.empty(n_days)
        close[0] = open_[0] = 100.0
        bounce_left = 0
        i = 1
        while i < n_days:
            kick = 0.0
            if bounce_left > 0:
                kick = edge * daily_vol * 3.0
                bounce_left -= 1

            can_plant = (i + 2 < n_days) and (i - 1 - trend_lookback >= 0)
            plant_here = can_plant and (rng.random() < plant_rate)
            if plant_here:
                down_leg = np.log(close[i - 1]) < np.log(close[i - 1 - trend_lookback])
                if down_leg:
                    prev_close = close[i - 1]
                    # bread 1: bearish
                    body2 = daily_vol * (0.8 + 0.8 * rng.random())
                    open_[i] = prev_close
                    close[i] = prev_close * np.exp(-body2)
                    c_bread = close[i]
                    # filling: bullish, rallies clear above the bread close
                    body1 = daily_vol * (1.0 + 1.2 * rng.random())
                    open_[i + 1] = close[i]
                    close[i + 1] = close[i] * np.exp(body1)
                    # bread 2: bearish, lands back within tolerance of the bread-1 close
                    open_[i + 2] = close[i + 1]
                    close[i + 2] = c_bread * np.exp(rng.normal(0.0, tol * 0.35))
                    n_planted_total += 1
                    if edge > 0.0:
                        bounce_left = 4
                    i += 3
                    continue

            eps = rng.normal(0.0, daily_vol)
            open_[i] = close[i - 1]
            close[i] = close[i - 1] * np.exp(eps + kick)
            i += 1

        wick = np.abs(rng.normal(0.0, daily_vol * 0.5, n_days)) * close
        hi = np.maximum(open_, close) + wick
        lo = np.minimum(open_, close) - wick
        data[name] = pd.DataFrame(
            {"open": open_, "high": hi, "low": lo, "close": close},
            index=pd.DatetimeIndex(sessions, name="date"),
        )

    truth = {"edge": edge, "annual_vol": annual_vol, "trend_lookback": trend_lookback,
              "n_names": n_names, "n_days": n_days, "seed": seed, "tol": tol,
              "n_planted": n_planted_total}
    return data, truth


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]


def panel_fingerprint(panel: dict[str, pd.DataFrame]) -> str:
    """One fingerprint over the whole basket (concatenated closes, ticker-sorted)."""
    h = hashlib.sha1()
    for t in sorted(panel):
        h.update(t.encode())
        h.update(np.ascontiguousarray(panel[t]["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]

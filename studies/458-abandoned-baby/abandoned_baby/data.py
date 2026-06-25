"""Data layer for Study 458 (Abandoned-Baby / island-doji reversal).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A bullish abandoned baby is a three-bar island bottom: a down candle, then a **doji**
  (open ≈ close) that **gaps down** away from it, then an up candle that **gaps up** away
  from the doji — the doji is marooned on its own price island and "calls" the turn. The
  believers' claim is that the bar *after* the confirmation candle keeps rising. We plant
  exactly that: with ``edge > 0`` we periodically stamp a real island-bottom (a clean
  down-gap doji, an up-gap confirmation) and pull the path **up** for several bars after
  it, so the next-close entry harvests a genuine bounce; with ``edge = 0`` the path is a
  pure random walk with the *same* incidental gaps/dojis, so a detected abandoned baby is
  a fair coin. This is the positive control — a harness that cannot bank the planted
  bounce proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the abandoned
baby needs all three bars (the up-gap confirmation is the *third* bar), the pattern is read
on the close of the confirmation bar *t*, and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs candlestick proponents draw on: the broad tape, big-cap tech, small
# caps, and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 458,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of abandoned-baby reversal.

    The close path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    Independently we draw, for *every* bar, a random open-gap (the open jumps away from the
    prior close) and a small body — this manufactures incidental dojis and gaps so that at
    ``edge = 0`` the abandoned-baby detector fires on pure noise (a fair coin). On top of
    that, with ``edge > 0`` we periodically *stamp* a genuine island bottom: a down candle,
    a **down-gapped doji**, an **up-gapped** confirmation candle, and then a real upward
    drift over the following bars. So a next-close entry after a planted island banks a true
    bounce; with ``edge = 0`` the same gaps/dojis carry no future information.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    log_close = np.empty(n_days)
    log_open = np.empty(n_days)
    body = np.empty(n_days)          # close - open in log units
    lp = np.log(100.0)

    # planted-island bookkeeping
    planted = np.zeros(n_days, dtype=int)   # +1 on confirmation bars we stamped
    pull = np.zeros(n_days)                 # extra upward drift injected per bar
    bounce_left = 0                          # remaining bars of post-island drift

    # schedule island attempts roughly every ~28 bars (only matters when edge>0)
    next_island = 30

    i = 0
    while i < n_days:
        if edge > 0.0 and i == next_island and i + 3 < n_days:
            # --- stamp a clean bullish abandoned baby over bars i, i+1, i+2 ---
            # bar i: a down candle
            o0 = lp
            c0 = lp - 1.6 * daily_vol
            log_open[i] = o0; log_close[i] = c0; body[i] = c0 - o0
            # bar i+1: the doji, GAPPED DOWN below bar i's range, body ~ 0
            gap = 1.4 * daily_vol
            od = c0 - gap
            cd = od + rng.normal(0.0, daily_vol * 0.04)   # near-zero body => doji
            log_open[i + 1] = od; log_close[i + 1] = cd; body[i + 1] = cd - od
            # bar i+2: confirmation, GAPPED UP above the doji's range, up candle
            oc = cd + gap
            cc = oc + 1.6 * daily_vol
            log_open[i + 2] = oc; log_close[i + 2] = cc; body[i + 2] = cc - oc
            planted[i + 2] = 1
            lp = cc
            # inject a real upward bounce over the next several bars
            bounce_left = 6
            i += 3
            next_island = i + 28
            continue

        # --- ordinary bar: random open-gap + random body (manufactures noise dojis) ---
        # Big incidental open-gaps + an occasional near-zero body so the detector fires on
        # pure noise too (this is what makes edge=0 a fair coin with a real sample size).
        open_gap = rng.normal(0.0, daily_vol * 1.6)
        op = lp + open_gap
        extra = 0.0
        if bounce_left > 0:
            extra = edge * 1.1 * daily_vol     # planted post-island upward drift
            bounce_left -= 1
        if rng.random() < 0.18:                # ~18% of bars are near-doji bodies
            body_sd = daily_vol * 0.05
        else:
            body_sd = daily_vol
        bd = rng.normal(0.0, body_sd) + extra
        cp = op + bd
        log_open[i] = op; log_close[i] = cp; body[i] = cp - op
        pull[i] = extra
        lp = cp
        i += 1

    open_ = np.exp(log_open)
    close = np.exp(log_close)
    # wicks: small noise beyond the body extremes
    wick = np.abs(rng.normal(0.0, daily_vol * 0.35, n_days)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "n_days": n_days,
             "seed": seed, "n_planted": int(planted.sum())}
    return bars, truth


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

    Reads a cached parquet if present. Otherwise — and only if ``allow_fetch`` — downloads
    from yfinance (with a couple of retries + back-off on rate limits) and caches the parquet,
    so every subsequent call is fully offline.
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


def have_real(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every cached parquet for ``tickers`` is present (offline-safe check)."""
    tickers = tickers or DEFAULT_TICKERS
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]

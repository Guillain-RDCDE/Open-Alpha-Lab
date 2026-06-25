"""Data layer for Study 465 (Broadening Formation / megaphone top).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  A broadening formation is a *megaphone*: swing highs keep rising and swing lows keep
  falling, so the trading range fans out. The believers' claim is that this expanding
  range marks an exhausted top that **reverses down** once the lower boundary breaks. We
  plant exactly that: with ``edge > 0`` we drive a slowly *widening* channel (vol fans out)
  and, whenever the close pierces the widening lower boundary, we add a small downward pull
  so a short on the break harvests a real decline; with ``edge = 0`` the log-return series
  is a pure random walk and the lower-boundary break is a fair coin. This is the positive
  control — a harness that cannot bank the planted reversal proves nothing by finding
  nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the megaphone is
anchored on pivots that are *confirmed* (a fractal needs ``k`` bars on each side), a
lower-boundary break is detected on the close of *t*, and the trade is entered at *t+1*'s
close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs broadening-formation proponents draw megaphones on: the broad tape,
# big-cap tech, small caps, and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    pivot_k: int = 10,
    start: str = "2010-01-04",
    seed: int = 465,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of megaphone-reversal structure.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    On top of that, when ``edge > 0`` we slowly *widen* a channel (the half-width breathes
    out, a literal expanding range) and plant a megaphone-respecting force: whenever the
    close pierces *below* the widening lower boundary we add a small **downward** pull
    proportional to ``edge`` (the "reversal" the lore promises after a broadening top), and a
    symmetric upward fade after an upper break. At ``edge = 0`` the tape is a pure martingale
    with a *constant* range and a lower-boundary break is a fair coin; at ``edge > 0`` a break
    of the expanding lower boundary is followed by a real decline the short should bank.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    log_p = np.log(100.0)
    med = log_p
    med_drift = rng.normal(0.0, daily_vol * 0.10, n_days)  # gentle wandering channel center

    # Half-width breathes: at edge=0 it stays ~constant; at edge>0 it slowly fans out
    # (the megaphone), cycling so several broadening episodes appear over the tape.
    base_w = 0.05
    t_axis = np.arange(n_days)
    if edge > 0.0:
        # a slow expand/contract cycle so the range repeatedly broadens then resets,
        # i.e. swing highs rise and swing lows fall (a real megaphone)
        cycle = 0.5 * (1.0 - np.cos(2.0 * np.pi * t_axis / 220.0))  # 0..1, period ~220d
        half_w = base_w * (1.0 + 2.0 * cycle)
    else:
        half_w = np.full(n_days, base_w)

    # A bounded, decaying downward momentum triggered when price breaks the (widening) lower
    # boundary: this plants a genuine multi-day *decline after the break*, the reversal the
    # lore promises — capped so the path can never run away (no overflow).
    impulse = 0.0
    decay = 0.92
    cooldown = 0
    for i in range(n_days):
        med += med_drift[i]
        lower = med - half_w[i]
        upper = med + half_w[i]
        impulse *= decay
        # Only react to a *lower-boundary* break (the short signal) so the planted reversal is
        # the thing the rule actually trades; keep it gentle so the megaphone keeps re-forming.
        if edge > 0.0 and cooldown == 0 and log_p < lower:
            impulse = -edge * daily_vol * 1.6      # gentle, decaying downward drift
            cooldown = 30                          # one episode per ~6 weeks, channel survives
        cooldown = max(0, cooldown - 1)
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + impulse
        close[i] = np.exp(log_p)

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "pivot_k": pivot_k,
             "n_days": n_days, "seed": seed}
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

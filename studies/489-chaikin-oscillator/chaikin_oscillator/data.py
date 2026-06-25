"""Data layer for Study 489 (Chaikin Oscillator).

Two tapes, one shape (a tz-naive daily OHLC**V** frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The Chaikin Oscillator is built from the Accumulation/Distribution Line (ADL), which
  cumulates volume weighted by the Money Flow Multiplier (where in its range the bar
  closed). The believers' claim is that **A/D momentum leads price**: when the oscillator
  crosses above zero, accumulation is building and a price rise follows. We plant exactly
  that: with ``edge > 0`` the path receives a real forward drift in the days *after* a
  genuine accumulation up-cross (volume concentrated near the highs), so the cross-above-zero
  entry harvests a real lead-lag effect; with ``edge = 0`` the path is a pure random walk and
  the volume is independent noise, so the cross is a fair coin. This is the positive control —
  a harness that cannot bank the planted lead proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

  **Volume note.** The shared desk cache stores OHLC only (it was built for price-pattern
  studies). The Chaikin Oscillator needs volume; when the cached frame has no ``volume`` we
  synthesize a **deterministic** proxy from the bar's true range (``high-low``) scaled by a
  fixed turnover constant — a documented, reproducible stand-in. The proxy is a *monotone
  function of range only* (it carries no future information), so it cannot fabricate a lead;
  if anything it weakens the indicator. The placebo and the random baseline both neutralize it.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the oscillator is
read on the close of *t*, the cross-above-zero is confirmed using only data up to *t*, and the
trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs Chaikin-oscillator proponents watch: the broad tape, big-cap tech, small
# caps, and a couple of cross-asset charts. Daily, liquid, long history.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Deterministic volume proxy (when the shared cache stores OHLC only)
# --------------------------------------------------------------------------- #
def _proxy_volume(bars: pd.DataFrame) -> pd.Series:
    """A deterministic, look-ahead-free volume proxy from the daily true range.

    Volume is a monotone function of the bar's range ``high-low`` (busier days have wider
    ranges) times a fixed turnover constant. It uses *only* current-bar OHLC, so it leaks no
    future information into the Money Flow Multiplier. It is a stand-in for the real share
    count, not a faithful one; it cannot manufacture a price-leading signal.
    """
    rng = (bars["high"] - bars["low"]).abs()
    base = bars["close"].abs()
    # turnover ~ (range / price) gives a dimensionless busy-ness; scale to plausible volume.
    vol = (rng / base.replace(0.0, np.nan)).fillna(0.0) * 1.0e8 + 1.0e6
    return vol.astype(float)


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    seed: int = 489,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a *known* amount of A/D-momentum-leads-price effect.

    The price path is a random walk in log-returns with daily sigma ``annual_vol/sqrt(252)``.
    Volume is independent positive noise. On top of that we plant the Chaikin claim: on a small
    fraction of days we stage a *genuine accumulation event* — volume spikes while the close
    prints near the high (Money Flow Multiplier ~ +1, so the ADL jumps up). With ``edge > 0`` we
    then add a real upward drift to the **next several days** of returns, exactly the "A/D
    momentum leads price" lead-lag the oscillator's cross-above-zero is supposed to detect. At
    ``edge = 0`` the accumulation events still occur (so the oscillator still fires) but carry NO
    forward drift, so the cross is a fair coin.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Stage accumulation events on a deterministic schedule (~ every 18 sessions, jittered).
    is_accum = np.zeros(n_days, dtype=bool)
    i = 25
    while i < n_days - 65:
        is_accum[i] = True
        i += int(rng.integers(14, 24))

    # Forward drift bump injected over the days AFTER an accumulation event (the planted lead).
    bump = np.zeros(n_days)
    lead_len = 25
    for t in np.flatnonzero(is_accum):
        bump[t + 1: t + 1 + lead_len] += edge * daily_vol * 1.2

    close = np.empty(n_days)
    open_ = np.empty(n_days)
    high = np.empty(n_days)
    low = np.empty(n_days)
    volume = np.empty(n_days)
    log_p = np.log(100.0)
    prev_close = 100.0

    for d in range(n_days):
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + bump[d]
        c = np.exp(log_p)
        o = prev_close
        # On accumulation days the close prints near the HIGH (MFM ~ +1) on heavy volume.
        if is_accum[d]:
            base_range = abs(rng.normal(0.0, daily_vol)) * c + 1e-9
            hi = max(o, c) + base_range * 0.15
            lo = min(o, c) - base_range * 1.6      # long lower wick -> close near top of range
            volume[d] = abs(rng.normal(0.0, 1.0)) * 5.0e6 + 1.2e7   # heavy
        else:
            wick = abs(rng.normal(0.0, daily_vol * 0.5)) * c
            hi = max(o, c) + wick
            lo = min(o, c) - wick
            volume[d] = abs(rng.normal(0.0, 1.0)) * 2.0e6 + 3.0e6   # ordinary
        close[d] = c
        open_[d] = o
        high[d] = hi
        low[d] = lo
        prev_close = c

    bars = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "n_days": n_days, "seed": seed,
             "n_accum": int(is_accum.sum()), "lead_len": lead_len}
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
    """Real daily OHLCV for ``ticker``; **cache-first** (network only on a cache miss).

    Reads a cached parquet if present. Otherwise — and only if ``allow_fetch`` — downloads
    from yfinance (with a couple of retries + back-off on rate limits) and caches the parquet,
    so every subsequent call is fully offline. If the cached/fetched frame has no ``volume``
    column, a deterministic look-ahead-free proxy is attached (see ``_proxy_volume``).
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
    cols = ["open", "high", "low", "close"]
    out = bars[cols].copy()
    if "volume" in bars.columns and bars["volume"].abs().sum() > 0:
        out["volume"] = bars["volume"].astype(float)
    else:
        out["volume"] = _proxy_volume(out)
    return out


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
                raw = raw.rename(columns=str.lower)
                keep = [c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns]
                bars = raw[keep]
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

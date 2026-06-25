"""Data layer for Study 469 (Relative Vigor Index).

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The RVI reads the bar *body* (close − open) relative to its *range* (high − low): it is high
  when bars close strongly above their open. The believers' claim is that when the smoothed body
  vigor turns up — the RVI **crosses above its signal line** — the move *continues* (momentum).
  We plant exactly that: with ``edge > 0`` the path carries a real, slow body-momentum — once the
  recent run of bodies turns positive, subsequent bodies/returns are pulled positive (and
  vice-versa), so an RVI up-cross harvests genuine follow-through; with ``edge = 0`` the bodies
  are an i.i.d. martingale and the cross is a fair coin. This is the positive control — a harness
  that cannot bank the planted momentum proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline. Daily history is
  long (20+ years) and free of the 60-day cap that affects sub-hourly bars.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the RVI and its
signal line are causal (only past/closed bars), the cross is detected on the close of *t*, and
the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Indices / ETFs RVI proponents draw on: the broad tape, big-cap tech, small caps, and a couple
# of cross-asset charts. Daily, liquid, long history. (Reused so the study runs offline.)
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    start: str = "2010-01-04",
    p_switch: float = 0.02,
    seed: int = 469,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLC tape with a *known* amount of persistent-regime momentum.

    The bar body (close − open) is what the RVI reads. We plant a **hidden two-state regime** (a
    bull / bear that flips with probability ``p_switch`` each bar, so it *persists* for tens of
    bars). When ``edge > 0`` the bull regime adds a positive drift to every body (and therefore to
    the close-to-close return), the bear regime a symmetric negative drift — so during a bull run
    the bodies are strong-up, the RVI rides **above** its signal line, the RVI **cross-up** marks
    *entering* the bull regime, and because the regime persists the **forward** return is genuinely
    positive (the follow-through the cross is supposed to forecast). A small independent overnight
    gap breaks the exact body==return identity so the RVI is a real smoother of vigor rather than a
    trivial proxy for the return.

    At ``edge = 0`` the regime contributes nothing: bodies are i.i.d. and the RVI cross is a fair
    coin (t ≈ 0). At ``edge > 0`` an RVI up-cross genuinely leads positive returns that the detector
    should bank. This is the positive control.

    Returns ``(bars, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    close = np.empty(n_days)
    open_ = np.empty(n_days)
    hi = np.empty(n_days)
    lo = np.empty(n_days)

    state = 1                     # +1 bull, -1 bear; flips with prob p_switch (persistent regime)
    log_c = np.log(100.0)

    for i in range(n_days):
        if rng.random() < p_switch:
            state = -state

        gap = rng.normal(0.0, daily_vol * 0.3)          # small overnight gap (open != prev close)
        log_o = log_c + gap

        drift = (edge * daily_vol * state) if edge > 0.0 else 0.0
        body = rng.normal(0.0, daily_vol) + drift
        log_c = log_o + body

        o = np.exp(log_o)
        c = np.exp(log_c)
        wick = abs(rng.normal(0.0, daily_vol * 0.5)) * c
        open_[i] = o
        close[i] = c
        hi[i] = max(o, c) + wick
        lo[i] = min(o, c) - wick

    bars = pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "p_switch": p_switch,
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

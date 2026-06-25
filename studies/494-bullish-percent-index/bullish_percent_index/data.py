"""Data layer for Study 494 (Bullish Percent Index).

The BPI is a *breadth* statistic: it needs a basket, not a single tape. We build the breadth
panel from the desk's cached liquid ETFs and read SPY as the traded instrument.

Two tapes, one shape (a tz-naive daily OHLC frame, calendar-date indexed):

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob**.
  The BPI claim is that when breadth gets *washed out* (few members above their MA) the index
  is oversold and bounces. We plant exactly that: with ``edge > 0`` we drive a common breadth
  factor that drags every member down together, and whenever that factor is deeply negative
  (broad oversold) we add an upward pull to the index path, so a "BPI < threshold → buy" entry
  harvests a real bounce; with ``edge = 0`` the index is a pure random walk and the oversold
  signal is a fair coin. This is the positive control — a harness that cannot bank the planted
  bounce proves nothing by finding nothing on the real tape.

- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads a
  cached parquet if present and only touches the network on an explicit cache miss (with a
  short back-off + retry), then caches the parquet so re-runs are offline.

- ``breadth_basket`` — loads the breadth members cache-first into a single aligned close panel.

BREADTH IS A PROXY. True BPI counts members on a Point & Figure buy signal across a *full
exchange* (e.g. all NYSE issues). We approximate it with the % of a small ETF basket above its
moving average. This is a coarse proxy and it **caps** the test — but the Signal axis question
is unchanged: does the breadth signal beat random-day entries on SPY? No look-ahead is baked in
here — that discipline lives in ``strategy.py``: the BPI is read on the close of *t*, the
oversold cross is confirmed on *t*, and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# The traded instrument is SPY; the breadth basket is the surviving liquid ETFs the desk caches
# (a coarse stand-in for true exchange breadth). Reused so the study runs fully offline.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

# Members whose "above-MA" votes form the breadth oscillator. We use the broad-market sleeves
# (drop GLD, a cross-asset hedge that is not part of equity breadth) so BPI measures equity
# participation. In a fuller study these would be XLK XLF XLE XLV XLI XLY XLP XLU XLB; offline
# we use what the desk caches.
BREADTH_MEMBERS = ["SPY", "QQQ", "IWM", "DIA"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    n_members: int = 9,
    ma_win: int = 50,
    start: str = "2010-01-04",
    seed: int = 494,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """A reproducible daily tape + breadth panel with a *known* amount of oversold mean reversion.

    A common breadth factor ``f_t`` (an AR(1) wandering between washed-out and frothy) drives
    every member's drift, so members rise and fall *together* — exactly the co-movement BPI is
    built to read. The traded index is a random walk in log-returns; with ``edge > 0`` we add a
    small upward pull to the index whenever the breadth factor is deeply negative (a broad
    oversold), so a "BPI low → buy" entry banks a real bounce. At ``edge = 0`` the index is a
    pure martingale and the oversold signal is a fair coin.

    Returns ``(index_bars, members_close, truth)``; ``members_close`` is an N-column close panel,
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Common breadth factor: slow AR(1), zero-mean. Negative => broad oversold.
    f = np.empty(n_days)
    f[0] = 0.0
    phi = 0.97
    fvol = 0.9
    for i in range(1, n_days):
        f[i] = phi * f[i - 1] + rng.normal(0.0, fvol)

    # Member close paths: each loads on the common factor (so breadth co-moves) + idio noise.
    members = np.empty((n_days, n_members))
    logp = np.full(n_members, np.log(100.0))
    loadings = rng.uniform(0.6, 1.0, n_members)
    for i in range(n_days):
        common = 0.0006 * f[i]  # factor tilts every member's daily drift together
        for j in range(n_members):
            logp[j] += loadings[j] * common + rng.normal(0.0, daily_vol)
        members[i] = np.exp(logp)

    # Traded index path: random walk, plus a planted bounce when breadth is deeply negative.
    close = np.empty(n_days)
    li = np.log(100.0)
    f_lo = np.quantile(f, 0.20)  # "broad oversold" threshold on the factor
    for i in range(n_days):
        pull = 0.0
        if edge > 0.0 and f[i] < f_lo:
            pull = edge * (f_lo - f[i]) * daily_vol
        li += rng.normal(0.0, daily_vol) + pull
        close[i] = np.exp(li)

    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(0.0, daily_vol * 0.5, close.size)) * close
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick

    idx = pd.DatetimeIndex(sessions, name="date")
    bars = pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close}, index=idx)
    mclose = pd.DataFrame(members, index=idx,
                          columns=[f"m{j}" for j in range(n_members)])
    truth = {"edge": edge, "annual_vol": annual_vol, "n_members": n_members,
             "ma_win": ma_win, "n_days": n_days, "seed": seed}
    return bars, mclose, truth


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


def breadth_basket(
    members: list[str] | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = False,
) -> pd.DataFrame:
    """Aligned close panel for the breadth members (cache-first), one column per member.

    Each member is loaded with :func:`load_real` (cache-first, optional single network fetch +
    back-off per ticker, then parquet-cached so re-runs are offline). Columns are inner-joined
    on the common date index so every breadth vote is computed on a fully-populated row.
    """
    members = members or BREADTH_MEMBERS
    cols = {}
    for t in members:
        cols[t] = load_real(t, cache_dir=cache_dir, allow_fetch=allow_fetch)["close"]
    panel = pd.DataFrame(cols).dropna(how="any")
    panel.index.name = "date"
    return panel


def have_real(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every cached parquet for ``tickers`` is present (offline-safe check)."""
    tickers = tickers or DEFAULT_TICKERS
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]

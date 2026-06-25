"""Data layer for Study 492 (Up-Down-Volume breadth).

Three tapes, two shapes:

- ``load_real`` — the real Yahoo! daily OHLC tape (``yfinance``), **cache-first** and identical
  to the desk template (Study 450). It serves the forward-return instrument (SPY) and the
  drift-matched random baseline. Cache-first: reads a cached parquet if present and only touches
  the network on an explicit cache miss (short back-off + retry), then caches it so re-runs are
  offline.

- ``load_breadth`` — a small **basket of liquid US sector ETFs** (XLK XLF XLE XLV XLI XLY XLP
  XLU XLB) plus SPY, loaded WITH volume (OHLCV), cache-first. This is the **breadth proxy**:
  from it we build daily up-volume vs down-volume. It is a *proxy* for true NYSE/exchange
  up-down volume (TRIN / Arms-index inputs), which would aggregate every listed issue. A
  nine-ETF basket cannot reproduce the full advance/decline volume of thousands of stocks, so
  this is an honest cap on the test (stated in the docs).

- ``synthetic_panel`` — a *deterministic, offline* generator with a **planted-edge knob** that
  is **specific to the up/down-volume indicator**. The believers' claim is that a *selling
  climax* (down-volume utterly dominating up-volume — a panic low) is followed by a bounce in
  the index. We plant exactly that: with ``edge > 0`` the synthetic index path mean-reverts
  upward on the bars whose breadth basket prints a down-volume climax, so a "buy the climax"
  entry harvests a real bounce; with ``edge = 0`` the index is a pure random walk *and* the
  breadth ratio is generated independently of future returns, so a climax entry is a fair coin.
  This is the positive control — a harness that cannot bank the planted bounce proves nothing
  by finding nothing on the real tape.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the breadth ratio is
read on the close of *t* and the trade is entered at *t+1*'s close.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Forward-return instruments + drift baseline (reused from the desk template so the OHLC cache
# is shared and the study runs offline).
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

# Breadth basket: the nine SPDR sector ETFs + the broad tape. Liquid, long history, daily volume.
# Up/down-volume is computed across this basket as a proxy for exchange-wide advance/decline volume.
BREADTH_TICKERS = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (positive control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_days: int = 1500,
    edge: float = 0.0,
    annual_vol: float = 0.16,
    climax_q: float = 0.10,
    start: str = "2010-01-04",
    seed: int = 492,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily tape carrying an index path **and** a breadth up/down-volume ratio,
    with a *known* amount of selling-climax mean reversion planted in.

    The index path is a random walk in log-returns (daily sigma ``annual_vol/sqrt(252)``).
    Alongside it we synthesize a breadth basket and reduce it to a daily up-volume **share**
    (the same indicator used on the real basket). At ``edge = 0`` the share is generated
    **independently** of future index returns, so a down-volume-climax entry (a low share) is a
    fair coin. At ``edge > 0`` the index gets a planted **upward** pull on the bars that print a
    down-volume climax, so the climax-buy banks a real bounce the detector should catch.

    The frame exposes the columns the strategy consumes: ``close`` (the index), and ``up_vol`` /
    ``down_vol`` (the basket's aggregate up/down volume for the day) so ``strategy`` can rebuild
    the ratio exactly as it does on the real basket.

    Returns ``(panel, truth)``; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    sessions = pd.bdate_range(start=start, periods=n_days)

    # Independent breadth ratio: a mean-reverting Gaussian in log up/down space.
    uvs = np.empty(n_days)
    x = 0.0
    for i in range(n_days):
        x = 0.85 * x + rng.normal(0.0, 0.5)
        uvs[i] = x
    # A down-volume climax = uvs well into the left tail.
    climax_thresh = np.quantile(uvs, climax_q)
    is_climax = uvs <= climax_thresh

    close = np.empty(n_days)
    log_p = np.log(100.0)
    for i in range(n_days):
        pull = 0.0
        if edge > 0.0 and is_climax[i]:
            # planted selling-climax bounce: a positive expected drift on the day AFTER a climax
            pull = edge * daily_vol * 6.0
        eps = rng.normal(0.0, daily_vol)
        log_p += eps + pull
        close[i] = np.exp(log_p)

    # Reconstruct an up/down volume pair consistent with uvs (base volume scaled by the ratio).
    base = 1.0e7
    up_vol = base * np.exp(uvs / 2.0)
    down_vol = base * np.exp(-uvs / 2.0)

    panel = pd.DataFrame(
        {"close": close, "up_vol": up_vol, "down_vol": down_vol},
        index=pd.DatetimeIndex(sessions, name="date"),
    )
    truth = {"edge": edge, "annual_vol": annual_vol, "climax_q": climax_q,
             "n_days": n_days, "seed": seed, "climax_thresh": float(climax_thresh)}
    return panel, truth


# --------------------------------------------------------------------------- #
# Real tape — Yahoo daily, cache-first
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str, with_vol: bool = False) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    tag = "1d_v" if with_vol else "1d"
    return os.path.join(cache_dir, f"bars_{safe}_{tag}.parquet")


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
        bars = _download(ticker, start, end, with_vol=False)
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


def load_breadth_member(
    ticker: str,
    start: str = "2005-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = True,
) -> pd.DataFrame:
    """Real daily OHLC**V** for a breadth-basket member; cache-first (network only on a miss).

    Same idiom as :func:`load_real` but keeps the **volume** column (the up/down-volume study
    needs it). Cached under a separate ``*_1d_v.parquet`` key so it never collides with the
    OHLC-only template cache.
    """
    path = _cache_path(ticker, cache_dir, with_vol=True)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
    elif allow_fetch:
        bars = _download(ticker, start, end, with_vol=True)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)
    else:
        raise FileNotFoundError(
            f"No cached OHLCV tape for {ticker} at {path}. "
            f"Call load_breadth_member({ticker!r}) once (network) to populate the cache."
        )
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars[["open", "high", "low", "close", "volume"]]


def load_breadth(
    tickers: list[str] | None = None,
    start: str = "2005-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    allow_fetch: bool = True,
) -> dict[str, pd.DataFrame]:
    """The breadth basket as ``{ticker: OHLCV frame}``; cache-first per member."""
    tickers = tickers or BREADTH_TICKERS
    return {t: load_breadth_member(t, start, end, cache_dir, allow_fetch) for t in tickers}


def _download(ticker: str, start: str, end: str | None, with_vol: bool = False) -> pd.DataFrame:
    import yfinance as yf  # lazy: only on a real cache miss

    cols = ["open", "high", "low", "close"] + (["volume"] if with_vol else [])
    last_err = None
    for attempt in range(3):
        try:
            raw = yf.download(ticker, start=start, end=end, interval="1d",
                              auto_adjust=True, progress=False)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                bars = raw.rename(columns=str.lower)[cols]
                bars.index.name = "date"
                return bars
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"yfinance returned no daily bars for {ticker}: {last_err}")


def have_real(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every cached OHLC parquet for ``tickers`` is present (offline-safe check)."""
    tickers = tickers or DEFAULT_TICKERS
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in tickers)


def have_breadth(tickers: list[str] | None = None, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every cached OHLCV breadth parquet is present (offline-safe check)."""
    tickers = tickers or BREADTH_TICKERS
    return all(os.path.exists(_cache_path(t, cache_dir, with_vol=True)) for t in tickers)


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]

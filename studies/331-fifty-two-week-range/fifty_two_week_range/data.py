"""Data layer for Study 331 (Fifty-Two-Week-Range).

The question this study turns on is *incremental*: study 236 already tested the
George-Hwang **distance-to-the-high** ratio ``close / high_52w`` and study 202 the
mirror **distance-to-the-low**. This study tests the **range position**

    pos(t) = (close(t) - low_52w(t)) / (high_52w(t) - low_52w(t))   ∈ [0, 1]

— George & Hwang's "nearness" measure that anchors on *both* endpoints at once — and
asks the only question that makes it a new study and not a re-run of 236/202: **does
the second anchor (the low) carry any forward information the high alone does not?**
So the synthetic generator here plants two *separable* things the strategy can race
apart, not one momentum knob.

Two tapes, one shape (a tz-naive daily close panel indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator with two independent
  knobs:
    * ``range_edge``  — forward drift that scales with the *range position* itself
      (a stock high in its range keeps rising). This is the thing range-position can
      harvest that the high-ratio captures *equally well* (both are ≈1 at the top), so
      it is a confound, not the discriminating signal.
    * ``low_edge``    — forward drift that scales with proximity to the *low* and is
      **orthogonal to the high ratio** by construction (it lives in the middle/bottom
      of the range, where ``close/high`` and range-position disagree most). This is the
      *only* thing a range-position signal can know that a high-only signal cannot.
  With ``low_edge = 0`` the two signals are informationally redundant — range-position
  must *not* beat the high ratio. With ``low_edge > 0`` it must. That contrast is the
  study's positive control and its null, in one bottle.

- ``load_panel`` — the real Yahoo! daily tape, **cache-first**. It reuses the shared
  S&P-500 large-cap basket cached by study 202 (same names, same ``ftw52l_*`` parquet
  layout) so the reproducible core never has to touch the network. On a cache miss it
  falls back gracefully (raises a clear ``FileNotFoundError`` unless ``fetch=True``).

Survivorship note: the basket is fixed to tickers still trading as of 2026; this
*biases toward survivors* and is named on the Signal axis throughout. The bias can
point either way for a range signal and we reason about it explicitly in the docs.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (signals are
formed on closes up to *t*; forward returns start at *t* and are realised at
*t + hold*, the one documented lag).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# Study 202 cached the same large-cap basket; reuse it so we never hit the network.
SHARED_CACHE = os.path.abspath(
    os.path.join(HERE, "..", "..", "202-fifty-two-week-low", "_cache")
)

# Representative S&P 500 large-cap basket — survivorship-biased (all still trade).
SP500_BASKET = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "JPM", "JNJ", "XOM", "PG",
    "V", "MA", "HD", "UNH", "CVX",
    "MRK", "ABBV", "PEP", "KO", "WMT",
]

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_stocks: int = 20,
    n_days: int = 1500,
    range_edge: float = 0.0,
    low_edge: float = 0.0,
    daily_vol: float = 0.015,
    window: int = 252,
    start: str = "2016-01-04",
    seed: int = 331,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily close panel with two *separable* forward-edge knobs.

    Each day, each stock's expected next-day log return is

        mu_{i,t} = range_edge * (pos_{i,t} - 0.5)
                 + low_edge   * (pos_{i,t} - high_ratio_{i,t})

    where ``pos`` is the 52-week range position ∈ [0,1] and ``high_ratio =
    close / high_52w`` ∈ (0,1] is the study-236 high-only signal. The discriminating
    term ``pos - high_ratio`` is exactly the *extra information the second anchor (the
    low) carries*: it is zero when the range is one-sided (high ratio ≈ range position)
    and grows when the low is far below — precisely where the two signals disagree. The
    construction guarantees:

    - ``range_edge``: a tilt **both** signals capture (a confound), so it must *not*
      separate range-position from the high ratio — both spreads move together.
    - ``low_edge``:  a tilt the high ratio is structurally blind to, so it is the
      single thing that lets range-position *beat* the high ratio in the horse race.

    - ``range_edge = low_edge = 0`` → a martingale; both signals are coins.

    Returns ``(panel, truth)``. ``panel`` is (n_days, n_stocks), columns 'S00'..,
    indexed by business-day dates; ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start=start, periods=n_days)
    tickers = [f"S{i:02d}" for i in range(n_stocks)]

    log_px = np.zeros((n_days, n_stocks))
    log_px[0] = np.log(100.0)
    # Rolling high/low trackers seeded from a warm-up burst so positions are defined.
    for t in range(1, n_days):
        lo_t = t - window if t >= window else 0
        prices = np.exp(log_px[lo_t:t])              # window of closes so far
        hi = prices.max(axis=0)
        lo = prices.min(axis=0)
        cur = np.exp(log_px[t - 1])
        rng_width = np.maximum(hi - lo, 1e-9)
        pos = np.clip((cur - lo) / rng_width, 0.0, 1.0)
        high_ratio = np.clip(cur / np.maximum(hi, 1e-9), 0.0, 1.0)
        # pos - high_ratio is the *extra* information the second anchor (the low) carries:
        # zero for a one-sided range, growing when the low is far below the close.
        drift = range_edge * (pos - 0.5) + low_edge * (pos - high_ratio)
        eps = rng.normal(0.0, daily_vol, n_stocks)
        log_px[t] = log_px[t - 1] + drift + eps

    panel = pd.DataFrame(
        np.exp(log_px), index=pd.DatetimeIndex(cal, name="date"), columns=tickers
    )
    truth = {
        "n_stocks": n_stocks,
        "n_days": n_days,
        "range_edge": range_edge,
        "low_edge": low_edge,
        "daily_vol": daily_vol,
        "window": window,
        "seed": seed,
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real tape — cache-first, reusing study 202's parquet layout
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    """Study 202's parquet naming, so we can read its cache directly."""
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"ftw52l_{safe}_1d.parquet")


def fetch_daily(
    ticker: str,
    period: str = "15y",
    fetch: bool = False,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``; cache-first, network only on ``fetch=True``.

    Looks first in ``cache_dir`` (defaults to this study's ``_cache``), then in the
    shared study-202 cache, before considering a network pull. The reproducible core
    and the test-suite therefore never touch the network on a populated machine.
    """
    candidates = []
    if cache_dir is not None:
        candidates.append(_cache_path(ticker, cache_dir))
    candidates.append(_cache_path(ticker, DEFAULT_CACHE))
    candidates.append(_cache_path(ticker, SHARED_CACHE))

    if not fetch:
        for path in candidates:
            if os.path.exists(path):
                bars = pd.read_parquet(path)
                if bars.index.tz is not None:
                    bars.index = bars.index.tz_localize(None)
                bars.index = pd.DatetimeIndex(bars.index, name="date")
                return bars
        raise FileNotFoundError(
            f"No cached daily tape for {ticker}. Looked in: {candidates}. "
            f"Call fetch_daily({ticker!r}, fetch=True) once to populate the cache."
        )

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download(
        ticker, period=period, interval="1d", auto_adjust=True, progress=False
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    bars.index.name = "date"
    out_dir = cache_dir or DEFAULT_CACHE
    os.makedirs(out_dir, exist_ok=True)
    bars.to_parquet(_cache_path(ticker, out_dir))
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def have_real_cache(
    tickers: list[str] = SP500_BASKET, cache_dir: str | None = None
) -> bool:
    """True iff every ticker resolves to a cached parquet (this study or shared 202)."""
    for t in tickers:
        paths = [_cache_path(t, DEFAULT_CACHE), _cache_path(t, SHARED_CACHE)]
        if cache_dir is not None:
            paths.insert(0, _cache_path(t, cache_dir))
        if not any(os.path.exists(p) for p in paths):
            return False
    return True


def load_panel(
    tickers: list[str] = SP500_BASKET,
    field: str = "close",
    cache_dir: str | None = None,
    allow_survivorship_bias: bool = False,
) -> pd.DataFrame:
    """Load a cached wide panel of one OHLC field for all tickers (cache-first).

    The basket is fixed to survivors; the opt-in guard makes the bias an explicit
    decision (see METHODOLOGY → *Survivorship is named on the Signal axis*).
    Returns a (n_days, n_tickers) frame on the inner-join of common dates.
    """
    if not allow_survivorship_bias:
        raise PermissionError(
            "load_panel uses a current-membership (survivorship-biased) basket. "
            "Pass allow_survivorship_bias=True to opt in, and carry the caveat to "
            "every place the Signal stamp appears."
        )
    frames = {}
    for t in tickers:
        bars = fetch_daily(t, fetch=False, cache_dir=cache_dir)
        frames[t] = bars[field]
    panel = pd.DataFrame(frames).sort_index()
    panel = panel.ffill(limit=5).dropna()
    return panel


def load_ohlc_panels(
    tickers: list[str] = SP500_BASKET,
    cache_dir: str | None = None,
    allow_survivorship_bias: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load close/high/low panels together (range-position needs all three).

    Returns ``{"close": df, "high": df, "low": df}`` aligned on common dates.
    """
    out = {}
    for field in ("close", "high", "low"):
        out[field] = load_panel(
            tickers, field=field, cache_dir=cache_dir,
            allow_survivorship_bias=allow_survivorship_bias,
        )
    common = out["close"].index
    for f in ("high", "low"):
        common = common.intersection(out[f].index)
    return {f: out[f].loc[common] for f in out}


def fingerprint(panel: pd.DataFrame) -> str:
    """A short content fingerprint of a panel, for the as-of stamp."""
    arr = np.ascontiguousarray(panel.to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]

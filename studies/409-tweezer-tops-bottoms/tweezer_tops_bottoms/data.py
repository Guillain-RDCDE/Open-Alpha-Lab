"""Data layer for Study 409 (Tweezer Tops & Bottoms).

Two tapes, one shape (a tz-naive daily OHLCV frame indexed by date):

- ``synthetic_panel`` — a *deterministic, offline* generator returning a dict of daily
  OHLCV frames plus a ``truth`` dict.  A planted-edge knob (``edge``) injects the exact
  structure the tweezer-reversal folklore claims: after a synthetic tweezer-bottom (two
  bars sharing a near-equal low at the end of a down-leg) the next few days drift *up* by
  ``edge`` (and the mirror for tweezer-tops drift *down*).  ``edge = 0`` is a pure random
  walk — a fair coin — so a test can assert the detector banks the planted reversal only
  when one is planted, and reads ≈ base-rate noise when it isn't.  This is the study's
  null and positive control in a bottle.
- ``load_real`` — the real Yahoo! daily tape (``yfinance``), **cache-first**: it reads the
  per-name parquet under ``_cache/`` and only touches the network on an explicit cache
  miss (with a small retry/backoff), then caches the parquet so re-runs are offline.

A tweezer is a **two-candle** pattern:

- **Tweezer bottom** (bullish reversal): in a short *downtrend*, two consecutive bars
  whose **lows match** within a tolerance — the second bar fails to make a new low and
  "tweezes" the first.  Folklore: buy, the turn is in.
- **Tweezer top** (bearish reversal): in a short *uptrend*, two consecutive bars whose
  **highs match** within a tolerance — folklore: sell.

The precise OHLC rules and the no-look-ahead discipline (signal known at close of the
second bar *t*, position entered at the *next* bar's open *t+1*) live in ``strategy.py``.

SURVIVORSHIP CAVEAT: the real basket is ~30 US large-caps + SPY that are *still trading*
in 2026.  A pattern conditioned on survivors can read more bullish than the true
population; the caveat travels with the Signal stamp.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# A fixed basket of ~30 liquid US large-caps + SPY.  Survivors (still trading 2026).
BASKET = [
    "SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "XOM", "BAC", "DIS", "KO", "PFE",
    "CSCO", "PEP", "WMT", "CVX", "ABT", "MRK", "INTC", "T", "VZ", "ORCL", "CMCSA",
]


# ---------------------------------------------------------------------------
# Synthetic panel — the deterministic offline core with a planted-edge knob
# ---------------------------------------------------------------------------
def _one_synthetic_tape(
    n_days: int,
    edge: float,
    daily_vol: float,
    tweezer_rate: float,
    start: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One daily OHLCV tape with optional planted tweezer-reversal structure.

    Base process is a random walk in log-price.  With probability ``tweezer_rate`` per
    bar we *plant* a tweezer event: we force the current and previous bar to share a
    near-equal extreme (low for a bottom, high for a top, chosen by the sign of the prior
    short-leg drift), and then nudge the following ``edge`` into the next few daily
    returns in the reversal direction.  ``edge = 0`` plants the geometry but no drift —
    the geometry alone must then read as noise.
    """
    cal = pd.bdate_range(start=start, periods=n_days)
    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = eps.copy()

    close = np.empty(n_days)
    close[0] = 100.0
    open_ = np.empty(n_days)
    hi = np.empty(n_days)
    lo = np.empty(n_days)
    open_[0] = 100.0

    # planted-drift buffer: future-return nudges injected by planted tweezers
    nudge = np.zeros(n_days)

    for i in range(1, n_days):
        r = log_ret[i] + nudge[i]
        close[i] = close[i - 1] * np.exp(r)
        open_[i] = close[i - 1]
        wick = abs(rng.normal(0.0, daily_vol * 0.5)) * close[i]
        hi[i] = max(open_[i], close[i]) + wick
        lo[i] = min(open_[i], close[i]) - wick

        # plant a tweezer with small probability, needs a prior bar to match
        if i >= 4 and rng.random() < tweezer_rate:
            recent = np.log(close[i - 1]) - np.log(close[i - 4])
            if recent < 0:  # short down-leg -> plant a tweezer BOTTOM (bullish)
                lo[i] = lo[i - 1]  # match the lows exactly
                for k in range(1, 6):
                    if i + k < n_days:
                        nudge[i + k] += edge / 5.0
            else:  # short up-leg -> plant a tweezer TOP (bearish)
                hi[i] = hi[i - 1]  # match the highs exactly
                for k in range(1, 6):
                    if i + k < n_days:
                        nudge[i + k] -= edge / 5.0

    hi[0] = max(open_[0], close[0]) * (1 + daily_vol * 0.5)
    lo[0] = min(open_[0], close[0]) * (1 - daily_vol * 0.5)
    vol = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": vol},
        index=pd.DatetimeIndex(cal, name="date"),
    )


def synthetic_panel(
    n_names: int = 8,
    n_days: int = 1500,
    edge: float = 0.0,
    daily_vol: float = 0.013,
    tweezer_rate: float = 0.01,
    start: str = "2018-01-02",
    seed: int = 409,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """A reproducible panel of daily OHLCV tapes with a planted tweezer-reversal knob.

    - ``edge = 0``   → geometry planted but no drift: the tweezer rule is a fair coin
                       against the unconditional base rate.
    - ``edge > 0``   → a genuine reversal follows each planted tweezer (bottoms drift up,
                       tops drift down) of cumulative magnitude ``edge`` over ~5 days.

    Returns ``(panel, truth)`` where ``panel`` maps a synthetic ticker to its OHLCV frame
    and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    panel = {}
    for j in range(n_names):
        panel[f"SYN{j:02d}"] = _one_synthetic_tape(
            n_days=n_days, edge=edge, daily_vol=daily_vol,
            tweezer_rate=tweezer_rate, start=start, rng=rng,
        )
    truth = {
        "edge": edge, "n_names": n_names, "n_days": n_days,
        "daily_vol": daily_vol, "tweezer_rate": tweezer_rate, "seed": seed,
    }
    return panel, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"tweezer_{safe}_1d.parquet")


def _download_one(ticker: str, period: str, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf  # lazy: only when we actually go to the network

    last_err = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker, period=period, interval="1d",
                auto_adjust=True, progress=False,
            )
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                bars = raw.rename(columns=str.lower)[
                    ["open", "high", "low", "close", "volume"]
                ]
                bars.index.name = "date"
                return bars
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"yfinance returned no daily bars for {ticker} after {retries} tries "
        f"({last_err})"
    )


def load_real(
    ticker: str,
    period: str = "10y",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker``, **cache-first**.

    Reads the cached parquet if present; on a cache miss it fetches from Yahoo (with
    retry/backoff) only when ``fetch=True`` and caches the result.  With ``fetch=False``
    and no cache it raises — so the reproducible core and any test run fully offline.

    Total-return adjusted (``auto_adjust=True``) daily bars.
    """
    path = _cache_path(ticker, cache_dir)
    if os.path.exists(path):
        bars = pd.read_parquet(path)
    elif fetch:
        bars = _download_one(ticker, period)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)
    else:
        raise FileNotFoundError(
            f"No cached daily tape for {ticker} at {path}. "
            f"Call load_real({ticker!r}, fetch=True) once to populate the cache."
        )

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index = pd.DatetimeIndex(bars.index, name="date")
    return bars


def load_basket(
    tickers: list[str] | None = None,
    period: str = "10y",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> dict[str, pd.DataFrame]:
    """Load the whole basket as ``{ticker: OHLCV}`` (cache-first; survivors caveat)."""
    tickers = tickers or BASKET
    return {t: load_real(t, period=period, fetch=fetch, cache_dir=cache_dir)
            for t in tickers}


def fingerprint(bars: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(bars["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]


def panel_fingerprint(panel: dict[str, pd.DataFrame]) -> str:
    """A single fingerprint over the whole basket (sorted by ticker)."""
    h = hashlib.sha1()
    for t in sorted(panel):
        h.update(t.encode())
        h.update(np.ascontiguousarray(panel[t]["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]

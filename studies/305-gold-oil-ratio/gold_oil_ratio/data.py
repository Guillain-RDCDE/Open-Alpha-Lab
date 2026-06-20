"""Data layer for Study 305 (Gold-Oil-Ratio).

Two tapes, one shape (a tz-naive daily price frame with columns ``gold``, ``oil``,
``equity`` and a constant short rate ``rf``):

- ``synthetic_daily`` — a *deterministic, offline* generator. A single knob, ``pred_r``,
  controls the only thing a gold/oil **timing switch** can possibly harvest: a forward link
  between the lagged gold/oil ratio *level* (relative to its own trailing mean) and the next
  day's equity return. ``pred_r = 0`` is the **null** — the equity tape is a pure random walk
  uncorrelated with the ratio, so a switch can do no better than chance. ``pred_r < 0`` plants
  the *folklore-consistent* control: a **high** gold/oil ratio (gold expensive vs oil →
  risk-off / contraction) precedes **lower** equity returns, so a "go to cash when the ratio
  is high" rule should add value. This is the study's null and positive control in one bottle.
- ``load_real`` / ``fetch_daily`` — the real tape. Cache-first: it reads cached parquets under
  ``_cache/`` (or, if available, the shared ``quantlab.data`` cache) and only touches the
  network on an explicit ``fetch=True``. Tickers: GLD (gold ETF), USO/CL=F (oil), SPY (equity),
  and ``^IRX`` (13-week T-bill yield) for the cash leg. Everything is graceful: a missing cache
  raises a clear ``FileNotFoundError`` rather than silently going to the network.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the switch is set
from the ratio known at the close of *t* and earns the return of *t+1*, one ``shift``).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# Real-tape tickers (Yahoo!): gold ETF, oil ETF, equity ETF, 13-week T-bill yield.
TICKERS = ["GLD", "USO", "SPY", "^IRX"]
START_DATE = "2006-04-10"  # GLD/USO common history starts here

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 4000,
    pred_r: float = 0.0,
    annual_vol: float = 0.16,
    rf_annual: float = 0.02,
    lookback: int = 60,
    start: str = "2006-04-10",
    seed: int = 305,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily gold/oil/equity tape with a controllable timing link.

    Three price series are simulated in daily log-returns:

    - ``gold`` and ``oil`` — independent random walks (log-vol ≈ ``annual_vol``). Their
      ratio ``gold/oil`` is the predictor variable; its deviation from a trailing
      ``lookback``-day mean is the *signal* the switch reads.
    - ``equity`` — the next day's log-return is ``pred_r * z_{t-1} + noise``, where
      ``z_{t-1}`` is the lagged standardised gold/oil-ratio deviation. ``rf`` is a constant
      daily risk-free rate (the cash leg of the switch).

    ``pred_r`` is the *only* forecasting structure in the tape:

    - ``pred_r = 0``   → no predictability; the switch is a fair coin (the **null**).
    - ``pred_r < 0``   → folklore-consistent: a high ratio (risk-off) precedes lower equity
      returns, so "to cash when the ratio is high" adds value (the **positive control**).
    - ``pred_r > 0``   → the inverted link (a high ratio precedes higher returns).

    Returns ``(daily, truth)`` where ``truth`` records the planted parameters. Dates are
    consecutive weekdays from ``start`` (kept well under any ns-Timestamp overflow horizon).
    """
    rng = np.random.default_rng(seed)
    dv = annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)

    gold_r = rng.normal(0.0, dv, n_days)
    oil_r = rng.normal(0.0, dv * 1.7, n_days)  # oil is the more volatile leg

    gold_p = 60.0 * np.exp(np.cumsum(gold_r))
    oil_p = 40.0 * np.exp(np.cumsum(oil_r))
    ratio = gold_p / oil_p
    log_ratio = np.log(ratio)

    # Standardised deviation of the log ratio from its trailing mean (the signal).
    s = pd.Series(log_ratio)
    mu = s.rolling(lookback, min_periods=lookback).mean()
    sd = s.rolling(lookback, min_periods=lookback).std(ddof=1)
    z = ((s - mu) / sd).to_numpy()

    equity_noise = rng.normal(0.0, dv, n_days)
    equity_r = np.zeros(n_days)
    for t in range(1, n_days):
        zt = z[t - 1]
        if not np.isfinite(zt):
            zt = 0.0
        equity_r[t] = pred_r * zt + equity_noise[t]

    equity_p = 100.0 * np.exp(np.cumsum(equity_r))

    dates = pd.bdate_range(start=start, periods=n_days)
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR

    daily = pd.DataFrame(
        {
            "gold": gold_p,
            "oil": oil_p,
            "equity": equity_p,
            "rf": np.full(n_days, rf_daily),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "pred_r": pred_r,
        "annual_vol": annual_vol,
        "rf_annual": rf_annual,
        "lookback": lookback,
        "n_days": n_days,
        "seed": seed,
    }
    return daily, truth


# ---------------------------------------------------------------------------
# Real tape — cache-first, network only on explicit fetch
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"daily_{safe}.parquet")


def fetch_daily(
    ticker: str,
    start: str = START_DATE,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily price history for ``ticker``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as a
    parquet under ``_cache/``). A missing cache raises a clear ``FileNotFoundError`` rather
    than silently hitting the network, so the test-suite and reproducible core stay offline.
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
            ticker, start=start, interval="1d", auto_adjust=True, progress=False
        )
        if raw.empty:
            raise RuntimeError(f"yfinance returned no daily bars for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def load_real(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
    oil_ticker: str = "USO",
) -> pd.DataFrame:
    """Assemble the real gold/oil/equity/rf frame, cache-first with graceful fallback.

    Returns a daily frame with columns ``gold``, ``oil``, ``equity``, ``rf`` aligned on the
    common business-day index (forward-filled at most 3 days over holidays before being
    dropped). ``rf`` is the ^IRX 13-week T-bill *daily* rate (annual yield / 252); if the
    ^IRX cache is absent it falls back to a flat 2%/yr cash leg so the frame still assembles.

    The cache is consulted first (``_cache/`` parquets, then, if importable, the shared
    ``quantlab.data`` daily cache); the network is touched only on ``fetch=True``.
    """
    gold = fetch_daily("GLD", fetch=fetch, cache_dir=cache_dir)["close"].rename("gold")
    oil = fetch_daily(oil_ticker, fetch=fetch, cache_dir=cache_dir)["close"].rename("oil")
    equity = fetch_daily("SPY", fetch=fetch, cache_dir=cache_dir)["close"].rename("equity")

    # Cash leg: ^IRX is an *annualised yield in percent*. Convert to a daily simple rate.
    try:
        irx = fetch_daily("^IRX", fetch=fetch, cache_dir=cache_dir)["close"]
        rf = (irx / 100.0 / TRADING_DAYS_PER_YEAR).rename("rf")
    except (FileNotFoundError, KeyError):
        rf = pd.Series(0.02 / TRADING_DAYS_PER_YEAR, index=equity.index, name="rf")

    df = pd.concat([gold, oil, equity, rf], axis=1)
    df["rf"] = df["rf"].ffill().fillna(0.02 / TRADING_DAYS_PER_YEAR)
    df = df.ffill(limit=3).dropna(subset=["gold", "oil", "equity"])
    return df


def have_real_cache(cache_dir: str = DEFAULT_CACHE, oil_ticker: str = "USO") -> bool:
    """True iff the minimal real tape (gold, oil, equity) is cached locally."""
    needed = ["GLD", oil_ticker, "SPY"]
    return all(os.path.exists(_cache_path(t, cache_dir)) for t in needed)


def fingerprint(df: pd.DataFrame, col: str = "equity") -> str:
    """A short content fingerprint of the panel (one column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]

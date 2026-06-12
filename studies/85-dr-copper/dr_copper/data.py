"""Data layer for Study 85 (Dr-Copper).

Two tapes, one shape (a daily price frame, weekly resampled for regression):

- ``synthetic_daily`` — a *deterministic, offline* generator. A predictive-strength
  knob (``pred_r``) controls how much the copper/gold ratio change forecasts the
  next-period equity return. ``pred_r=0`` is the null model — pure random walks with
  no cross-series predictability — so a test can assert predictive regressions beat
  the naive baseline *only* when we plant a real link, and not otherwise. This is the
  study's null in a bottle.
- ``fetch_daily`` — the real Yahoo! daily close history (``yfinance``), cache-only by
  default so the test-suite and the reproducible core never touch the network. We pull
  HG=F (copper front-month), GC=F (gold front-month), ^TNX (10y Treasury yield), and
  ^GSPC (S&P 500 index) back to 2000-01-01, giving ~25 years of daily data — a long
  enough tape to estimate slow macro regressions.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the signal is
the *lagged* ratio-change, the return measured over the *forward* period).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKERS = ["HG=F", "GC=F", "^GSPC", "^TNX"]
START_DATE = "2000-01-01"

# Weekly resampling anchor used throughout
WEEKLY_ANCHOR = "W-FRI"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 20,
    pred_r: float = 0.0,
    annual_vol: float = 0.18,
    start: str = "2004-01-02",
    seed: int = 85,
) -> tuple[pd.DataFrame, dict]:
    """Reproducible daily price frame with a controllable cross-series predictive link.

    Four series are simulated in daily log-returns:

    - ``copper`` and ``gold`` — independent random walks (log-vol ≈ ``annual_vol``).
      Their ratio ``cu_au = log(copper/gold)`` is the predictor variable.
    - ``equity`` — next-day log-return is ``pred_r * delta_cu_au_lag1 + noise``. When
      ``pred_r=0`` the equity series is a martingale uncorrelated with the ratio.
    - ``yield10`` — simulated as a random walk in yield-change space.

    ``pred_r`` is the *only* forecasting structure in the tape:

    - ``pred_r = 0``   → no predictability; any regression should fail.
    - ``pred_r > 0``   → a genuine copper-leads-equity link the regression should find.
    - ``pred_r < 0``   → an inverse link (copper up → equity down).

    Dates are consecutive weekdays from ``start``. Returns ``(daily, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * 252

    # Daily log-vol from annual via sqrt(252)
    dv = annual_vol / np.sqrt(252)

    copper_r = rng.normal(0.0, dv, n_days)
    gold_r = rng.normal(0.0, dv, n_days)
    yield_r = rng.normal(0.0, dv * 0.1, n_days)  # yields move less

    # Copper/gold log-ratio changes
    delta_cu_au = copper_r - gold_r
    # Equity: forward return predicted by lagged ratio change + noise
    equity_noise = rng.normal(0.0, dv, n_days)
    equity_r = np.zeros(n_days)
    for t in range(1, n_days):
        equity_r[t] = pred_r * delta_cu_au[t - 1] + equity_noise[t]

    # Cumulative to prices
    dates = pd.bdate_range(start=start, periods=n_days)
    copper_p = 4.50 * np.exp(np.cumsum(copper_r))
    gold_p = 1900.0 * np.exp(np.cumsum(gold_r))
    equity_p = 3000.0 * np.exp(np.cumsum(equity_r))
    yield10 = 3.0 + 100.0 * np.cumsum(yield_r)
    yield10 = np.clip(yield10, 0.1, 15.0)  # keep realistic bounds

    daily = pd.DataFrame(
        {
            "copper": copper_p,
            "gold": gold_p,
            "equity": equity_p,
            "yield10": yield10,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "pred_r": pred_r,
        "annual_vol": annual_vol,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
    }
    return daily, truth


# ---------------------------------------------------------------------------
# Real tape — Yahoo daily close, cache-only by default
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
    """Real daily OHLCV for ``ticker`` back to ``start``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). With ~25 years of daily bars (for macro/slow signals)
    this is a much better-powered tape than the 5-minute studies; the limiting factor is
    changes in the copper/gold relationship across regimes, not sample length.
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
        if "close" not in bars.columns:
            # Some tickers (e.g. ^TNX) come with slightly different column names
            bars = raw.copy()
            bars.columns = [str(c).lower() for c in bars.columns]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def load_panel(fetch: bool = False, cache_dir: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Load all four real tickers into a dict, using cache or network.

    Returns ``{"HG=F": bars, "GC=F": bars, "^GSPC": bars, "^TNX": bars}``
    where each ``bars`` is a daily OHLCV (or price-only for index/yield) frame.
    """
    return {t: fetch_daily(t, fetch=fetch, cache_dir=cache_dir) for t in TICKERS}


def build_ratio_frame(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align the four tickers and compute the copper/gold ratio.

    Returns a daily frame with columns:
    - ``copper``  — HG=F close ($/lb)
    - ``gold``    — GC=F close ($/troy oz)
    - ``cu_au``   — copper/gold price ratio
    - ``equity``  — ^GSPC close
    - ``yield10`` — ^TNX close (10y Treasury yield, %)

    All series aligned on business days where all four have a valid close. Series are
    forward-filled at most 3 days over holidays/missing prints before being dropped.
    """
    copper = panel["HG=F"]["close"].rename("copper")
    gold = panel["GC=F"]["close"].rename("gold")

    # ^GSPC is an index; yfinance gives 'close'
    equity = panel["^GSPC"]["close"].rename("equity")

    # ^TNX comes as a yield level (percent); keep raw
    tnx = panel["^TNX"]["close"].rename("yield10")

    df = pd.concat([copper, gold, equity, tnx], axis=1)
    df = df.ffill(limit=3).dropna()
    df["cu_au"] = df["copper"] / df["gold"]
    return df


def fingerprint(df: pd.DataFrame, col: str = "equity") -> str:
    """A short content fingerprint of the panel (equity column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]

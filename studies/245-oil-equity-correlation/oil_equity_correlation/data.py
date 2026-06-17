"""Data layer for Study 245 (Oil-Equity-Correlation).

Two tapes, one shape (a daily price frame, weekly resampled for regression):

- ``synthetic_daily`` — a *deterministic, offline* generator. A predictive-strength
  knob (``pred_r``) controls how much the crude-oil log-return forecasts the
  next-period equity return. ``pred_r=0`` is the null model — pure random walks with
  no cross-series predictability — so a test can assert predictive regressions beat
  the naive baseline *only* when we plant a real link, and not otherwise. This is the
  study's null in a bottle.
- ``fetch_daily`` — the real Yahoo! daily close history (``yfinance``), cache-only by
  default so the test-suite and the reproducible core never touch the network. We pull
  USO (crude oil ETF), CL=F (WTI crude front-month), and SPY (S&P 500 ETF) back to
  2006-04-10 (USO inception date), giving ~20 years of daily data — a long enough tape
  to estimate slow macro regressions.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the signal is
the *lagged* oil return, the equity return measured over the *forward* period).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKERS = ["USO", "CL=F", "SPY"]
START_DATE = "2006-04-10"  # USO inception date

# Weekly resampling anchor used throughout
WEEKLY_ANCHOR = "W-FRI"


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 20,
    pred_r: float = 0.0,
    annual_vol: float = 0.25,
    start: str = "2006-04-10",
    seed: int = 245,
) -> tuple[pd.DataFrame, dict]:
    """Reproducible daily price frame with a controllable cross-series predictive link.

    Three series are simulated in daily log-returns:

    - ``oil`` — random walk (log-vol ≈ ``annual_vol``).
    - ``equity`` — next-day log-return is ``pred_r * oil_lag1 + noise``. When
      ``pred_r=0`` the equity series is a martingale uncorrelated with oil.
    - ``vix`` — simulated as a mean-reverting proxy (for context only; not a regressor).

    ``pred_r`` is the *only* forecasting structure in the tape:

    - ``pred_r = 0``   → no predictability; any regression should fail.
    - ``pred_r > 0``   → a genuine oil-leads-equity link the regression should find.
    - ``pred_r < 0``   → an inverse link (oil up → equity down).

    Dates are consecutive weekdays from ``start``. Returns ``(daily, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * 252

    # Daily log-vol from annual via sqrt(252)
    dv = annual_vol / np.sqrt(252)

    oil_r = rng.normal(0.0, dv, n_days)
    # Equity: forward return predicted by lagged oil return + noise
    equity_noise = rng.normal(0.0, dv * 0.8, n_days)
    equity_r = np.zeros(n_days)
    for t in range(1, n_days):
        equity_r[t] = pred_r * oil_r[t - 1] + equity_noise[t]

    # VIX: mean-reverting, higher when equity falls
    vix_noise = rng.normal(0.0, 0.02, n_days)
    vix = np.zeros(n_days)
    vix[0] = 20.0
    for t in range(1, n_days):
        vix[t] = max(5.0, vix[t - 1] * 0.99 - 3.0 * equity_r[t] + vix_noise[t])

    # Cumulative to prices
    dates = pd.bdate_range(start=start, periods=n_days)
    oil_p = 60.0 * np.exp(np.cumsum(oil_r))
    equity_p = 300.0 * np.exp(np.cumsum(equity_r))

    daily = pd.DataFrame(
        {
            "oil": oil_p,
            "equity": equity_p,
            "vix": vix,
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
    a parquet under ``_cache/``). With ~20 years of daily bars (for macro/slow signals)
    this is a well-powered tape; the limiting factor is changes in the oil/equity
    relationship across regimes (supply shocks, COVID, ESG), not sample length.
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
            bars = raw.copy()
            bars.columns = [str(c).lower() for c in bars.columns]
        bars.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    return bars


def load_panel(fetch: bool = False, cache_dir: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Load all three real tickers into a dict, using cache or network.

    Returns ``{"USO": bars, "CL=F": bars, "SPY": bars}``
    where each ``bars`` is a daily OHLCV frame.
    """
    return {t: fetch_daily(t, fetch=fetch, cache_dir=cache_dir) for t in TICKERS}


def build_panel_frame(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align the three tickers and return a daily price frame.

    Returns a daily frame with columns:
    - ``oil``    — USO close (crude oil ETF, $/share)
    - ``cl``     — CL=F close (WTI crude front-month, $/bbl)
    - ``equity`` — SPY close (S&P 500 ETF, $/share)

    All series aligned on business days where all three have a valid close. Series are
    forward-filled at most 3 days over holidays/missing prints before being dropped.
    """
    uso = panel["USO"]["close"].rename("oil")
    cl = panel["CL=F"]["close"].rename("cl")
    spy = panel["SPY"]["close"].rename("equity")

    df = pd.concat([uso, cl, spy], axis=1)
    df = df.ffill(limit=3).dropna()
    return df


def fingerprint(df: pd.DataFrame, col: str = "equity") -> str:
    """A short content fingerprint of the panel (equity column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]

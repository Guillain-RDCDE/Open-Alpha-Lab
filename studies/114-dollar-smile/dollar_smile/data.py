"""Data layer for Study 114 (Dollar-Smile).

Two tapes, one shape (a daily price frame, weekly resampled for regression):

- ``synthetic_daily`` — a *deterministic, offline* generator. A predictive-strength
  knob (``pred_r``) controls how much the dollar-index change forecasts the next-period
  equity return. ``pred_r=0`` is the null model — pure random walks with no cross-series
  predictability — so a test can assert predictive regressions beat the naive baseline
  *only* when we plant a real link, and not otherwise. This is the study's null in a
  bottle. A second knob ``contemp_r`` governs the contemporaneous (same-period)
  correlation to model the well-known coincident link.
- ``fetch_daily`` — the real Yahoo! daily close history (``yfinance``), cache-only by
  default so the test-suite and the reproducible core never touch the network. We pull
  DX-Y.NYB (Dollar Index), SPY (US large-cap equity), EEM (EM equity ETF), and UUP
  (dollar ETF, an alternative) back to 2004-01-01 (EEM inception), giving ~22 years of
  daily data — a long enough tape to estimate macro regressions.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the signal is
the *lagged* dollar-index change, the return measured over the *forward* period).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKERS = ["DX-Y.NYB", "SPY", "EEM"]
START_DATE = "2004-01-01"  # EEM inception


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 20,
    pred_r: float = 0.0,
    contemp_r: float = 0.0,
    annual_vol: float = 0.12,
    start: str = "2004-01-02",
    seed: int = 114,
) -> tuple[pd.DataFrame, dict]:
    """Reproducible daily price frame with a controllable cross-series predictive link.

    Three series are simulated in daily log-returns:

    - ``dollar`` — a random walk (log-vol ≈ ``annual_vol / 3``). Its daily change is
      the predictor variable (dollar up = positive).
    - ``spy`` — next-day log-return is ``pred_r * delta_dollar_lag1 + contemp_r *
      delta_dollar_t + noise``. When ``pred_r=0`` and ``contemp_r=0`` the equity series
      is a martingale uncorrelated with the dollar.
    - ``eem`` — same structure as spy but with ``pred_r * 1.5`` to model the folk claim
      that EM is more dollar-sensitive than the US market.

    ``pred_r`` is the *only forecastable* structure in the tape:

    - ``pred_r = 0``   → no predictability; any regression should fail.
    - ``pred_r < 0``   → a genuine strong-dollar-hurts-equity link (negative beta).
    - ``pred_r > 0``   → an inverse link (strong dollar helps equity).

    Dates are consecutive weekdays from ``start``. Returns ``(daily, truth)`` where
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * 252

    # Daily log-vol from annual via sqrt(252)
    dv_eq = annual_vol / np.sqrt(252)
    dv_usd = (annual_vol / 3.0) / np.sqrt(252)  # dollar index is less volatile

    dollar_r = rng.normal(0.0, dv_usd, n_days)
    spy_noise = rng.normal(0.0, dv_eq, n_days)
    eem_noise = rng.normal(0.0, dv_eq * 1.3, n_days)  # EM is more volatile

    spy_r = np.zeros(n_days)
    eem_r = np.zeros(n_days)
    for t in range(1, n_days):
        # contemporaneous link (same day): dollar moves with / against equity now
        # predictive link (lag-1): lagged dollar predicts next-period equity
        spy_r[t] = (
            contemp_r * dollar_r[t]
            + pred_r * dollar_r[t - 1]
            + spy_noise[t]
        )
        eem_r[t] = (
            contemp_r * 1.5 * dollar_r[t]
            + pred_r * 1.5 * dollar_r[t - 1]
            + eem_noise[t]
        )

    # Cumulative to price index (start 100)
    dates = pd.bdate_range(start=start, periods=n_days)
    dollar_p = 100.0 * np.exp(np.cumsum(dollar_r))
    spy_p = 100.0 * np.exp(np.cumsum(spy_r))
    eem_p = 100.0 * np.exp(np.cumsum(eem_r))

    daily = pd.DataFrame(
        {"dollar": dollar_p, "spy": spy_p, "eem": eem_p},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "pred_r": pred_r,
        "contemp_r": contemp_r,
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
    safe = ticker.replace("=", "").replace("^", "").replace("/", "").replace("-", "")
    return os.path.join(cache_dir, f"daily_{safe}.parquet")


def fetch_daily(
    ticker: str,
    start: str = START_DATE,
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real daily OHLCV for ``ticker`` back to ``start``; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached as
    a parquet under ``_cache/``). With ~22 years of daily bars (for macro/slow signals)
    this is a well-powered tape; the limiting factor is shifts in dollar-equity correlations
    across regimes, not sample length.
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


def load_panel(fetch: bool = False, cache_dir: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Load all three real tickers into a dict, using cache or network.

    Returns ``{"DX-Y.NYB": bars, "SPY": bars, "EEM": bars}`` where each ``bars`` is a
    daily OHLCV frame.
    """
    return {t: fetch_daily(t, fetch=fetch, cache_dir=cache_dir) for t in TICKERS}


def build_panel_frame(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Align the three tickers and compute the dollar-index close series.

    Returns a daily frame with columns:
    - ``dollar``  — DX-Y.NYB close (USD Index level)
    - ``spy``     — SPY close (US equity ETF, adjusted)
    - ``eem``     — EEM close (EM equity ETF, adjusted)

    All series aligned on business days where all three have a valid close. Series are
    forward-filled at most 3 days over holidays/missing prints before being dropped.
    """
    dollar = panel["DX-Y.NYB"]["close"].rename("dollar")
    spy = panel["SPY"]["close"].rename("spy")
    eem = panel["EEM"]["close"].rename("eem")

    df = pd.concat([dollar, spy, eem], axis=1)
    df = df.ffill(limit=3).dropna()
    return df


def fingerprint(df: pd.DataFrame, col: str = "spy") -> str:
    """A short content fingerprint of the panel (spy column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]

"""Data layer for Study 865 — Credit → Equity Lead-Lag.

The claim under test — **"credit leads equity"**: high-yield credit, measured
*duration-hedged* as **HYG in excess of IEF**, is said to *turn before* the stock market
at risk-on/off inflections. The self-contained, weekly form of the folk rule is a
**Granger-style lead-lag**: does the trailing 1-4-week HY-excess return **predict the
NEXT week's SPY return**? And can a timing overlay that goes risk-on when credit is
trending up beat a buy-and-hold?

Two tapes, one shape (a tz-naive daily adjusted-close frame for four ETFs, resampled to
weekly by the strategy layer):

* **Real tape — ``fetch`` / ``load_panel``.** Daily total-return (``auto_adjust=True``)
  closes for HYG, IEF, LQD, SPY pulled with ``yfinance`` and cached as a single parquet
  under this study's OWN ``_cache/``. ``fetch()`` (network) runs once to build the cache,
  retrying up to four times with a short sleep; ``load_panel`` / ``load_series`` read the
  cached parquet **OFFLINE** (no yfinance import). ``AS_OF = 2026-06-30`` drops the
  partial current calendar month.

* **Synthetic world — ``synthetic_panel`` — the positive control.** A deterministic,
  seeded generator with a TUNABLE knob ``edge``: a persistent latent risk-on factor drives
  the duration-hedged HY-excess return and — only when ``edge > 0`` — the *next* week's SPY
  return (a genuine one-week lead of credit over equity). ``edge = 0`` is the null (credit
  still moves, but leads nothing), so the lead-lag test must find nothing.

The four tickers:

* **HYG** — iShares iBoxx $ High Yield Corporate Bond ETF (high-yield credit).
* **IEF** — iShares 7-10 Year Treasury Bond ETF (the duration hedge).
* **LQD** — iShares iBoxx $ Investment Grade Corporate Bond ETF (an IG cross-check).
* **SPY** — SPDR S&P 500 ETF (the equity leg credit is said to lead).

The offline path is pure numpy + pandas + stdlib.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
CACHE_PATH = os.path.join(CACHE_DIR, "etf_panel_daily.parquet")

TICKERS = ["HYG", "IEF", "LQD", "SPY"]
START = "2007-05-01"        # HYG/LQD inception is 2007-04; start just after
AS_OF = "2026-06-30"        # last complete calendar month at publication

__all__ = [
    "TICKERS", "START", "AS_OF", "CACHE_DIR", "CACHE_PATH",
    "fetch", "have_real", "load_panel", "load_series", "synthetic_panel",
    "fingerprint",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily, cached once, read offline thereafter
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str | None = None, retries: int = 4) -> str:
    """Download the four-ETF total-return close panel; cache it as one parquet.

    Network only — retries up to ``retries`` times with a short sleep on failure.
    Returns the cache path. Never imported by the offline notebook cells.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    os.makedirs(CACHE_DIR, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                TICKERS, start=start, end=end, interval="1d",
                auto_adjust=True, progress=False, group_by="column",
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned an empty frame")
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].copy()
            else:  # single ticker fallback
                close = raw[["Close"]].copy()
                close.columns = TICKERS[:1]
            close = close[[t for t in TICKERS if t in close.columns]]
            close = close.dropna(how="all").ffill().dropna()
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            close.index.name = "date"
            if len(close) < 500 or any(t not in close.columns for t in TICKERS):
                raise RuntimeError(f"panel too short / missing tickers: {close.shape}")
            close.to_parquet(CACHE_PATH)
            return CACHE_PATH
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2.0 + attempt)
    raise RuntimeError(f"fetch failed after {retries} attempts: {last_err}")


def have_real() -> bool:
    return os.path.exists(CACHE_PATH)


def load_panel(asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily total-return closes for [HYG, IEF, LQD, SPY], sliced to ``<= asof``.

    Reads the parquet directly — OFFLINE, no yfinance import. Columns are the tickers
    (upper-case); the DatetimeIndex is tz-naive daily.
    """
    df = pd.read_parquet(CACHE_PATH)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[df.index <= pd.Timestamp(asof)]
    return df.sort_index()


def load_series(ticker: str, asof: str = AS_OF) -> pd.Series:
    """A single cached total-return close series (OFFLINE)."""
    return load_panel(asof=asof)[ticker]


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of a prices frame, for the as-of stamp."""
    arr = np.ascontiguousarray(df.to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — planted credit-leads-equity lead-lag (the control)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    edge: float = 0.0,
    seed: int = 865,
    n_days: int = 3000,
    start: str = "2010-01-04",
    rate_vol: float = 0.0035,
    credit_beta: float = 0.9,
    credit_vol: float = 0.004,
    equity_vol: float = 0.011,
    equity_drift: float = 0.05 / 252,
    factor_rho: float = 0.97,
) -> pd.DataFrame:
    """Deterministic seeded [HYG, IEF, LQD, SPY] close panel with a TUNABLE planted
    **one-week lead** of duration-hedged credit over equity.

    A persistent latent **risk-on factor** ``f[t]`` (AR(1), autocorrelation ``factor_rho``)
    drives the *duration-hedged credit* excess return contemporaneously and — only when
    ``edge > 0`` — the SPY return **one trading week later**::

        f[t]      = factor_rho * f[t-1] + eps
        ief_r[t]  = base treasury return (rate factor)
        hyg_r[t]  = ief_r[t] + credit_beta * credit_vol * f[t] + credit noise
        lqd_r[t]  = ief_r[t] + 0.4 * credit_beta * credit_vol * f[t] + noise
        spy_r[t]  = equity_drift + edge * f[t-5] + equity noise      # credit LEADS by ~1wk

    So when the trailing HY-excess trend is positive (``f`` high and persistent) the SPY
    return **the following week** is nudged up (with ``edge > 0``) — exactly the "credit
    leads equity" lead-lag under test. ``edge = 0`` is the null: credit still varies but
    leads nothing, so the Granger-style test must find no predictability.
    Business-day index; span well below the pandas ns-timestamp horizon (rule: OOB-safe).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)  # daily, n_days <= ~10000 -> OOB-safe

    innov_sd = np.sqrt(1.0 - factor_rho ** 2)
    f = np.empty(n_days)
    f[0] = rng.normal(0.0, 1.0)
    eps = rng.normal(0.0, innov_sd, n_days)
    for t in range(1, n_days):
        f[t] = factor_rho * f[t - 1] + eps[t]

    ief_r = rng.normal(0.0001, rate_vol, n_days)
    hyg_r = ief_r + credit_beta * credit_vol * f + rng.normal(0.0, credit_vol, n_days)
    lqd_r = ief_r + 0.4 * credit_beta * credit_vol * f + rng.normal(0.0, credit_vol * 0.6, n_days)

    # credit LEADS equity by ~one trading week: f[t-5] -> spy_r[t]
    f_lead = np.concatenate([np.zeros(5), f[:-5]])
    spy_r = equity_drift + edge * f_lead + rng.normal(0.0, equity_vol, n_days)

    def _px(r):
        return 100.0 * np.cumprod(1.0 + r)

    return pd.DataFrame(
        {"HYG": _px(hyg_r), "IEF": _px(ief_r), "LQD": _px(lqd_r), "SPY": _px(spy_r)},
        index=idx,
    )

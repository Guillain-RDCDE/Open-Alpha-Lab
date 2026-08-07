"""Data layer for Study 825 — Oil Predicts Equities.

The claim under test (Driesprong, Jacobsen & Maat 2008, *"Striking Oil: Another Puzzle?"*):
a change in the **oil price this month** predicts the **stock-market return next month**,
*negatively* — an oil shock diffuses slowly into equities, so a rising oil price is a
headwind for stocks one month out. The tradable reading is a monthly predictive regression

    r_equity[t+1] = alpha + beta * r_oil[t] + eps ,   beta expected < 0 .

Two tapes, one shape (a daily adjusted-close frame, resampled to month-end for the
regression).

* **Real tape — two liquid ETFs.** Daily adjusted close for **USO** (United States Oil
  Fund, the crude-oil proxy) and **SPY** (S&P 500 ETF, the equity proxy), pulled with
  yfinance (``auto_adjust=True``, total-return) back to USO's 2006-04-10 inception. The
  aligned two-column frame is cached as a single parquet under this study's OWN
  ``_cache/``. ``fetch()`` (network, retried) builds the cache once; ``load_series()`` /
  ``load_panel()`` read it back OFFLINE (no yfinance import).

* **Synthetic world — the positive control.** A deterministic, seeded daily generator
  (``synthetic_panel``) with a TUNABLE knob ``edge``: the equity month-``t+1`` return is
  ``-edge * oil_ret[t] + noise``, so ``edge > 0`` plants exactly the Driesprong *negative*
  oil→equity predictive link and ``edge = 0`` is the null (oil and equities are
  independent random walks, the regression must find nothing). Used only to prove the
  regression machinery is unbiased — never to support a real-tape stamp.

Monthly frequency is the whole point: the Driesprong effect is a *slow-diffusion* monthly
signal, distinct from the contemporaneous same-period co-movement (study 245) and from
crude's calendar seasonality (study 226). No look-ahead is baked in here — the discipline
lives in ``strategy.py`` (the predictor is the trailing-month oil return, the target the
*forward*-month equity return).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKERS = ["USO", "SPY"]
START = "2006-04-10"        # USO inception date
AS_OF = "2026-06-30"        # last complete calendar month at publication
CACHE_PATH = os.path.join(CACHE_DIR, "oil_equity_USO_SPY.parquet")

__all__ = [
    "TICKERS", "START", "AS_OF", "CACHE_DIR", "CACHE_PATH",
    "fetch", "have_real", "load_series", "load_panel", "fingerprint",
    "synthetic_panel", "synthetic_series",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily adjusted close, cached under the study's _cache/
# --------------------------------------------------------------------------- #
def fetch(start: str = START, retries: int = 4, pause: float = 3.0) -> pd.DataFrame:
    """Download USO + SPY daily adjusted close and cache the aligned frame.

    Network is touched only here. yfinance is flaky, so we retry up to ``retries``
    times with a short ``pause`` between attempts before giving up. On success the
    two-column daily frame (index=date, columns=``USO``, ``SPY``) is written to
    ``CACHE_PATH`` as a parquet and returned.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    os.makedirs(CACHE_DIR, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                TICKERS, start=start, interval="1d",
                auto_adjust=True, progress=False, threads=False,
            )
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned an empty frame")
            # Extract the Close panel across tickers.
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].copy()
            else:  # single ticker degenerate case
                close = raw[["Close"]].copy()
                close.columns = [TICKERS[0]]
            close = close[[c for c in TICKERS if c in close.columns]]
            if close.shape[1] < len(TICKERS):
                raise RuntimeError(f"missing tickers, got columns {list(close.columns)}")
            close.index = pd.DatetimeIndex(close.index).tz_localize(None)
            close.index.name = "date"
            close = close.sort_index().ffill(limit=3).dropna()
            if len(close) < 100:
                raise RuntimeError(f"too few aligned rows: {len(close)}")
            close.to_parquet(CACHE_PATH)
            return close
        except Exception as exc:  # noqa: BLE001 — retry any failure
            last_err = exc
            if attempt < retries:
                time.sleep(pause)
    raise RuntimeError(f"fetch failed after {retries} attempts: {last_err}")


def have_real() -> bool:
    """True iff the real cache parquet exists (offline-readable)."""
    return os.path.exists(CACHE_PATH)


def load_series(asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily adjusted-close frame (columns ``USO``, ``SPY``), sliced to ``[, asof]``.

    Reads the parquet directly — OFFLINE, no yfinance import. Drops the partial current
    month by cutting at ``asof`` (the last complete calendar month at publication).
    """
    df = pd.read_parquet(CACHE_PATH)
    df.index = pd.DatetimeIndex(df.index)
    hi = pd.Timestamp(asof)
    return df[df.index <= hi].sort_index()


def load_panel(asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached tape as ``{ticker: DataFrame[Close]}`` — mirrors the house panel shape."""
    df = load_series(asof)
    return {t: df[[t]].rename(columns={t: "Close"}) for t in TICKERS if t in df.columns}


def fingerprint(df: pd.DataFrame, col: str = "SPY") -> str:
    """Short content fingerprint of the panel (``col`` column) for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df[col].to_numpy()).tobytes())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic world — planted negative oil->equity predictive link (positive control)
# --------------------------------------------------------------------------- #
def synthetic_series(
    edge: float = 0.0,
    seed: int = 825,
    n_days: int = 5040,          # ~20 years of business days
    start: str = "2006-04-10",
    oil_annual_vol: float = 0.32,
    eq_annual_vol: float = 0.16,
    eq_drift: float = 0.07 / 252,
) -> pd.DataFrame:
    """Deterministic seeded daily USO/SPY tape with a TUNABLE planted predictive link.

    Oil is a pure random walk. The equity daily return is a martingale **plus** a small
    drag proportional to *last month's* oil return: on each day ``t`` we carry the
    trailing 21-day oil return ``m`` and add ``-edge * m / 21`` to the equity daily
    return, so the *forward monthly* equity return loads on the *trailing monthly* oil
    return with slope ``≈ -edge`` — exactly the Driesprong pattern. ``edge = 0`` is the
    null: oil and equities are independent random walks, and the predictive regression
    must find nothing. Business-day index; span well below the ns-timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    dv_oil = oil_annual_vol / np.sqrt(252)
    dv_eq = eq_annual_vol / np.sqrt(252)

    oil_r = rng.normal(0.0, dv_oil, n_days)
    eq_noise = rng.normal(0.0, dv_eq, n_days)

    # Trailing 21-day oil return, known at t-1, drives the day-t equity return.
    cum = np.concatenate([[0.0], np.cumsum(oil_r)])
    trail = np.zeros(n_days)
    trail[21:] = cum[21:n_days] - cum[:n_days - 21]      # oil log-return over (t-21, t-1]
    eq_r = eq_drift + eq_noise - edge * trail / 21.0

    idx = pd.bdate_range(start=start, periods=n_days)
    oil_p = 60.0 * np.exp(np.cumsum(oil_r))
    eq_p = 130.0 * np.exp(np.cumsum(eq_r))
    return pd.DataFrame({"USO": oil_p, "SPY": eq_p},
                        index=pd.DatetimeIndex(idx, name="date"))


def synthetic_panel(edge: float = 0.0, seed: int = 825, **kw) -> dict[str, pd.DataFrame]:
    """``synthetic_series`` reshaped to ``{ticker: DataFrame[Close]}`` (house panel shape)."""
    df = synthetic_series(edge=edge, seed=seed, **kw)
    return {t: df[[t]].rename(columns={t: "Close"}) for t in TICKERS}

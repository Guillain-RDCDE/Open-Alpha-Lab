"""Data layer for Study 353 -- Smart-Money-Concepts (ICT order blocks & fair-value gaps).

Two sources, both offline-friendly:

* **Real tape.** SPY daily OHLC bars (yfinance public download, no key) cached under
  ``_cache/spy_daily.csv`` -- columns ``Date, Open, High, Low, Close, Volume``. This is the
  tape the ICT/"Smart Money Concepts" YouTube meme is most often demonstrated on. We treat
  each bar as a candle and detect three-bar **fair-value gaps** (FVGs) and **order blocks**
  (OBs) on it directly.

* **Synthetic positive control.** A deterministic random-walk OHLC generator (fixed seed).
  This is the *null*: a driftless random walk with realistic per-bar volatility *also*
  produces three-bar gaps and "bounces". The control proves two things at once -- (a) the
  detector fires on pure noise, so an FVG/OB is not evidence of "smart money"; and (b) the
  measured fill-rate / forward edge on the real tape must be compared to *this* number, not
  to zero. The generator builds an OHLC bar per step from a within-bar mini-path so highs and
  lows are internally consistent (high >= max(open,close) >= ... >= low).

Pure numpy + pandas + stdlib; the synthetic core never touches the network.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "spy_daily.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_real(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Return the cached SPY daily OHLC frame, indexed by date.

    Columns: ``open, high, low, close`` (float), index a ``DatetimeIndex``.
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    return df[["open", "high", "low", "close"]].astype(float)


def fetch_real(path: str = DEFAULT_CACHE, start: str = "1993-01-29",
               end: str = "2026-06-20") -> pd.DataFrame:
    """Download SPY daily OHLC from yfinance and cache it. Network required; used once."""
    import yfinance as yf

    df = yf.download("SPY", start=start, end=end, interval="1d",
                     progress=False, auto_adjust=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return load_real(path)


# --------------------------------------------------------------------------- #
# Synthetic positive control / null -- deterministic random-walk OHLC
# --------------------------------------------------------------------------- #
def synthetic_ohlc(n: int = 8_000, sigma: float = 0.011, drift: float = 0.0,
                   start_px: float = 100.0, intrabar_steps: int = 8,
                   seed: int = 353) -> pd.DataFrame:
    """Deterministic random-walk OHLC bars -- the null tape.

    Each bar is a log-random-walk over ``intrabar_steps`` sub-steps; the bar's open/close are
    the first/last sub-prices and the high/low are the running max/min, so every bar is
    internally consistent. ``sigma`` is the per-bar log-return std (SPY daily ~1.1%). With
    ``drift = 0`` the tape has no edge of any kind -- any non-trivial FVG fill-rate or OB
    forward "edge" measured here is pure geometry of a random walk, which is precisely the
    control the ICT claim has to beat.
    """
    rng = np.random.default_rng(seed)
    per_step = sigma / np.sqrt(intrabar_steps)
    mu_step = drift / intrabar_steps
    # (n, intrabar_steps) log-returns; cumulate within and across bars
    steps = rng.normal(mu_step, per_step, size=(n, intrabar_steps))
    log_sub = np.cumsum(steps.reshape(-1)) + np.log(start_px)
    sub = np.exp(log_sub).reshape(n, intrabar_steps)
    o = sub[:, 0]
    c = sub[:, -1]
    hi = sub.max(axis=1)
    lo = sub.min(axis=1)
    idx = pd.date_range("2000-01-03", periods=n, freq="B")
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c}, index=idx)

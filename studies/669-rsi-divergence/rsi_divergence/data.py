"""Data layer for Study 669 — RSI-Divergence.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily OHLC for SPY plus a five-name liquid basket (QQQ, IWM, AAPL, MSFT,
  NVDA) from yfinance (no key), cached as CSV under the study's own ``_cache/``. No hardcoded
  calendar is needed here — unlike the FOMC-style studies, "bullish RSI divergence" is a purely
  algorithmic pattern detected from price + RSI(14), not tied to a scheduled public event.

* **Synthetic world.** A deterministic, seeded random-walk price path with a TUNABLE planted
  bounce right after a manufactured bullish-divergence pattern (knob ``bounce``, in forward
  log-return). ``bounce = 0`` is the null world — divergence-flagged bars behave exactly like
  any other bar; the Welch machinery must NOT manufacture significance from it.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build the
cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

# SPY + a liquid five-name basket (large-cap index/sector proxies + mega-cap single names) —
# the same flavour of universe used by sibling technical-pattern studies on this desk.
TICKERS = ("SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA")

START = "2010-01-04"     # 16+ full years, matches sibling technical-pattern studies
AS_OF = "2026-06-30"     # last complete calendar month at publication (2026-07-10)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"rsidiv_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2009-11-01", end: str = "2026-07-01") -> None:
    """Download raw daily OHLC for every ticker in TICKERS; cache each as CSV. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in TICKERS:
        raw = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = pd.DataFrame({
            "Open": raw["Open"], "High": raw["High"], "Low": raw["Low"],
            "Close": raw["Close"],
            "AdjClose": raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"],
        }).dropna(how="all")
        df.to_csv(_cache_path(t))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in TICKERS)


def load_real(start: str = START, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached {ticker: OHLC frame}, each sliced to [start, asof]."""
    out = {}
    for t in TICKERS:
        df = pd.read_csv(_cache_path(t), index_col=0, parse_dates=True).sort_index()
        out[t] = df.loc[(df.index >= start) & (df.index <= asof)].copy()
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted post-divergence bounce (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(bounce: float = 0.0, seed: int = 669,
                     n: int = 12000, mu: float = 0.0003, sig: float = 0.011,
                     period: int = 47, dip: float = 0.55, order: int = 5,
                     ) -> pd.DataFrame:
    """Deterministic random-walk price series with periodic dips and a TUNABLE planted bounce.

    Two passes, both fully deterministic given ``seed``:

    1. A geometric random walk (drift ``mu``, daily vol ``sig``) gets an extra downward push
       every ``period`` bars scaled by ``dip`` — engineered so genuine local price minima
       (swing lows) recur regularly, exactly like a real, choppy tape.
    2. The study's OWN detector (``strategy.detect_bullish_divergence``) runs on that base
       world to find which bars it would actually flag as a confirmed bullish divergence —
       and a ``bounce`` (log-return, spread over the 5 bars starting the next session's open,
       the study's own execution-lag convention) is added right there. ``bounce = 0`` means no
       extra push at all: the detector must then find nothing on its own flagged dates.

    This avoids guessing which synthetic dips the detector will call "divergence" — it plants
    the effect exactly where the real pipeline would trade, so the positive control tests the
    actual detector, not a hand-tuned proxy for it.

    A plain business-day index of ``n`` bars (~46 years) — far below the ns-timestamp trap.
    Returns a frame with an OHLC-shaped Close column (Open/High/Low approximated from Close
    for the swing detector, which only needs Close).
    """
    from . import strategy as _st  # local import: strategy.py has no dependency on data.py

    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    shocks = rng.normal(mu, sig, size=n)
    dip_bars = np.arange(period - 1, n, period)
    shocks[dip_bars] -= dip * sig * 6

    def _to_frame(sh):
        log_close = np.cumsum(sh) + np.log(100.0)
        close = np.exp(log_close)
        return pd.DataFrame({
            "Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close,
            "AdjClose": close,
        }, index=idx)

    base = _to_frame(shocks)
    if bounce == 0.0:
        return base

    ev = _st.detect_bullish_divergence(base, order=order)
    if not len(ev):
        return base
    pos_of = {d: i for i, d in enumerate(idx)}
    for d in ev["confirm_date"]:
        b = pos_of[d] + 1                      # the study's own next-open entry convention
        end = min(b + 5, n - 1)
        if b < n:
            shocks[b:end + 1] += bounce / 5.0
    return _to_frame(shocks)

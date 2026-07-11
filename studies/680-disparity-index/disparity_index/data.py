"""Data layer for Study 680 — Disparity Index.

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily OHLC for SPY (the primary, long-history index — no survivorship: an
  index, not a selected panel) plus a basket of five liquid large caps/ETFs (QQQ, IWM,
  AAPL, TSLA, NVDA) for breadth, all from yfinance (no key), cached as CSV under the
  study's own ``_cache/``. **Named honestly:** the single-name sleeve (AAPL/TSLA/NVDA) is
  three well-known, currently-thriving mega caps chosen for liquidity and name recognition,
  *not* a systematically-selected historical panel — a much milder selection bias than a
  "top N by market cap today" survivorship panel, but worth naming: a 1999-vintage tech
  basket including the era's flame-outs would likely look different. SPY/QQQ/IWM are index
  ETFs and carry no such bias. This is the identical universe used by sibling study
  [679-psychological-line](../679-psychological-line/) — same tape, same protocol shape, a
  different oscillator.

* **Synthetic world.** A deterministic, seeded daily price tape with a TUNABLE planted
  short-horizon mean-reversion knob (the only structure a "buy far-below-MA / sell
  far-above-MA" rule can possibly harvest). ``reversal = 0`` is a pure random walk — the
  null world, the disparity-conditioned split must NOT fire; ``reversal > 0`` plants a
  genuine bounce/pullback after the close has drifted away from its own trailing average.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

UNIVERSE = ("SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA")
PRIMARY = "SPY"

START = "2003-01-02"     # ~23 years, well past TSLA (2010) and NVDA (1999) inceptions
AS_OF = "2026-06-30"      # last complete month at publication (2026-07-10)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"di_{ticker}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01") -> None:
    """Download daily OHLC (total-return adjusted) for the whole universe; cache CSVs."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for ticker in UNIVERSE:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw[["Open", "Close"]].dropna(how="all")
        bars.to_csv(_cache_path(ticker))


def have_real() -> bool:
    return all(os.path.exists(_cache_path(t)) for t in UNIVERSE)


def load_real(start: str = START, asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """Cached {ticker: OHLC frame} dict, each sliced to [start, asof]."""
    out = {}
    for ticker in UNIVERSE:
        df = pd.read_csv(_cache_path(ticker), index_col=0, parse_dates=True).sort_index()
        out[ticker] = df.loc[(df.index >= start) & (df.index <= asof)].copy()
    return out


# --------------------------------------------------------------------------- #
# Synthetic world — planted disparity-conditioned reversion (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(reversal: float = 0.0, seed: int = 680, n_days: int = 4000,
                    daily_vol: float = 0.012, window: int = 10, oversold: float = 95.0,
                    overbought: float = 105.0, start: str = "2010-01-04") -> pd.DataFrame:
    """Deterministic daily OHLC tape with a TUNABLE planted disparity-conditioned reversal.

    Each day's return is plain i.i.d. noise **plus** a drift that switches on exactly
    when the **prior** close's disparity reading — DI = 100 x close / trailing-``window``
    SMA, computed causally from the ``window`` closes *before* that close, no look-ahead —
    sits in the oversold/overbought zone: a *positive* kicker after an oversold reading
    (price notably below its own trailing average), a *negative* kicker after an
    overbought one. This plants the literal effect the claim describes, on the literal
    condition the detector measures — a faithful engine, not a proxy for it.

    - ``reversal = 0``   -> a martingale; disparity extremes carry no forward information.
    - ``reversal > 0``   -> a genuine oversold-bounce / overbought-pullback bias.

    Business-day index, well under the pandas ns-timestamp span trap.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    eps = rng.normal(0.0, daily_vol, n_days)
    close = np.empty(n_days)
    prev_close = 100.0
    for t in range(n_days):
        drift = 0.0
        if t >= window:
            sma_prev = close[t - window:t].mean()
            di_prev = 100.0 * close[t - 1] / sma_prev
            if di_prev < oversold:
                drift = reversal
            elif di_prev > overbought:
                drift = -reversal
        r = drift + eps[t]
        new_close = prev_close * np.exp(r)
        close[t] = new_close
        prev_close = new_close
    open_ = np.empty_like(close)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    return pd.DataFrame({"Open": open_, "Close": close}, index=idx)


def synthetic_basket(reversal: float = 0.0, seed: int = 680, n_series: int = 6,
                     **kwargs) -> dict[str, pd.DataFrame]:
    """A basket of ``n_series`` independent synthetic tapes sharing one ``reversal`` knob.

    Mirrors the real-tape test's design (pooled across a multi-instrument universe): a
    single synthetic path gives disparity extremes too few observations for a stable
    Welch t, so the null-world check pools several independent paths per seed — exactly
    as the real headline pools six tickers.
    """
    return {
        f"SYN{i}": synthetic_world(reversal=reversal, seed=seed * 1000 + i, **kwargs)
        for i in range(n_series)
    }

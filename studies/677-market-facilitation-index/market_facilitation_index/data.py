"""Data layer for Study 677 — Market Facilitation Index (BW-MFI).

Two ingredients:

* **Real tape.** Daily raw OHLCV for SPY (headline) + a five-ETF basket (QQQ, DIA, IWM,
  XLE, GLD), from yfinance (no key), cached as CSV under the study's own ``_cache/``. We
  keep the **raw** High/Low/Volume (needed for the BW-MFI = (High-Low)/Volume ratio and
  its bar-to-bar deltas) alongside the **adjusted** close (splits + dividends, total
  return) used for every return computation — the same two-birds-one-download shape as
  sibling study 637.

* **Synthetic world.** A deterministic, seeded random-walk tape whose day-to-day *range*
  and *volume* processes are generated independently of the return path (exactly like the
  real mechanics: a bar's range and volume don't know tomorrow's return), so the four BW
  states fall out of range/volume alone. A TUNABLE knob then plants a continuation effect
  on **Green** bars (``fwd_ret`` leans the same way as today's return) and a mirror-image
  reversal effect on **Squat** bars (``fwd_ret`` leans the opposite way) — ``planted = 0``
  is the null world, where the four states carry no information about tomorrow.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

TICKERS = ["SPY", "QQQ", "DIA", "IWM", "XLE", "GLD"]

START = "1993-02-01"   # SPY inception is 1993-01-29; GLD inception is 2004-11-18
AS_OF = "2026-06-30"   # last complete month at publication (2026-07-10)


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"mfi_{ticker.lower()}.csv")


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(tickers: list[str] = TICKERS, start: str = START, end: str = "2026-07-01") -> None:
    """Download raw OHLCV + adjusted close per ticker; cache them. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for t in tickers:
        raw = yf.download(t, start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = pd.DataFrame({
            "Open": raw["Open"], "High": raw["High"], "Low": raw["Low"],
            "Close": raw["Close"], "Volume": raw["Volume"],
            "AdjClose": raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"],
        }).dropna(how="all")
        df.to_csv(_cache_path(t))


def have_real(tickers: list[str] = TICKERS) -> bool:
    return all(os.path.exists(_cache_path(t)) for t in tickers)


def load_real(ticker: str, start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached raw OHLCV + adjusted-close frame for ``ticker``, sliced to [start, asof]."""
    df = pd.read_csv(_cache_path(ticker), index_col=0, parse_dates=True).sort_index()
    return df.loc[(df.index >= start) & (df.index <= asof)].copy()


def load_basket(tickers: list[str] = TICKERS, start: str = START,
                asof: str = AS_OF) -> dict[str, pd.DataFrame]:
    """``{ticker: frame}`` for every ticker with a cache hit."""
    return {t: load_real(t, start, asof) for t in tickers if os.path.exists(_cache_path(t))}


# --------------------------------------------------------------------------- #
# Synthetic world — planted Green-continuation / Squat-reversal effect
# --------------------------------------------------------------------------- #
def synthetic_world(planted: float = 0.0, seed: int = 677, n_days: int = 4000,
                    daily_vol: float = 0.011, start: str = "2008-01-02",
                    ) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily OHLCV tape with a tunable BW-MFI continuation/reversal knob.

    Two independent AR(1)-in-logs processes drive the day's **range** (High-Low, as a
    fraction of price) and **volume** — exactly like the real mechanics, where the bar's
    range and volume are set by the session's trading, not by tomorrow's return. Their
    bar-to-bar deltas classify each day into green/fade/fake/squat by the same rule as
    ``strategy.classify_states``.

    The planted mechanism mirrors the claim under test::

        fwd_ret(t) = planted * daily_vol * effect[state(t)] * sign(ret(t)) + eps(t+1)

    with ``effect = {green: +1, squat: -1, fake: 0, fade: 0}`` — a Green bar nudges
    tomorrow's return to *continue* today's direction, a Squat bar nudges it to *reverse*,
    Fade/Fake carry no planted effect. ``planted = 0`` is the null world: the four states
    are statistically inert and the Welch machinery must NOT manufacture significance.

    Business-day index, ``n_days`` <= ~8,000 (~32 years) — far below the 250-year pandas
    ns-timestamp trap. Returns ``(bars, truth)``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    # Volume: persistent log-AR(1) (heavy days cluster), independent of the return path.
    log_v = np.empty(n_days)
    log_v[0] = 0.0
    v_noise = rng.normal(0.0, 0.45, n_days)
    for i in range(1, n_days):
        log_v[i] = 0.85 * log_v[i - 1] + v_noise[i]
    volume = np.maximum((5_000_000.0 * np.exp(log_v)).round(), 1.0)

    # Day range (H-L as a fraction of price): also a persistent log-AR(1), independent of
    # both the return path AND the volume path (range and volume co-move only through the
    # coincidental sign draws below, exactly the "two independent inputs" BW-MFI logic).
    r_noise = rng.normal(0.0, 0.35, n_days)
    log_r = np.empty(n_days)
    log_r[0] = 0.0
    for i in range(1, n_days):
        log_r[i] = 0.70 * log_r[i - 1] + r_noise[i]
    range_frac = 0.008 * np.exp(log_r)

    # Raw MFI and the four-state classification (range/volume only — no return leaks in).
    raw_mfi = range_frac / (volume / 1.0e6)
    dmfi = np.diff(raw_mfi, prepend=np.nan)
    dvol = np.diff(volume, prepend=np.nan)
    state = np.full(n_days, "", dtype=object)
    for i in range(1, n_days):
        if dmfi[i] > 0 and dvol[i] > 0:
            state[i] = "green"
        elif dmfi[i] < 0 and dvol[i] < 0:
            state[i] = "fade"
        elif dmfi[i] > 0 and dvol[i] < 0:
            state[i] = "fake"
        elif dmfi[i] < 0 and dvol[i] > 0:
            state[i] = "squat"

    effect = {"green": 1.0, "squat": -1.0, "fake": 0.0, "fade": 0.0, "": 0.0}

    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = np.zeros(n_days)
    close = np.empty(n_days)
    close[0] = 100.0
    for i in range(1, n_days):
        if i >= 2:
            planted_term = planted * daily_vol * effect[state[i - 1]] * np.sign(log_ret[i - 1])
        else:
            planted_term = 0.0
        log_ret[i] = planted_term + eps[i]
        close[i] = close[i - 1] * np.exp(log_ret[i])

    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    # High/Low bracket the CLOSE ONLY (not open..close), each side ``range_frac*100/2`` off
    # a fixed reference price. Anchoring on max(open,close)/min(open,close) instead would
    # add |open-close| (the day's own realized move) into High-Low, contaminating it with
    # the return path and decoupling the recomputed state from the ``raw_mfi``/state used
    # to plant the effect above — this way High-Low stays EXACTLY proportional to
    # ``range_frac`` (a fixed positive multiple), so ``strategy.classify_states`` run on
    # the output bars reproduces the planning state bar-for-bar.
    half = range_frac * 100.0 / 2.0
    hi = close + half
    lo = np.maximum(close - half, 0.01)

    bars = pd.DataFrame({
        "Open": open_, "High": hi, "Low": lo, "Close": close,
        "AdjClose": close, "Volume": volume,
    }, index=idx)
    truth = {"planted": planted, "seed": seed, "n_days": n_days}
    return bars, truth

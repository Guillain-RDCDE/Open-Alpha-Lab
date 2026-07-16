"""Data layer for Study 779 — Devil-Price.

The claim under test: **a stock whose price sits on a "666" level then underperforms.**
The number of the beast is the world's most famous unlucky number, and traders trade
folklore. The testable version: at each weekly rebalance, rank a real large-cap universe
by how close each name's price is to a **666-family level** — 6.66, 66.6, 666, 6660 … —
and ask whether the names *touching* the devil level go on to *lag* the ones far from it.

This is a **cross-sectional characteristic sort** (shape B): the characteristic is a
name's log-price distance to the nearest 666 level; the sort is a dollar-neutral
long/short of the *farthest* quintile ("clean") minus the *nearest* quintile ("devil").
If the superstition bit, the devil quintile would underperform (a positive clean−devil
spread). The universe is the **S&P 100** (OEX) mega-caps and the market benchmark is
``SPY``; the sort itself is cross-sectionally demeaned so it is market-neutral by
construction.

Why "distance to a 666 level" is decade-invariant. A $66.60 stock and a $666 stock are
both "on the beast" — only the *mantissa* matters, not the decade. So we measure a name's
position on the log-price circle: ``f = frac(log10 price)`` and the devil sits at
``log10 6.66 = 0.8235``. The circular distance ``min(|f−f_devil|, 1−|f−f_devil|)`` lives
in ``[0, 0.5]`` and is small exactly when the price is on a 666 handle in *any* decade —
so the "devil" quintile is populated at every rebalance even though almost no mega-cap
literally prints $666.

Four ingredients:

* **The universe, hardcoded.** The S&P 100 constituents (a real, published index — the
  CBOE OEX). Each weekly rebalance only uses the names that actually have a price on that
  date and enough forward history, so the funnel is auditable and survivorship is not
  smuggled in beyond the index membership itself.

* **The tradable instruments (yfinance).** ~100 S&P 100 tickers plus ``SPY`` (S&P 500
  total-return proxy), daily adjusted (total-return) closes. The sort's legs are
  cross-sectionally demeaned, so the market drops out; ``SPY`` is used for the
  "does the devil quintile lag the *market*?" cross-check.

* **No fundamental proxy needed.** "Sitting on a 666 level" is a pure price-path fact:
  the price and the calendar of rebalances are all the event needs.

* **Synthetic world.** A deterministic, seeded panel of correlated log-return names with
  prices spread across the log-price circle and a TUNABLE planted "devil-quintile
  underperformance." ``bump = 0`` is the null world; the one-sample-t machinery must not
  manufacture significance from it and must recover a planted drag monotonically.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")

AS_OF = "2026-06-30"     # last complete month at publication
BENCHMARK = "SPY"        # S&P 500 total-return proxy

# The devil sits at mantissa 6.66 -> fractional log-price log10(6.66).
DEVIL_LOG = float(np.log10(6.66))   # 0.8234742...

# --------------------------------------------------------------------------- #
# The S&P 100 (CBOE OEX) universe, hardcoded. A real, published mega-cap index. Names
# with dual share classes (GOOG/GOOGL) are both kept; each weekly rebalance only uses the
# names that actually have price + forward history on that date, so late IPOs (TSLA 2010,
# META 2012, ABBV 2013, PYPL 2015) simply enter the cross-section when their tape begins.
# Membership drifts year to year; this is a representative recent snapshot, used only to
# define the sortable cross-section, never as a survivorship-clean backtest universe.
# --------------------------------------------------------------------------- #
UNIVERSE = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE",
    "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "INTC", "INTU",
    "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ",
    "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX",
    "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM", "RTX",
    "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN",
    "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT", "XOM",
]

CACHE_FILE = os.path.join(CACHE_DIR, "devil_panel.csv")


def all_tickers() -> list[str]:
    return list(UNIVERSE) + [BENCHMARK]


# --------------------------------------------------------------------------- #
# Devil distance: circular log-price distance to the nearest 666 level, in [0, 0.5].
# --------------------------------------------------------------------------- #
def devil_distance(price):
    """Circular distance of ``frac(log10 price)`` to ``log10 6.66``.

    Decade-invariant: $6.66, $66.6, $666, $6660 all map to distance 0. Accepts a scalar
    or an array; returns the same shape, values in ``[0, 0.5]`` (0 = exactly on a beast
    handle, 0.5 = maximally far).
    """
    p = np.asarray(price, dtype=float)
    f = np.mod(np.log10(p), 1.0)
    d = np.abs(f - DEVIL_LOG)
    return np.minimum(d, 1.0 - d)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2007-01-01", end: str = "2026-07-01", retries: int = 4) -> None:
    """Download adjusted (total-return) daily closes for the S&P 100 + SPY; cache one
    wide CSV (dates x tickers).

    Retries with linear backoff — Yahoo rate-limits transient bursts, so a first empty
    frame is usually cured by a short wait rather than a real "no such ticker".
    """
    import time

    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    tickers = all_tickers()
    last_err = None
    for attempt in range(retries):
        try:
            d = yf.download(tickers, start=start, end=end, auto_adjust=True,
                            progress=False, group_by="column")
            if isinstance(d.columns, pd.MultiIndex):
                # level 0 = field ('Close', ...), level 1 = ticker
                close = d["Close"].copy()
            else:
                close = d[["Close"]].copy()
                close.columns = [tickers[0]]
            close = close.dropna(how="all")
            # keep only tickers that came back with some data
            close = close.loc[:, close.notna().any(axis=0)]
            if close.shape[1] >= 0.8 * len(tickers) and len(close) > 0:
                close.to_csv(CACHE_FILE)
                return
            last_err = (f"only {close.shape[1]}/{len(tickers)} tickers returned "
                        f"{len(close)} rows")
        except Exception as e:  # noqa: BLE001 -- transient network/rate-limit
            last_err = str(e)
        time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"fetch failed after {retries} tries: {last_err}")


def have_real() -> bool:
    return os.path.exists(CACHE_FILE)


def load_real(asof: str = AS_OF) -> tuple[pd.DataFrame, pd.Series]:
    """Cached (universe close panel, SPY close Series), both sliced to <= asof.

    The panel columns are the S&P 100 names that returned data; SPY is split off as the
    market benchmark. Missing cells (a name not yet public) stay NaN and are dropped
    per-rebalance by the strategy.
    """
    df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True).sort_index()
    df = df[df.index <= pd.Timestamp(asof)]
    spy = df[BENCHMARK] if BENCHMARK in df.columns else pd.Series(dtype=float)
    panel = df[[c for c in df.columns if c != BENCHMARK]]
    return panel, spy


# --------------------------------------------------------------------------- #
# Synthetic world -- panel of names on the log-price circle + planted devil-quintile drag
# --------------------------------------------------------------------------- #
def synthetic_world(bump: float = 0.0, seed: int = 789, n_stocks: int = 60,
                    n_days: int = 1500, spacing: int = 5, fwd: int = 5,
                    ) -> tuple[pd.DataFrame, list[int]]:
    """Deterministic panel of correlated log-return names with a planted devil drag.

    Names share a common market factor (rho ~ 0.5) plus idiosyncratic noise; their
    *initial* prices are spread log-uniformly across ~[10, 3000] so their mantissas cover
    the whole log-price circle and the "devil" (nearest-6.66) quintile is populated at
    every rebalance. On each rebalance day (every ``spacing`` business days) the names in
    the nearest-devil quintile get an EXTRA ``-bump`` total return spread over the next
    ``fwd`` days — a planted "touching-666 then underperforms." ``bump = 0`` is the null.

    Returns (price_panel [n_days x n_stocks], rebalance_positions). The strategy re-derives
    the devil quintile from the returned prices exactly as it does on the real tape, so
    this is an honest end-to-end positive control, not a hand-fed answer.
    """
    rng = np.random.default_rng(seed)
    rho = 0.5
    common = rng.normal(0.0, 0.010, (n_days, 1))
    idio = rng.normal(0.0, 0.016, (n_days, n_stocks))
    rets = rho * common + np.sqrt(1 - rho ** 2) * idio   # daily log returns

    p0 = 10.0 * (10 ** (rng.uniform(0.0, 2.5, n_stocks)))  # spread ~[10, 3160]
    # baseline prices (pre-plant) used to fix the devil membership truth
    base_prices = p0[None, :] * np.exp(np.cumsum(rets, axis=0))

    rebal = list(range(spacing, n_days - fwd - 1, spacing))
    if bump != 0.0:
        per_day = bump / fwd
        for p in rebal:
            dist = devil_distance(base_prices[p])
            order = np.argsort(dist)
            k = max(1, n_stocks // 5)
            near = order[:k]
            rets[p + 1:p + 1 + fwd, near] -= per_day   # planted drag on the devil quintile
    prices = p0[None, :] * np.exp(np.cumsum(rets, axis=0))
    cols = [f"S{i:02d}" for i in range(n_stocks)]
    return pd.DataFrame(prices, columns=cols), rebal

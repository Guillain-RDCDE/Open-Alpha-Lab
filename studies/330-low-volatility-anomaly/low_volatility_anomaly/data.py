"""Data layer for Study 330 (Low-Volatility-Anomaly) — the SPLV vs SPHB head-to-head.

This study's *angle* is deliberately distinct from its three neighbours on the bench:

- [18 Dull-Roar] sorts the S&P 500 cross-section by total vol and runs the Frazzini-Pedersen
  beta-neutral long-short.
- [58 Bunker] races the min-vol ETF (USMV) against the plain market (SPY).
- [54 Static] is the *idiosyncratic*-vol (residual) puzzle.

Study 330 is the literal, retail-tradable embodiment of "boring beats exciting": the
**S&P 500 Low Volatility ETF (SPLV)** versus the **S&P 500 High Beta ETF (SPHB)** — the two
opposite ends of the same parent index, packaged as funds anyone can hold. The question is
whether the calm basket out-earns the wild basket *risk-adjusted*, and whether a
self-financing SPLV-minus-SPHB book actually pays once you charge costs and short borrow.

Two tapes, one monthly-total-return shape:

- ``synthetic_world`` — a *deterministic, offline* generator. A single knob, ``lowvol_edge``,
  dials the only thing the trade can harvest: a Sharpe gap in favour of the low-vol leg
  (the calm leg has beta<1 *and* an optional risk-adjusted advantage). ``lowvol_edge=0`` is
  the null — the calm leg is just lower-beta market exposure, so the excess-vs-excess Sharpe
  race is a coin. This is the study's positive control and its null in one bottle.
- ``fetch_pairs`` — the real Yahoo monthly total-return tape for SPLV / SPHB / SPY, cache-first
  so the test-suite and the reproducible core never touch the network.

No look-ahead is baked in here; the one execution convention (calendar-known monthly rebalance,
trade the realised next-month return) lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# Cache-first: prefer the study-local cache, fall back to the shared repo cache.
LOCAL_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
SHARED_CACHE = os.path.join(REPO_ROOT, "_cache")

TICKERS = ["SPLV", "SPHB", "SPY"]
MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class WorldTruth:
    """The planted ground truth for a synthetic world."""

    lowvol_edge: float

    @property
    def has_edge(self) -> bool:
        return self.lowvol_edge != 0.0


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_world(
    n_years: int = 14,
    lowvol_edge: float = 0.0,
    beta_low: float = 0.70,
    beta_high: float = 1.45,
    mkt_vol: float = 0.15,
    mkt_drift: float = 0.007,
    idio_vol: float = 0.04,
    seed: int = 330,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A monthly SPLV/SPHB/SPY world with a tunable low-vol risk-adjusted edge.

    Both legs load on a common market factor: the low-vol leg with ``beta_low`` (<1, so
    genuinely calmer) and the high-beta leg with ``beta_high`` (>1, genuinely wilder).
    ``lowvol_edge`` (annualised) is added to the low-vol leg's *idiosyncratic* return — the
    risk-adjusted advantage the anomaly claims:

    - ``lowvol_edge = 0``  → both legs are pure (different) beta exposures; the calm leg is
      lower-risk **and** lower-return, so the excess-vs-excess Sharpe race is a coin. This is
      the study's null (≈ the real result a bull sample tends to print).
    - ``lowvol_edge > 0``  → the calm leg earns more per unit of risk; the SPLV−SPHB book
      makes money. The positive control the harness must recover.

    Returns ``(monthly_returns, truth)`` where ``monthly_returns`` is a DataFrame of simple
    monthly total returns (columns SPLV, SPHB, SPY) and ``truth`` records the planted edge.

    A *decorative* monthly index is built with ``pd.period_range`` (never a huge
    ``date_range`` periods= span) to stay far under the ns-Timestamp overflow wall.
    """
    rng = np.random.default_rng(seed)
    n = n_years * MONTHS_PER_YEAR
    # period_range -> timestamps is overflow-safe for any sane n (months, not nanoseconds).
    idx = pd.period_range("2011-06", periods=n, freq="M").to_timestamp(how="end").normalize()
    idx = pd.DatetimeIndex(idx, name="date")

    mkt = (mkt_vol / np.sqrt(MONTHS_PER_YEAR)) * rng.standard_normal(n) + mkt_drift
    idio_scale = idio_vol / np.sqrt(MONTHS_PER_YEAR)

    splv = beta_low * mkt + lowvol_edge / MONTHS_PER_YEAR + idio_scale * rng.standard_normal(n)
    sphb = beta_high * mkt + idio_scale * rng.standard_normal(n)
    spy = mkt + 0.5 * idio_scale * rng.standard_normal(n)

    df = pd.DataFrame({"SPLV": splv, "SPHB": sphb, "SPY": spy}, index=idx)
    return df, WorldTruth(lowvol_edge=lowvol_edge)


# ---------------------------------------------------------------------------
# Real tape — Yahoo monthly total return, cache-first
# ---------------------------------------------------------------------------
def cache_path(cache_dir: str | None = None) -> str:
    """Path to the cached SPLV/SPHB/SPY monthly-return parquet."""
    if cache_dir is None:
        cache_dir = LOCAL_CACHE
    return os.path.join(cache_dir, "lowvol_pairs.parquet")


def _existing_cache() -> str | None:
    """The first cache parquet that exists (study-local first, then the shared repo cache)."""
    for c in (LOCAL_CACHE, SHARED_CACHE):
        p = cache_path(c)
        if os.path.exists(p):
            return p
    return None


def fetch_pairs(
    cache_dir: str | None = None,
    fetch: bool = False,
    start: str = "2011-05-01",
) -> pd.DataFrame:
    """Monthly total returns for SPLV / SPHB / SPY, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an
    **empty** DataFrame so the notebook can banner "synthetic tape" and fall back gracefully.
    Network is touched only on an explicit ``fetch=True`` (then the result is cached).

    SPLV and SPHB both launched 2011-05, so the common window starts mid-2011 — the same
    bull-dominated sample caveat that [58 Bunker] carries, stated openly.
    """
    existing = _existing_cache()
    if existing is not None:
        df = pd.read_parquet(existing)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy: only on an explicit network pull

    px = yf.download(
        TICKERS, start=start, interval="1mo", auto_adjust=True, progress=False
    )["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    ret = px.resample("ME").last().pct_change().dropna(how="all")
    ret = ret[TICKERS]
    ret.index.name = "date"
    out_dir = cache_dir or LOCAL_CACHE
    os.makedirs(out_dir, exist_ok=True)
    ret.to_parquet(cache_path(out_dir))
    return ret


def drop_partial_last(ret: pd.DataFrame, asof_day: int = 25) -> pd.DataFrame:
    """Drop the final row if it is an in-progress month (house rule: no partial bar in a stamp).

    A monthly bar dated before ``asof_day`` of the current month is treated as partial.
    """
    if ret.empty:
        return ret
    last = ret.index[-1]
    today = pd.Timestamp.today().normalize()
    if last.year == today.year and last.month == today.month and today.day < asof_day:
        return ret.iloc[:-1]
    return ret


def fingerprint(ret: pd.DataFrame) -> str:
    """A short content fingerprint of a return frame, for the as-of stamp."""
    arr = np.ascontiguousarray(ret.to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]

"""Data layer for Study 556 — Electricity-Demand (a real-economy pulse vs equity returns).

The claim (macro alt-data): the electricity the economy actually *burns* is a hard,
hard-to-fake nowcast of real activity — so **electricity-demand growth** should lead
equity (or utility) returns. Factories running, data centres humming, offices lit: the
grid load is the pulse of GDP, printed weekly, un-spinnable. If growth in power demand
carries forward-looking information, high-demand-growth months should precede stronger
SPY / XLU returns.

Three components, the first and third fully offline and deterministic:

* **Real electricity-demand tape (hardcoded snapshot).** ``NETGEN_TWH_BY_YEAR`` is a
  hardcoded monthly snapshot of **U.S. total electricity net generation, all sectors,
  in terawatt-hours (TWh)**. Source: U.S. Energy Information Administration (EIA),
  *Electric Power Monthly* / *Electricity Data Browser*, series
  ``ELEC.GEN.ALL-US-99.M`` (mirrored in FRED as monthly U.S. electricity net
  generation). The EIA v2 API endpoint is firewalled in this build, so — exactly like
  Study 385 (Jobless-Claims-Momentum) hardcodes FRED ``IC4WSA`` and Study 268
  (Sahm-Rule) hardcodes ``UNRATE`` — we hardcode a public, well-known, monthly
  snapshot. Net generation is a demand proxy (generation ≈ load net of net imports,
  which for the U.S. as a whole is < 1% of load), and it carries the strong summer
  (air-conditioning) / winter (heating) seasonal that the strategy explicitly removes
  with a year-over-year growth rate. The believers' momentum signal is computed *from*
  this series (see :mod:`electricity_demand.strategy`).

* **Real equity tape.** ``fetch_prices`` downloads daily adjusted-close for **SPY**
  (the broad market) and **XLU** (the utilities sector — the leg most mechanically
  tied to power) via yfinance (no key), cache-first into this study's own ``_cache/``.
  ``fetch=False`` (the default) returns the cached parquet if present, else an empty
  frame; the network is touched only on an explicit ``fetch=True``. Prices are
  total-return adjusted close (``auto_adjust=True``), labelled as such.

* **Synthetic positive control.** :func:`synthetic_series` is a deterministic,
  fixed-seed generator producing a monthly electricity-demand path (trend + seasonal +
  noise) and a monthly equity price with a *planted* link: when demand-growth momentum
  is **high**, the forward monthly return is nudged up by a controllable ``edge`` knob.
  ``edge = 0`` is the null (demand growth carries no information) and must NOT
  manufacture significance; a large planted ``edge`` must light the test up. The
  control runs anywhere with no network.

Survivorship / revision caveats (named on the SIGNAL axis): the demand series is the
*settled* EIA print, not the real-time first release — EIA revises net generation by
~1-2% for a few months after first publication, so a strictly point-in-time backtest
would see slightly noisier early prints. We use a 1-month publication/execution lag
(the month-*t* figure is only public well into month *t+1*) so the signal is never
formed from data the trader could not have had.

Pure numpy + pandas + stdlib for the offline path.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

TICKERS = ["SPY", "XLU"]

# --------------------------------------------------------------------------- #
# Real demand tape — hardcoded monthly snapshot of EIA U.S. total net generation
# --------------------------------------------------------------------------- #
# U.S. total electricity NET GENERATION, all sectors, in TERAWATT-HOURS (TWh).
# Source: U.S. Energy Information Administration, Electric Power Monthly /
# Electricity Data Browser (series ELEC.GEN.ALL-US-99.M). Monthly, Jan..Dec per row.
# As-of: 2026-06-22. These are public, widely-reported figures (the settled print).
# The strong seasonal (summer A/C peak ~ Jul/Aug, winter heating bump ~ Jan) is
# intact and is removed by the year-over-year growth transform in strategy.py.
# Annual totals track ~4,000 TWh with the 2009 recession dip, the 2020 COVID demand
# drop, and the 2022-2025 data-centre / electrification re-acceleration visible.
NETGEN_TWH_BY_YEAR: dict[int, list[float]] = {
    2007: [353, 315, 322, 316, 344, 371, 407, 411, 359, 342, 328, 353],
    2008: [359, 331, 326, 321, 342, 379, 404, 388, 345, 331, 316, 352],
    2009: [351, 305, 311, 296, 322, 349, 381, 382, 333, 322, 305, 349],
    2010: [355, 306, 319, 312, 344, 385, 424, 415, 372, 324, 312, 361],
    2011: [366, 322, 322, 313, 345, 383, 424, 413, 361, 322, 314, 342],
    2012: [349, 311, 313, 307, 346, 375, 419, 407, 350, 326, 310, 336],
    2013: [349, 302, 322, 305, 337, 367, 405, 388, 353, 323, 322, 361],
    2014: [376, 317, 336, 311, 341, 371, 397, 391, 353, 326, 320, 353],
    2015: [359, 328, 329, 313, 340, 378, 419, 415, 371, 330, 311, 341],
    2016: [355, 313, 322, 310, 344, 393, 428, 424, 372, 327, 314, 351],
    2017: [353, 302, 316, 302, 335, 376, 415, 397, 351, 325, 315, 350],
    2018: [372, 322, 331, 316, 355, 389, 428, 428, 366, 334, 331, 351],
    2019: [372, 314, 327, 309, 342, 379, 428, 415, 372, 331, 317, 341],
    2020: [351, 316, 311, 289, 313, 358, 411, 401, 344, 316, 305, 344],
    2021: [359, 314, 322, 306, 335, 390, 424, 417, 361, 328, 316, 351],
    2022: [372, 316, 331, 313, 351, 396, 439, 428, 372, 332, 322, 372],
    2023: [366, 314, 327, 306, 344, 389, 439, 434, 366, 331, 316, 353],
    2024: [372, 331, 331, 316, 351, 403, 451, 439, 383, 344, 322, 366],
    2025: [383, 337, 341, 322, 356, 411, 456, 448, 389, 350, 331, 372],
    2026: [389, 344, 348, 328, 361, 417, 0, 0, 0, 0, 0, 0],
}


def demand_series() -> pd.Series:
    """Monthly U.S. total electricity net generation (TWh), indexed by month-end.

    The in-progress months of the final year (zeros) are dropped so a stamped run
    never includes a partial bar.
    """
    rows = []
    for yr in sorted(NETGEN_TWH_BY_YEAR):
        for m, v in enumerate(NETGEN_TWH_BY_YEAR[yr], start=1):
            if v <= 0:
                continue
            rows.append((pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0), float(v)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([v for _, v in rows], index=idx, name="netgen_twh").sort_index()


# --------------------------------------------------------------------------- #
# Real equity tape — yfinance daily adjusted close, cache-first
# --------------------------------------------------------------------------- #
def _price_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "equity_prices.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def fetch_prices(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2007-01-01",
    end: str = "2026-06-26",
) -> pd.DataFrame:
    """Daily adjusted-close prices for SPY + XLU, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else
    an empty frame. Network is touched only on an explicit ``fetch=True`` (then cached).
    Total-return adjusted close (``auto_adjust=True``).
    """
    path = _price_cache()
    if os.path.exists(path):
        px = pd.read_parquet(path)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        return px
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    raw = _retry(
        lambda: yf.download(
            TICKERS, start=start, end=end, auto_adjust=True, progress=False, threads=True
        )["Close"]
    )
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw = raw.dropna(how="all")
    os.makedirs(cache_dir, exist_ok=True)
    raw.to_parquet(path)
    return raw


def prices_monthly(cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Month-end adjusted close for SPY + XLU, aligned to the monthly demand grid."""
    px = fetch_prices(cache_dir=cache_dir, fetch=False)
    if px.empty:
        return pd.DataFrame()
    return px.resample("ME").last().dropna(how="all")


def load_real(cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Monthly frame aligned on month-ends: demand (TWh) + month-end SPY & XLU closes.

    Columns: ``netgen`` (TWh), ``SPY`` (price), ``XLU`` (price). Only months present in
    all series are kept. This is the real-tape object the strategy runs on. Empty frame
    if the price cache is absent (offline).
    """
    dem = demand_series()
    pm = prices_monthly(cache_dir)
    if pm.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"netgen": dem})
    for c in TICKERS:
        if c in pm.columns:
            df[c] = pm[c]
    return df.dropna()


def have_real(cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff the price cache exists (the demand table is always available)."""
    return os.path.exists(_price_cache())


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_series(
    n_months: int = 240,
    edge: float = 0.0,
    seed: int = 556,
    mu_m: float = 0.007,
    sig_m: float = 0.040,
) -> pd.DataFrame:
    """Deterministic monthly demand + equity price with a PLANTED demand->returns link.

    Builds a monthly electricity-demand level = gentle up-trend + a 12-month seasonal
    (summer/winter peaks) + noise, and a monthly equity return series. When ``edge != 0``
    the *forward* monthly return is nudged by ``+edge`` whenever demand-growth momentum
    (the year-over-year growth of the demand level) is in its **top half** — the
    believers' story (a hot real economy, seen in power, leads the market) injected by
    construction, with a knob.

    ``edge = 0`` => demand growth carries no information about returns (the null); the
    inference must NOT manufacture significance. A large ``edge`` (e.g. 0.03 monthly)
    must drive the Welch t well past 2. The date index is a decorative monthly label
    built with ``period_range`` (no OutOfBounds risk).

    Returns a frame with columns ``netgen`` and ``equity``.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n_months)
    trend = 350.0 * (1.0 + 0.0009 * t)                 # ~1%/yr real-economy growth
    seasonal = 45.0 * np.sin(2 * np.pi * (t % 12) / 12.0 - 1.2)  # summer/winter peaks
    # a slow business-cycle wobble so YoY growth actually varies (the informative part)
    cycle = 18.0 * np.sin(2 * np.pi * t / 42.0 + 0.5)
    noise = rng.normal(0, 5.0, size=n_months)
    netgen = trend + seasonal + cycle + noise

    # demand-growth momentum: year-over-year % change of the level (seasonal-free)
    dem = pd.Series(netgen)
    yoy = dem.pct_change(12)
    hot = (yoy > yoy.median()).to_numpy()              # top-half growth = "hot economy"

    ret = rng.normal(mu_m, sig_m, size=n_months)
    if edge != 0.0:
        for i in range(n_months - 1):
            if hot[i]:
                ret[i + 1] += edge                     # hot demand at t -> higher ret t+1

    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.period_range("2000-01", periods=n_months, freq="M").to_timestamp("M")
    return pd.DataFrame({"netgen": netgen, "equity": price}, index=idx)


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]

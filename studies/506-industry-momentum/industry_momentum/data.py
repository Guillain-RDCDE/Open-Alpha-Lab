"""Data layer for Study 506 (Industry-Momentum).

Two tapes, one monthly-total-return shape:

- ``synthetic_world`` -- a *deterministic, offline* generator. A single knob,
  ``industry_mom``, dials the only thing the industry-momentum trade can harvest: a
  *persistent* industry-level return that auto-correlates month to month (winner sectors
  keep winning). ``industry_mom = 0`` is the null -- sectors are i.i.d. around the market and
  the rank-and-hold book is a coin. This is the study's positive control and its null in one
  bottle. (A *decorative* monthly index is built with ``pd.period_range`` -- never a huge
  ``date_range`` periods= span -- to stay far under the ns-Timestamp overflow wall.)

- ``fetch_prices`` -- the real Yahoo daily adjusted-close tape for the 11 SPDR sector ETFs
  plus SPY, and (separately) a fixed large-cap survivor basket for the single-name momentum
  comparison. Cache-first, so the reproducible core and the notebooks never touch the network.

SURVIVORSHIP (named explicitly on the SIGNAL axis): the single-name basket is large-cap names
still trading in 2026. Failed/delisted firms -- the natural loser-leg short -- are absent, so
any single-name momentum loser leg is an UPPER-BOUND. The sector-ETF book is essentially
survivorship-free (the 11 SPDR sectors are stable, index-tracking funds), which is one reason
the industry expression is the cleaner test.

No look-ahead: the 12-1 momentum signal is formed on months t-12..t-1 (skipping the most
recent month to dodge the short-term reversal) and the portfolio is held over month t+1 -- one
documented execution lag lives in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOCAL_CACHE = os.path.join(STUDY_DIR, "_cache")

# The 11 SPDR sector ETFs (XLRE launched 2015-10, XLC 2018-06; the rest 1998-12).
SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
]
MARKET = "SPY"

# A fixed ~40-name large-cap survivor basket for the single-name momentum comparison.
# Names still trading in 2026 -> survivorship-biased, stated on the SIGNAL axis.
SINGLE_NAME_BASKET = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM", "UNH", "XOM", "JNJ",
    "PG", "MA", "HD", "MRK", "CVX", "BAC", "PEP", "KO", "ABBV", "WMT",
    "MCD", "CSCO", "ACN", "DHR", "VZ", "TXN", "ADBE", "NEE", "RTX", "INTC",
    "IBM", "CAT", "HON", "LOW", "GS", "AXP", "BKNG", "SBUX", "GILD", "DUK",
]
_seen: set[str] = set()
SINGLE_NAME_BASKET = [t for t in SINGLE_NAME_BASKET if not (t in _seen or _seen.add(t))]  # type: ignore[func-returns-value]

MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class WorldTruth:
    """The planted ground truth for a synthetic world."""

    industry_mom: float  # strength of the persistent industry-momentum component

    @property
    def has_momentum(self) -> bool:
        return self.industry_mom != 0.0


# ---------------------------------------------------------------------------
# Synthetic world -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_world(
    n_sectors: int = 11,
    n_years: int = 18,
    industry_mom: float = 0.0,
    mkt_vol: float = 0.15,
    mkt_drift: float = 0.006,
    sector_vol: float = 0.05,
    persistence: float = 0.85,
    seed: int = 506,
) -> tuple[pd.DataFrame, pd.Series, WorldTruth]:
    """A monthly sector-return world with a tunable *persistent* industry-momentum component.

    Each sector loads on a common market factor plus a sector-specific component that is an
    AR(1) process with autocorrelation ``persistence`` and innovation scale driven by
    ``industry_mom`` (annualised). When ``industry_mom > 0`` the sector-specific piece *trends*
    -- a sector that did well last month tends to do well this month -- which is exactly what a
    12-1 rank-and-hold book harvests. When ``industry_mom = 0`` the sector pieces are flat
    (pure market beta + tiny noise): rank-and-hold is a coin (the null).

    Returns ``(sector_rets, mkt_rets, truth)`` as monthly simple total returns. A *decorative*
    period_range index keeps us overflow-safe.
    """
    rng = np.random.default_rng(seed)
    n = n_years * MONTHS_PER_YEAR
    idx = pd.period_range("2008-01", periods=n, freq="M").to_timestamp(how="end").normalize()
    idx = pd.DatetimeIndex(idx, name="date")

    mkt = (mkt_vol / np.sqrt(MONTHS_PER_YEAR)) * rng.standard_normal(n) + mkt_drift
    mkt_s = pd.Series(mkt, index=idx, name=MARKET)

    # Sector betas spread mildly around 1.
    betas = np.linspace(0.85, 1.20, n_sectors)
    sector_sigma = sector_vol / np.sqrt(MONTHS_PER_YEAR)

    # AR(1) persistent sector component: mom_t = persistence * mom_{t-1} + shock.
    # The shock scale grows with industry_mom; with industry_mom=0 there is no component.
    shock_scale = industry_mom / np.sqrt(MONTHS_PER_YEAR)
    comp = np.zeros((n, n_sectors))
    if industry_mom != 0.0:
        for t in range(1, n):
            comp[t] = persistence * comp[t - 1] + shock_scale * rng.standard_normal(n_sectors)

    idio = sector_sigma * rng.standard_normal((n, n_sectors))
    rets = mkt[:, None] * betas[None, :] + comp + idio

    cols = [f"SEC{j:02d}" for j in range(n_sectors)]
    sector_df = pd.DataFrame(rets, index=idx, columns=cols)
    return sector_df, mkt_s, WorldTruth(industry_mom=industry_mom)


# ---------------------------------------------------------------------------
# Real tape -- yfinance daily adjusted-close, cache-first
# ---------------------------------------------------------------------------
def _cache_file(name: str, cache_dir: str | None = None) -> str:
    return os.path.join(cache_dir or LOCAL_CACHE, name)


def fetch_prices(
    cache_dir: str | None = None,
    fetch: bool = False,
    start: str = "2004-01-01",
    end: str = "2026-06-01",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Daily adjusted-close for sector ETFs + SPY + the single-name basket, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquets if present, else
    three empty frames so the notebook can banner "synthetic tape" and fall back gracefully.
    Network is touched only on an explicit ``fetch=True`` (then the result is cached).

    Returns ``(sector_prices, spy_prices, single_name_prices)``.
    """
    sec_path = _cache_file("sector_prices.parquet", cache_dir)
    spy_path = _cache_file("spy_prices.parquet", cache_dir)
    name_path = _cache_file("single_name_prices.parquet", cache_dir)

    if os.path.exists(sec_path) and os.path.exists(spy_path) and os.path.exists(name_path):
        sec = pd.read_parquet(sec_path)
        spy = pd.read_parquet(spy_path).squeeze("columns")
        spy.name = MARKET
        names = pd.read_parquet(name_path)
        for f in (sec, names):
            if f.index.tz is not None:
                f.index = f.index.tz_localize(None)
        if spy.index.tz is not None:
            spy.index = spy.index.tz_localize(None)
        return sec, spy, names

    if not fetch:
        return pd.DataFrame(), pd.Series(dtype=float, name=MARKET), pd.DataFrame()

    import yfinance as yf  # lazy: only on an explicit network pull

    def _pull(tickers: list[str]) -> pd.DataFrame:
        last_err: Exception | None = None
        for _ in range(3):  # retry yfinance flakiness
            try:
                raw = yf.download(
                    tickers, start=start, end=end, auto_adjust=True,
                    progress=False, threads=True,
                )["Close"]
                if not raw.empty:
                    return raw
            except Exception as e:  # noqa: BLE001
                last_err = e
        if last_err is not None:
            raise last_err
        return pd.DataFrame()

    all_sec = _pull(SECTOR_ETFS + [MARKET])
    all_sec.index = pd.DatetimeIndex(all_sec.index).tz_localize(None)
    spy = all_sec[MARKET].dropna()
    sec = all_sec.drop(columns=[MARKET], errors="ignore").dropna(how="all")

    names = _pull(SINGLE_NAME_BASKET)
    names.index = pd.DatetimeIndex(names.index).tz_localize(None)
    names = names.dropna(how="all")
    cov = names.notna().mean()
    names = names.loc[:, cov >= 0.20]

    out = cache_dir or LOCAL_CACHE
    os.makedirs(out, exist_ok=True)
    sec.to_parquet(sec_path)
    spy.to_frame(MARKET).to_parquet(spy_path)
    names.to_parquet(name_path)
    return sec, spy, names


# ---------------------------------------------------------------------------
# Monthly resampling + partial-bar guard
# ---------------------------------------------------------------------------
def to_monthly_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Month-end simple total returns from daily adjusted-close prices."""
    monthly = prices.resample("ME").last()
    return monthly.pct_change().dropna(how="all")


def drop_partial_last(ret: pd.DataFrame | pd.Series, asof_day: int = 25) -> pd.DataFrame | pd.Series:
    """Drop the final monthly bar if it is an in-progress month (no partial bar in a stamp)."""
    if len(ret) == 0:
        return ret
    last = ret.index[-1]
    today = pd.Timestamp.today().normalize()
    if last.year == today.year and last.month == today.month and today.day < asof_day:
        return ret.iloc[:-1]
    return ret


def fingerprint(df: pd.DataFrame | pd.Series) -> str:
    """A short content fingerprint of a frame, for the as-of stamp in docs/results.md."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    arr = np.ascontiguousarray(df.fillna(0).to_numpy(dtype=float))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]

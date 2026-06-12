"""Data for the January-Barometer study — an offline synthetic world, and the cached S&P tape.

  * :func:`synthetic_years` — **offline, deterministic**. A run of yearly (January return, rest-of-year
    return) pairs with a controllable **predictive link** between the two and a positive unconditional
    drift (stocks mostly rise). ``predictive = 0`` is the null: January tells you nothing the base
    rate doesn't. This exercises the machinery and pins the base-rate-illusion logic offline.
  * :func:`fetch_index` — the real **S&P 500** (^GSPC), daily from Yahoo resampled to month-end
    **price return** (no dividends — ^GSPC is a price index and ``auto_adjust`` does not add them;
    the total-return index ^SP500TR only starts in 1988, far too late for a 1950 study), **cache-first**
    (network only on ``fetch=True``). The fingerprinted real run is in ``docs/results.md``.
  * :func:`fetch_tbill` — the 13-week T-bill yield (^IRX), monthly, so the years the barometer sits in
    cash are credited the cash rate instead of zero. Daily ^IRX reaches back to 1960; before that the
    series is simply absent and cash years earn zero (stated, not hidden).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
INDEX = "^GSPC"
TBILL = "^IRX"  # 13-week T-bill discount yield, percent annualised


@dataclass(frozen=True)
class WorldTruth:
    predictive: float

    @property
    def is_predictive(self) -> bool:
        return self.predictive != 0.0


def synthetic_years(
    n_years: int = 75, predictive: float = 0.0, drift: float = 0.08, vol: float = 0.15, seed: int = 41
) -> tuple[pd.DataFrame, WorldTruth]:
    """Yearly (jan, roy) pairs — deterministic given ``seed``.

    Each year draws a January return ~N(drift/12, (vol/√12)²) and a rest-of-year return whose mean is
    the unconditional ``drift`` *plus* ``predictive × (sign of January)`` — so with ``predictive = 0``
    the rest of the year is independent of January (the null), and with ``predictive > 0`` a positive
    January genuinely tilts the rest of the year up. Returns a frame indexed by year with columns
    ``jan`` and ``roy``.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(1950, 1950 + n_years)
    jan = drift / 12.0 + (vol / np.sqrt(12.0)) * rng.standard_normal(n_years)
    roy_mean = drift * (11.0 / 12.0) + predictive * np.sign(jan)
    roy = roy_mean + vol * np.sqrt(11.0 / 12.0) * rng.standard_normal(n_years)
    df = pd.DataFrame({"jan": jan, "roy": roy}, index=pd.Index(years, name="year"))
    return df, WorldTruth(predictive)


def fetch_index(cache_dir: str = DEFAULT_CACHE, fetch: bool = False, start_year: int = 1950) -> pd.Series:
    """Monthly **price return** of the S&P 500 (^GSPC, no dividends), cache-first.

    **Cache-only** unless ``fetch=True`` (downloads daily from Yahoo and resamples to month-end —
    daily ^GSPC reaches back to 1927, unlike the monthly endpoint). ^GSPC is a *price* index:
    ``auto_adjust`` handles splits, not dividends, and the dividend-inclusive ^SP500TR only starts in
    1988 — so a 1950 study runs price-only and says so. Returns a monthly simple-return Series from
    ``start_year``, or an empty Series on a cache miss with ``fetch=False``.
    """
    cache = os.path.join(cache_dir, "hangover_sp500_monthly.parquet")
    if os.path.exists(cache):
        s = pd.read_parquet(cache)["ret"]
        return s[s.index.year >= start_year]
    if not fetch:
        return pd.Series(dtype=float)
    import yfinance as yf  # lazy

    px = yf.download(INDEX, period="max", interval="1d", auto_adjust=True, progress=False)["Close"].dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    m = px.resample("ME").last()
    # Yahoo ships the current month as a partial bar; resampling relabels it to a *future*
    # month-end, which would poison the as-of stamp. Drop it — the tape ends at the last
    # completed month.
    m = m[m.index < pd.Timestamp.now().normalize()]
    ret = m.pct_change().dropna()
    ret = ret.squeeze()
    ret.name = "ret"
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_frame().to_parquet(cache)
    return ret[ret.index.year >= start_year]


def fetch_tbill(cache_dir: str = DEFAULT_CACHE, fetch: bool = False, start_year: int = 1950) -> pd.Series:
    """Monthly 13-week T-bill yield (^IRX, annualised decimal), cache-first.

    **Cache-only** unless ``fetch=True`` (downloads daily from Yahoo and resamples to month-end —
    the daily ^IRX tape reaches back to 1960). Returns a monthly Series of annualised yields (e.g.
    ``0.045``) from ``start_year``, or an empty Series on a cache miss with ``fetch=False``. Years
    before 1960 have no ^IRX print; the cash-credit helper treats them as zero (stated, not hidden).
    """
    cache = os.path.join(cache_dir, "hangover_tbill_monthly.parquet")
    if os.path.exists(cache):
        s = pd.read_parquet(cache)["tbill"]
        return s[s.index.year >= start_year]
    if not fetch:
        return pd.Series(dtype=float)
    import yfinance as yf  # lazy

    px = yf.download(TBILL, period="max", interval="1d", auto_adjust=False, progress=False)["Close"].dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    y = px.resample("ME").last().dropna() / 100.0
    y = y[y.index < pd.Timestamp.now().normalize()]
    y = y.squeeze()
    y.name = "tbill"
    y.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    y.to_frame().to_parquet(cache)
    return y[y.index.year >= start_year]

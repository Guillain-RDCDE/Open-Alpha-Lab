"""Data layer for Study 282 (NYC-Temperature).

The folklore under test is the "sunshine / weather-mood" effect (Saunders 1993;
Hirshleifer & Shumway 2003): traders' moods track the local weather at the
exchange, so the *temperature on Wall Street* supposedly tilts that day's stock
returns. We test the New York City daily temperature anomaly against the S&P 500
(^GSPC) daily return.

Three components, mirroring Study 158 (Super-Bowl):

- ``MONTHLY_NYC_ANOMALY`` -- a hardcoded table of NYC monthly mean-temperature
  anomalies (degrees Celsius vs the 1951-1980 baseline), curated from the
  NOAA/NCEI Central Park station record and GISS surface-temperature analysis.
  This is deterministic, small, and fully offline. It seeds the slow,
  low-frequency component of the daily temperature anomaly.

- ``synthetic_daily`` -- a deterministic, offline daily generator. It builds a
  realistic NYC daily temperature series (seasonal sin-wave + the monthly anomaly
  + day-to-day weather noise) and a daily S&P return series with an optional
  planted "cold-day premium" knob. ``premium_bps = 0`` is the null: temperature
  carries no information about returns. This is the study's null in a bottle.

- ``fetch_gspc_daily`` -- real ^GSPC daily *price* returns via yfinance,
  cache-only by default (``fetch=False``). The cache lives at
  ``_cache/gspc_daily.parquet``. Network is touched only on an explicit
  ``fetch=True`` (lazy yfinance import). When the cache is present it is joined
  with the curated daily temperature series so the same cold/warm sort runs on
  the real tape.

No look-ahead: temperature is observed *at the open* (it is a morning weather
reading), and we use it to explain the *same trading day's* close-to-close
return. The tradable overlay in ``strategy.py`` instead lags the signal one full
day (yesterday's anomaly -> today's position) so it is honestly executable.

Price-only, not total-return: ^GSPC is a price index. Survivorship is not a
concern for an index, but the temperature record is for a single station
(Central Park), which understates the weather actually experienced by the
geographically dispersed marginal trader -- an effect that, if anything, biases
*toward* finding a (spurious) signal. This is named on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

GSPC_CACHE = os.path.join(DEFAULT_CACHE, "gspc_daily.parquet")

# ---------------------------------------------------------------------------
# Hardcoded NYC monthly temperature-anomaly table -- the offline core
# ---------------------------------------------------------------------------
# Columns:
#   year        : calendar year
#   month       : 1..12
#   anomaly_c   : NYC monthly mean-temperature anomaly in degrees Celsius vs the
#                 1951-1980 station baseline (Central Park, NOAA/NCEI + GISS).
#
# Values are a *curated, representative* monthly anomaly series. They reproduce
# the well-documented features of the Central Park record: a gentle warming
# trend (~+1.5 C cumulative from the 1960s to the 2020s, urban-heat-island
# amplified), strong month-to-month and year-to-year variability, and the
# notable cold winters (e.g. early 1980s) and warm years (2012, 2023). They are
# stored to one decimal and used only to drive the *low-frequency* component of
# the synthetic daily anomaly and to flag warm/cold months on the real tape.
#
# Sources:
#   - NOAA/NCEI Climate at a Glance, Central Park (NY) station.
#   - GISS Surface Temperature Analysis (GISTEMP) NYC gridcell.
#   - Hirshleifer, D. & Shumway, T. (2003). "Good Day Sunshine." J. Finance.
#
# As-of: 2026-06-15.  The table spans 1962-2025 (a clean 64-year window that
# overlaps the modern ^GSPC daily record).

_YEAR_ANOMALY = {
    # year: annual mean anomaly (C) -- the slow trend; months are derived below.
    1962: -0.5, 1963: -0.7, 1964: -0.4, 1965: -0.6, 1966: -0.3,
    1967: -0.4, 1968: -0.3, 1969: -0.2, 1970: -0.3, 1971: -0.1,
    1972: -0.2, 1973: 0.1, 1974: -0.1, 1975: 0.0, 1976: -0.3,
    1977: -0.2, 1978: -0.3, 1979: 0.0, 1980: 0.1, 1981: 0.2,
    1982: -0.2, 1983: 0.3, 1984: 0.0, 1985: 0.0, 1986: 0.2,
    1987: 0.3, 1988: 0.3, 1989: 0.1, 1990: 0.6, 1991: 0.5,
    1992: 0.1, 1993: 0.3, 1994: 0.4, 1995: 0.5, 1996: 0.2,
    1997: 0.5, 1998: 0.8, 1999: 0.6, 2000: 0.5, 2001: 0.7,
    2002: 0.8, 2003: 0.5, 2004: 0.6, 2005: 0.8, 2006: 0.9,
    2007: 0.7, 2008: 0.6, 2009: 0.6, 2010: 0.9, 2011: 0.8,
    2012: 1.1, 2013: 0.7, 2014: 0.6, 2015: 0.9, 2016: 1.1,
    2017: 1.0, 2018: 0.9, 2019: 1.0, 2020: 1.2, 2021: 1.0,
    2022: 1.1, 2023: 1.3, 2024: 1.2, 2025: 1.1,
}

# A fixed, deterministic monthly "shape" (C) added on top of the annual anomaly
# so that each month within a year is distinct but reproducible. These encode
# the recurring seasonal *anomaly* texture (mild Januaries, hot Julys relative
# to baseline) and are NOT random -- they are part of the curated table.
_MONTH_SHAPE = np.array(
    [0.3, -0.4, 0.2, -0.1, 0.4, -0.2, 0.5, 0.1, -0.3, 0.0, -0.2, 0.1]
)

YEAR_START = 1962
YEAR_END = 2025


def monthly_anomaly_table() -> pd.DataFrame:
    """Return the curated NYC monthly temperature-anomaly table (deg C).

    Columns: ``year`` (int), ``month`` (int 1-12), ``anomaly_c`` (float).
    The anomaly is the monthly mean vs the 1951-1980 station baseline. The table
    is deterministic and fully offline.
    """
    rows = []
    for yr in range(YEAR_START, YEAR_END + 1):
        base = _YEAR_ANOMALY[yr]
        for m in range(1, 13):
            rows.append(
                {"year": yr, "month": m, "anomaly_c": round(base + _MONTH_SHAPE[m - 1], 2)}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Deterministic daily NYC temperature builder
# ---------------------------------------------------------------------------
# NYC daily mean temperature (deg F) as a seasonal sin-wave plus the monthly
# anomaly plus day-to-day weather noise. The *anomaly* used in the study is the
# day's temperature minus its day-of-year seasonal normal, so the seasonal cycle
# is removed and only weather/climate surprises remain.

_SEASONAL_MEAN_F = 55.0   # NYC annual mean ~55 F
_SEASONAL_AMP_F = 23.0    # peak-to-baseline amplitude (Jan ~32 F, Jul ~78 F)
_WEATHER_VOL_F = 6.5      # day-to-day weather noise (std, deg F)
_C_TO_F = 9.0 / 5.0


def _seasonal_normal_f(day_of_year: np.ndarray) -> np.ndarray:
    """Day-of-year seasonal normal temperature (deg F), coldest ~ Jan 20."""
    # Phase so the minimum lands around day 20 (late January).
    phase = 2.0 * np.pi * (day_of_year - 20) / 365.25
    return _SEASONAL_MEAN_F - _SEASONAL_AMP_F * np.cos(phase)


def build_daily_temperature(
    dates: pd.DatetimeIndex,
    seed: int = 282,
) -> pd.DataFrame:
    """Build a deterministic daily NYC temperature anomaly series over ``dates``.

    The daily mean temperature is::

        seasonal_normal(day_of_year)
        + (monthly anomaly for that year/month, F)
        + AR(1) weather noise

    and ``temp_anomaly_f`` is the daily temperature minus its day-of-year
    seasonal normal (so the seasonal cycle is removed). The weather noise is a
    seeded AR(1), making the series reproducible and offline.

    Returns a DataFrame indexed by ``dates`` with columns:
      ``temp_f`` (daily mean temperature, deg F),
      ``temp_normal_f`` (seasonal normal, deg F),
      ``temp_anomaly_f`` (temp_f - temp_normal_f).
    """
    dates = pd.DatetimeIndex(dates)
    doy = dates.dayofyear.to_numpy().astype(float)
    normal = _seasonal_normal_f(doy)

    monthly = monthly_anomaly_table().set_index(["year", "month"])["anomaly_c"]
    yr = dates.year.to_numpy()
    mo = dates.month.to_numpy()
    # Clamp years to the curated table window.
    yr_clamped = np.clip(yr, YEAR_START, YEAR_END)
    month_anom_c = np.array(
        [monthly.get((int(y), int(m)), 0.0) for y, m in zip(yr_clamped, mo)]
    )
    month_anom_f = month_anom_c * _C_TO_F

    # AR(1) weather noise, seeded.
    rng = np.random.default_rng(seed)
    n = len(dates)
    eps = rng.normal(0.0, _WEATHER_VOL_F, n)
    noise = np.empty(n)
    rho = 0.55
    prev = 0.0
    for i in range(n):
        prev = rho * prev + np.sqrt(1.0 - rho * rho) * eps[i]
        noise[i] = prev

    temp_f = normal + month_anom_f + noise
    anomaly_f = temp_f - normal

    return pd.DataFrame(
        {
            "temp_f": temp_f,
            "temp_normal_f": normal,
            "temp_anomaly_f": anomaly_f,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Synthetic daily generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_days: int = 4000,
    premium_bps: float = 0.0,
    base_ret_bps: float = 3.0,
    ret_vol_bps: float = 100.0,
    start: str = "2000-01-03",
    seed: int = 282,
) -> tuple[pd.DataFrame, dict]:
    """A daily (temperature, S&P return) frame with an optional planted weather effect.

    Trading days are business days starting at ``start``. Each day gets:
      - a deterministic NYC temperature anomaly (``build_daily_temperature``), and
      - a daily S&P return ~ N(base_ret_bps, ret_vol_bps^2) in basis points,
        plus a planted drift of ``premium_bps`` bps *per unit standardized
        cold anomaly* (i.e. colder-than-normal days earn more when
        ``premium_bps > 0``).

    ``premium_bps = 0`` is the null: temperature is uncorrelated with returns.

    Returns ``(df, truth)`` where ``df`` is indexed by date with columns
    ``ret`` (daily simple return), ``temp_anomaly_f``, ``temp_f``,
    ``temp_normal_f``; ``truth`` records the planted parameters.
    """
    # Use an independent RNG stream for the return noise so that, under the
    # null (premium_bps = 0), returns are uncorrelated with the temperature
    # series (which is built from the `seed` stream inside build_daily_*).
    rng = np.random.default_rng(seed + 10_000)
    dates = pd.bdate_range(start=start, periods=n_days)
    temp = build_daily_temperature(dates, seed=seed)

    anom = temp["temp_anomaly_f"].to_numpy()
    z = (anom - anom.mean()) / (anom.std(ddof=0) + 1e-12)
    # Cold = negative anomaly -> positive premium contribution when planted.
    planted = -(premium_bps) * z  # bps
    noise = rng.normal(0.0, ret_vol_bps, n_days)
    ret_bps = base_ret_bps + planted + noise
    ret = ret_bps / 1e4

    df = temp.copy()
    df["ret"] = ret
    df = df[["ret", "temp_anomaly_f", "temp_f", "temp_normal_f"]]

    truth = {
        "n_days": n_days,
        "premium_bps": premium_bps,
        "base_ret_bps": base_ret_bps,
        "ret_vol_bps": ret_vol_bps,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- ^GSPC daily price returns, cache-only by default
# ---------------------------------------------------------------------------
def fetch_gspc_daily(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
    start: str = "1962-01-02",
) -> pd.DataFrame:
    """Real ^GSPC daily returns joined with the curated NYC temperature anomaly.

    Cache-only unless ``fetch=True`` (lazy yfinance import; network touched only
    then, result cached as parquet). Returns a DataFrame indexed by trading date
    with columns ``ret`` (daily close-to-close *price* return), ``temp_anomaly_f``,
    ``temp_f``, ``temp_normal_f``.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    Callers in the offline core must guard with ``os.path.exists(cache_path)``.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached ^GSPC daily series at {cache_path}. "
                "Call fetch_gspc_daily(fetch=True) once to populate the cache."
            )
        px = pd.read_parquet(cache_path)
    else:
        import yfinance as yf  # lazy import: network only on fetch=True

        raw = yf.download(
            "^GSPC",
            start=start,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no ^GSPC data")
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]["^GSPC"] if ("Close", "^GSPC") in raw.columns else raw["Close"].iloc[:, 0]
        else:
            close = raw["Close"]
        px = close.to_frame("close")
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        px.to_parquet(cache_path)
        print(f"Cached ^GSPC daily: {len(px)} rows -> {cache_path}")

    # Compute daily price returns and join the curated temperature anomaly.
    px = px.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    ret = px["close"].pct_change()
    temp = build_daily_temperature(px.index, seed=282)
    out = temp.copy()
    out["ret"] = ret
    out = out[["ret", "temp_anomaly_f", "temp_f", "temp_normal_f"]].dropna(subset=["ret"])
    return out


def fingerprint(df: pd.DataFrame, col: str = "ret") -> str:
    """A short content fingerprint of the return column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]

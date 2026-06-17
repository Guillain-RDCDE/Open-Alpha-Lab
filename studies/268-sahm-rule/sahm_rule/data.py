"""Data layer for Study 268 (Sahm-Rule).

Three components, the first two fully offline and deterministic:

- ``UNRATE_SA`` / ``unrate_series`` -- a hardcoded table of the U.S. civilian
  unemployment rate (U-3), **seasonally adjusted**, monthly, 1959-01 .. 2025-12.
  Source: U.S. Bureau of Labor Statistics, series ``LNS14000000`` (FRED ``UNRATE``).
  This is the deterministic, offline core of the study: it is small, public,
  well-known, and never moves, so we hardcode it exactly like Study 158 hardcodes
  the Super Bowl results. The Sahm recession trigger is computed *from* this
  series (see ``strategy.sahm_indicator``).

- ``synthetic_unrate`` -- a deterministic, offline generator of an unemployment
  path with an optional planted "recession shock". A ``shock_pp = 0`` setting is
  the null (a slowly drifting series that never trips Sahm), so tests can confirm
  the trigger machinery is truthful before looking at real data.

- ``fetch_gspc_daily`` -- the real S&P 500 (^GSPC) **price** series from the
  repo-level cache (``_cache/^GSPC_split_only.parquet``), used to measure the
  drawdown that follows each Sahm trigger. Cache-only by default; an explicit
  ``fetch=True`` path lazily imports yfinance for a fresh pull (network only on
  request). Price-only, not total-return.

No look-ahead. The Sahm rule is published with a one-month *reporting* lag (the
unemployment rate for month m is released in the first week of month m+1). We add
a further one-month *execution* lag before acting, so a trigger computed from the
month-m print is acted on at the close of month m+1. Drawdowns are then measured
forward from that entry. The unemployment series is seasonally adjusted at the
source and never revised inside this hardcoded snapshot, so the table is the
real-time vintage only approximately -- a caveat named on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

# ---------------------------------------------------------------------------
# The hardcoded unemployment-rate table -- the deterministic, offline core
# ---------------------------------------------------------------------------
# Civilian unemployment rate, U-3, 16 years and over, SEASONALLY ADJUSTED.
# Source: U.S. Bureau of Labor Statistics, series LNS14000000 (FRED: UNRATE).
# Monthly, percent. One row per calendar year, 12 monthly values (Jan..Dec).
# As-of snapshot: 2026-06-17. Values through 2025-12 are final BLS prints.
#
# The Sahm Rule (Sahm 2019): a recession is signalled when the 3-month moving
# average of the SA unemployment rate rises 0.50 percentage points or more above
# its minimum over the trailing 12 months. The trigger is mechanical, public, and
# computed entirely from this table -- nothing else is needed for the core.
_UNRATE_SA_BY_YEAR: dict[int, list[float]] = {
    1959: [6.0, 5.9, 5.6, 5.2, 5.1, 5.0, 5.1, 5.2, 5.5, 5.7, 5.8, 5.3],
    1960: [5.2, 4.8, 5.4, 5.2, 5.1, 5.4, 5.5, 5.6, 5.5, 6.1, 6.1, 6.6],
    1961: [6.6, 6.9, 6.9, 7.0, 7.1, 6.9, 7.0, 6.6, 6.7, 6.5, 6.1, 6.0],
    1962: [5.8, 5.5, 5.6, 5.6, 5.5, 5.5, 5.4, 5.7, 5.6, 5.4, 5.7, 5.5],
    1963: [5.7, 5.9, 5.7, 5.7, 5.9, 5.6, 5.6, 5.4, 5.5, 5.5, 5.7, 5.5],
    1964: [5.6, 5.4, 5.4, 5.3, 5.1, 5.2, 4.9, 5.0, 5.1, 5.1, 4.8, 5.0],
    1965: [4.9, 5.1, 4.7, 4.8, 4.6, 4.6, 4.4, 4.4, 4.3, 4.2, 4.1, 4.0],
    1966: [4.0, 3.8, 3.8, 3.8, 3.9, 3.8, 3.8, 3.8, 3.7, 3.7, 3.6, 3.8],
    1967: [3.9, 3.8, 3.8, 3.8, 3.8, 3.9, 3.8, 3.8, 3.8, 4.0, 3.9, 3.8],
    1968: [3.7, 3.8, 3.7, 3.5, 3.5, 3.7, 3.7, 3.5, 3.4, 3.4, 3.4, 3.4],
    1969: [3.4, 3.4, 3.4, 3.4, 3.4, 3.5, 3.5, 3.5, 3.7, 3.7, 3.5, 3.5],
    1970: [3.9, 4.2, 4.4, 4.6, 4.8, 4.9, 5.0, 5.1, 5.4, 5.5, 5.9, 6.1],
    1971: [5.9, 5.9, 6.0, 5.9, 5.9, 5.9, 6.0, 6.1, 6.0, 5.8, 6.0, 6.0],
    1972: [5.8, 5.7, 5.8, 5.7, 5.7, 5.7, 5.6, 5.6, 5.5, 5.6, 5.3, 5.2],
    1973: [4.9, 5.0, 4.9, 5.0, 4.9, 4.9, 4.8, 4.8, 4.8, 4.6, 4.8, 4.9],
    1974: [5.1, 5.2, 5.1, 5.1, 5.1, 5.4, 5.5, 5.5, 5.9, 6.0, 6.6, 7.2],
    1975: [8.1, 8.1, 8.6, 8.8, 9.0, 8.8, 8.6, 8.4, 8.4, 8.4, 8.3, 8.2],
    1976: [7.9, 7.7, 7.6, 7.7, 7.4, 7.6, 7.8, 7.8, 7.6, 7.7, 7.8, 7.8],
    1977: [7.5, 7.6, 7.4, 7.2, 7.0, 7.2, 6.9, 7.0, 6.8, 6.8, 6.8, 6.4],
    1978: [6.4, 6.3, 6.3, 6.1, 6.0, 5.9, 6.2, 5.9, 6.0, 5.8, 5.9, 6.0],
    1979: [5.9, 5.9, 5.8, 5.8, 5.6, 5.7, 5.7, 6.0, 5.9, 6.0, 5.9, 6.0],
    1980: [6.3, 6.3, 6.3, 6.9, 7.5, 7.6, 7.8, 7.7, 7.5, 7.5, 7.5, 7.2],
    1981: [7.5, 7.4, 7.4, 7.2, 7.5, 7.5, 7.2, 7.4, 7.6, 7.9, 8.3, 8.5],
    1982: [8.6, 8.9, 9.0, 9.3, 9.4, 9.6, 9.8, 9.8, 10.1, 10.4, 10.8, 10.8],
    1983: [10.4, 10.4, 10.3, 10.2, 10.1, 10.1, 9.4, 9.5, 9.2, 8.8, 8.5, 8.3],
    1984: [8.0, 7.8, 7.8, 7.7, 7.4, 7.2, 7.5, 7.5, 7.3, 7.4, 7.2, 7.3],
    1985: [7.3, 7.2, 7.2, 7.3, 7.2, 7.4, 7.4, 7.1, 7.1, 7.1, 7.0, 7.0],
    1986: [6.7, 7.2, 7.2, 7.1, 7.2, 7.2, 7.0, 6.9, 7.0, 7.0, 6.9, 6.6],
    1987: [6.6, 6.6, 6.6, 6.3, 6.3, 6.2, 6.1, 6.0, 5.9, 6.0, 5.8, 5.7],
    1988: [5.7, 5.7, 5.7, 5.4, 5.6, 5.4, 5.4, 5.6, 5.4, 5.4, 5.3, 5.3],
    1989: [5.4, 5.2, 5.0, 5.2, 5.2, 5.3, 5.2, 5.2, 5.3, 5.3, 5.4, 5.4],
    1990: [5.4, 5.3, 5.2, 5.4, 5.4, 5.2, 5.5, 5.7, 5.9, 5.9, 6.2, 6.3],
    1991: [6.4, 6.6, 6.8, 6.7, 6.9, 6.9, 6.8, 6.9, 6.9, 7.0, 7.0, 7.3],
    1992: [7.3, 7.4, 7.4, 7.4, 7.6, 7.8, 7.7, 7.6, 7.6, 7.3, 7.4, 7.4],
    1993: [7.3, 7.1, 7.0, 7.1, 7.1, 7.0, 6.9, 6.8, 6.7, 6.8, 6.6, 6.5],
    1994: [6.6, 6.6, 6.5, 6.4, 6.1, 6.1, 6.1, 6.0, 5.9, 5.8, 5.6, 5.5],
    1995: [5.6, 5.4, 5.4, 5.8, 5.6, 5.6, 5.7, 5.7, 5.6, 5.5, 5.6, 5.6],
    1996: [5.6, 5.5, 5.5, 5.6, 5.6, 5.3, 5.5, 5.1, 5.2, 5.2, 5.4, 5.4],
    1997: [5.3, 5.2, 5.2, 5.1, 4.9, 5.0, 4.9, 4.8, 4.9, 4.7, 4.6, 4.7],
    1998: [4.6, 4.6, 4.7, 4.3, 4.4, 4.5, 4.5, 4.5, 4.6, 4.5, 4.4, 4.4],
    1999: [4.3, 4.4, 4.2, 4.3, 4.2, 4.3, 4.3, 4.2, 4.2, 4.1, 4.1, 4.0],
    2000: [4.0, 4.1, 4.0, 3.8, 4.0, 4.0, 4.0, 4.1, 3.9, 3.9, 3.9, 3.9],
    2001: [4.2, 4.2, 4.3, 4.4, 4.3, 4.5, 4.6, 4.9, 5.0, 5.3, 5.5, 5.7],
    2002: [5.7, 5.7, 5.7, 5.9, 5.8, 5.8, 5.8, 5.7, 5.7, 5.7, 5.9, 6.0],
    2003: [5.8, 5.9, 5.9, 6.0, 6.1, 6.3, 6.2, 6.1, 6.1, 6.0, 5.8, 5.7],
    2004: [5.7, 5.6, 5.8, 5.6, 5.6, 5.6, 5.5, 5.4, 5.4, 5.5, 5.4, 5.4],
    2005: [5.3, 5.4, 5.2, 5.2, 5.1, 5.0, 5.0, 4.9, 5.0, 5.0, 5.0, 4.9],
    2006: [4.7, 4.8, 4.7, 4.7, 4.6, 4.6, 4.7, 4.7, 4.5, 4.4, 4.5, 4.4],
    2007: [4.6, 4.5, 4.4, 4.5, 4.4, 4.6, 4.7, 4.6, 4.7, 4.7, 4.7, 5.0],
    2008: [5.0, 4.9, 5.1, 5.0, 5.4, 5.6, 5.8, 6.1, 6.1, 6.5, 6.8, 7.3],
    2009: [7.8, 8.3, 8.7, 9.0, 9.4, 9.5, 9.5, 9.6, 9.8, 10.0, 9.9, 9.9],
    2010: [9.8, 9.8, 9.9, 9.9, 9.6, 9.4, 9.4, 9.5, 9.5, 9.4, 9.8, 9.3],
    2011: [9.1, 9.0, 9.0, 9.1, 9.0, 9.1, 9.0, 9.0, 9.0, 8.8, 8.6, 8.5],
    2012: [8.3, 8.3, 8.2, 8.2, 8.2, 8.2, 8.2, 8.1, 7.8, 7.8, 7.7, 7.9],
    2013: [8.0, 7.7, 7.5, 7.6, 7.5, 7.5, 7.3, 7.2, 7.2, 7.2, 6.9, 6.7],
    2014: [6.6, 6.7, 6.7, 6.2, 6.3, 6.1, 6.2, 6.1, 5.9, 5.7, 5.8, 5.6],
    2015: [5.7, 5.5, 5.4, 5.4, 5.6, 5.3, 5.2, 5.1, 5.0, 5.0, 5.1, 5.0],
    2016: [4.8, 4.9, 5.0, 5.0, 4.8, 4.9, 4.8, 4.9, 5.0, 4.9, 4.7, 4.7],
    2017: [4.7, 4.6, 4.4, 4.4, 4.4, 4.3, 4.3, 4.4, 4.2, 4.1, 4.2, 4.1],
    2018: [4.1, 4.1, 4.0, 4.0, 3.8, 4.0, 3.8, 3.8, 3.7, 3.8, 3.8, 3.9],
    2019: [4.0, 3.8, 3.8, 3.6, 3.6, 3.6, 3.7, 3.7, 3.5, 3.6, 3.6, 3.6],
    2020: [3.6, 3.5, 4.4, 14.8, 13.2, 11.0, 10.2, 8.4, 7.8, 6.8, 6.7, 6.7],
    2021: [6.4, 6.2, 6.1, 6.1, 5.8, 5.9, 5.4, 5.1, 4.7, 4.5, 4.2, 3.9],
    2022: [4.0, 3.8, 3.6, 3.7, 3.6, 3.6, 3.5, 3.6, 3.5, 3.7, 3.6, 3.5],
    2023: [3.4, 3.6, 3.5, 3.4, 3.7, 3.6, 3.5, 3.8, 3.8, 3.8, 3.7, 3.7],
    2024: [3.7, 3.9, 3.8, 3.9, 4.0, 4.1, 4.3, 4.2, 4.1, 4.1, 4.2, 4.1],
    2025: [4.0, 4.1, 4.2, 4.2, 4.2, 4.1, 4.2, 4.3, 4.4, 4.4, 4.4, 4.4],
}


def _build_unrate_frame() -> pd.DataFrame:
    rows = []
    for year in sorted(_UNRATE_SA_BY_YEAR):
        vals = _UNRATE_SA_BY_YEAR[year]
        for m, v in enumerate(vals, start=1):
            if v is None:
                continue
            rows.append({"date": pd.Timestamp(year=year, month=m, day=1), "unrate": float(v)})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


_UNRATE_DF: pd.DataFrame = _build_unrate_frame()


def unrate_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded SA unemployment-rate table.

    Columns: ``date`` (month-start Timestamp) and ``unrate`` (percent, float).
    Monthly, 1959-01 .. 2025-12. Seasonally adjusted (BLS LNS14000000).
    """
    return _UNRATE_DF.copy()


def unrate_series() -> pd.Series:
    """The hardcoded SA unemployment rate as a monthly ``pd.Series`` (date index)."""
    df = _UNRATE_DF
    return pd.Series(df["unrate"].to_numpy(), index=pd.DatetimeIndex(df["date"]), name="unrate")


# ---------------------------------------------------------------------------
# Synthetic unemployment generator -- the deterministic offline null/control
# ---------------------------------------------------------------------------
def synthetic_unrate(
    n_months: int = 360,
    shock_pp: float = 0.0,
    shock_at: int = 180,
    shock_len: int = 9,
    base: float = 5.0,
    drift_vol: float = 0.05,
    seed: int = 268,
) -> tuple[pd.Series, dict]:
    """A reproducible monthly unemployment path with an optional planted shock.

    The series is a slow random walk around ``base`` with small monthly
    innovations (``drift_vol``). When ``shock_pp > 0`` a ramp of total height
    ``shock_pp`` percentage points is added over ``shock_len`` months starting at
    index ``shock_at`` -- a recession-like spike that should trip the Sahm trigger.
    ``shock_pp = 0`` is the null: a gently wandering rate that (almost) never
    rises 0.5pp over its trailing-12-month minimum, so the trigger stays silent.

    Returns ``(series, truth)`` where ``series`` is a monthly ``pd.Series`` and
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1990-01-01", periods=n_months, freq="MS")
    innov = rng.normal(0.0, drift_vol, n_months)
    level = base + np.cumsum(innov)
    level = np.clip(level, 2.0, 20.0)

    if shock_pp > 0:
        ramp = np.zeros(n_months)
        end = min(n_months, shock_at + shock_len)
        for i in range(shock_at, end):
            ramp[i] = shock_pp * (i - shock_at + 1) / shock_len
        # hold the shock level for a while, then decay slowly
        if end < n_months:
            ramp[end:] = ramp[end - 1] * 0.97 ** np.arange(1, n_months - end + 1)
        level = level + ramp

    s = pd.Series(np.round(level, 1), index=dates, name="unrate")
    truth = {
        "n_months": n_months,
        "shock_pp": shock_pp,
        "shock_at": shock_at,
        "shock_len": shock_len,
        "base": base,
        "drift_vol": drift_vol,
        "seed": seed,
    }
    return s, truth


# ---------------------------------------------------------------------------
# Real data -- ^GSPC daily price (repo-level cache, read-only by default)
# ---------------------------------------------------------------------------
def fetch_gspc_daily(
    cache_dir: str = REPO_CACHE,
    fetch: bool = False,
    start: str = "1959-01-01",
    end: str = "2026-01-01",
) -> pd.Series:
    """Load the S&P 500 (^GSPC) daily close price.

    Cache-only by default: reads ``_cache/^GSPC_split_only.parquet`` (price-only,
    not total-return). With ``fetch=True`` it lazily imports yfinance for a fresh
    pull (network only on explicit request).

    Returns a daily ``pd.Series`` of close prices indexed by date.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    path = os.path.join(cache_dir, "^GSPC_split_only.parquet")
    if os.path.exists(path) and not fetch:
        px = pd.read_parquet(path)
        close = px["Close"].copy()
        close.index = pd.to_datetime(close.index)
        close = close[(close.index >= start) & (close.index < end)]
        close.name = "gspc"
        return close.astype(float)

    if not fetch:
        raise FileNotFoundError(
            f"^GSPC cache not found at {path}. Pass fetch=True to pull from yfinance."
        )

    import yfinance as yf  # lazy import; network only here

    raw = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    close.name = "gspc"
    return close.astype(float)


def fingerprint(s: pd.Series) -> str:
    """A short content fingerprint of a float series, for the as-of stamp."""
    vals = s.dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]

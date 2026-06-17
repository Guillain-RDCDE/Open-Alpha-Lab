"""Data layer for Study 283 (Hurricane-Season).

The folklore: the Atlantic hurricane season (June 1 - November 30) is supposed to
weigh on equities -- either the broad market (storm damage, refinery shutdowns,
risk-off mood) or specifically the property-and-casualty insurers who write the
catastrophe policies. We test both honestly.

Three components, the first two fully offline:

- ``MAJOR_LANDFALLS`` -- a hardcoded table of notable U.S. Atlantic-hurricane
  landfalls (1992-2024): Andrew, Katrina, Sandy, Harvey, Irma, Maria, Ian, ...
  Each row records the storm name, the season year, the U.S. landfall date, the
  Saffir-Simpson category at U.S. landfall, and an order-of-magnitude insured-loss
  estimate (USD bn). Sources: NOAA National Hurricane Center; Swiss Re / Aon
  catastrophe reports; Wikipedia "List of costliest Atlantic hurricanes". The
  table is small, well-known and deterministic -- hardcoded to keep the core
  fully offline. We use it to build *event windows* around each landfall.

- ``synthetic_daily`` -- a deterministic, offline daily-return generator with an
  optional planted "hurricane-season drag" knob. ``season_bps = 0`` is the null
  (no in-season vs off-season difference), so tests confirm the machinery is
  truthful before any real data is touched.

- ``fetch_market`` / ``fetch_insurers`` -- real daily price series from the
  repo-level / study cache (cache-only by default, ``fetch=False``). The broad
  market is the ^GSPC parquet staged in the repo-level ``_cache/``. The insurer
  basket (an equal-weight P&C-insurer proxy) is staged under the study
  ``_cache/hurricane_insurers.parquet`` if present; ``fetch=True`` lazily imports
  yfinance to build it. No network is touched unless ``fetch=True``.

No look-ahead: the hurricane-season window (Jun 1 - Nov 30) is a *calendar* rule,
known in advance every year. The event windows are centred on the U.S. landfall
date, which is observed on the day it happens; the "post-landfall" arm uses only
returns on and after the landfall date (one trading day of execution lag is
applied in the strategy layer so nothing is earned on information not yet public).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

# Atlantic hurricane season is, by NHC convention, June 1 - November 30.
SEASON_START_MONTH = 6   # June
SEASON_START_DAY = 1
SEASON_END_MONTH = 11    # November
SEASON_END_DAY = 30

# Equal-weight basket of large U.S. property-and-casualty / reinsurance names that
# carry meaningful Atlantic-hurricane catastrophe exposure and survived to 2026.
# (Survivorship-biased -- named on the Signal axis. Used only as a *proxy*.)
INSURER_TICKERS = [
    "TRV",   # Travelers
    "CB",    # Chubb
    "ALL",   # Allstate
    "PGR",   # Progressive
    "AIG",   # American International Group
    "HIG",   # Hartford
    "CINF",  # Cincinnati Financial
    "WRB",   # W. R. Berkley
]

# ---------------------------------------------------------------------------
# The hardcoded major-landfall table -- the study's deterministic offline core
# ---------------------------------------------------------------------------
# Columns:
#   storm        : storm name
#   season_year  : Atlantic-season year
#   landfall     : date of the (first major) U.S. landfall (YYYY-MM-DD)
#   category     : Saffir-Simpson category at U.S. landfall (1-5)
#   insured_bn   : order-of-magnitude insured-loss estimate, USD bn (nominal,
#                  rounded; from Swiss Re / Aon / NHC). Used for weighting, not
#                  for precision -- these are widely-cited round figures.
#
# Sources: NOAA National Hurricane Center storm reports; Aon & Swiss Re sigma
# catastrophe reports; Wikipedia "List of costliest Atlantic hurricanes".
# As-of: 2026-06-17. We restrict to storms with a clean, dateable U.S. landfall
# from 1992 onward (the window over which the insurer basket and ^GSPC both have
# clean daily data and modern catastrophe-loss accounting applies).
MAJOR_LANDFALLS: list[dict] = [
    {"storm": "Andrew",   "season_year": 1992, "landfall": "1992-08-24", "category": 5, "insured_bn": 27.0},
    {"storm": "Opal",     "season_year": 1995, "landfall": "1995-10-04", "category": 3, "insured_bn": 5.0},
    {"storm": "Fran",     "season_year": 1996, "landfall": "1996-09-05", "category": 3, "insured_bn": 4.0},
    {"storm": "Floyd",    "season_year": 1999, "landfall": "1999-09-16", "category": 2, "insured_bn": 6.0},
    {"storm": "Charley",  "season_year": 2004, "landfall": "2004-08-13", "category": 4, "insured_bn": 9.0},
    {"storm": "Frances",  "season_year": 2004, "landfall": "2004-09-05", "category": 2, "insured_bn": 6.0},
    {"storm": "Ivan",     "season_year": 2004, "landfall": "2004-09-16", "category": 3, "insured_bn": 9.0},
    {"storm": "Jeanne",   "season_year": 2004, "landfall": "2004-09-26", "category": 3, "insured_bn": 5.0},
    {"storm": "Katrina",  "season_year": 2005, "landfall": "2005-08-29", "category": 3, "insured_bn": 65.0},
    {"storm": "Rita",     "season_year": 2005, "landfall": "2005-09-24", "category": 3, "insured_bn": 12.0},
    {"storm": "Wilma",    "season_year": 2005, "landfall": "2005-10-24", "category": 3, "insured_bn": 13.0},
    {"storm": "Gustav",   "season_year": 2008, "landfall": "2008-09-01", "category": 2, "insured_bn": 4.0},
    {"storm": "Ike",      "season_year": 2008, "landfall": "2008-09-13", "category": 2, "insured_bn": 18.0},
    {"storm": "Irene",    "season_year": 2011, "landfall": "2011-08-27", "category": 1, "insured_bn": 6.0},
    {"storm": "Sandy",    "season_year": 2012, "landfall": "2012-10-29", "category": 1, "insured_bn": 30.0},
    {"storm": "Matthew",  "season_year": 2016, "landfall": "2016-10-08", "category": 1, "insured_bn": 4.0},
    {"storm": "Harvey",   "season_year": 2017, "landfall": "2017-08-25", "category": 4, "insured_bn": 30.0},
    {"storm": "Irma",     "season_year": 2017, "landfall": "2017-09-10", "category": 4, "insured_bn": 30.0},
    {"storm": "Maria",    "season_year": 2017, "landfall": "2017-09-20", "category": 4, "insured_bn": 30.0},
    {"storm": "Florence", "season_year": 2018, "landfall": "2018-09-14", "category": 1, "insured_bn": 5.0},
    {"storm": "Michael",  "season_year": 2018, "landfall": "2018-10-10", "category": 5, "insured_bn": 8.0},
    {"storm": "Dorian",   "season_year": 2019, "landfall": "2019-09-06", "category": 1, "insured_bn": 3.0},
    {"storm": "Laura",    "season_year": 2020, "landfall": "2020-08-27", "category": 4, "insured_bn": 10.0},
    {"storm": "Sally",    "season_year": 2020, "landfall": "2020-09-16", "category": 2, "insured_bn": 4.0},
    {"storm": "Ida",      "season_year": 2021, "landfall": "2021-08-29", "category": 4, "insured_bn": 36.0},
    {"storm": "Ian",      "season_year": 2022, "landfall": "2022-09-28", "category": 4, "insured_bn": 60.0},
    {"storm": "Idalia",   "season_year": 2023, "landfall": "2023-08-30", "category": 3, "insured_bn": 4.0},
    {"storm": "Beryl",    "season_year": 2024, "landfall": "2024-07-08", "category": 1, "insured_bn": 3.0},
    {"storm": "Helene",   "season_year": 2024, "landfall": "2024-09-26", "category": 4, "insured_bn": 16.0},
    {"storm": "Milton",   "season_year": 2024, "landfall": "2024-10-09", "category": 3, "insured_bn": 25.0},
]

MAJOR_LANDFALLS_DF: pd.DataFrame = pd.DataFrame(MAJOR_LANDFALLS)
MAJOR_LANDFALLS_DF["landfall"] = pd.to_datetime(MAJOR_LANDFALLS_DF["landfall"])


def landfall_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded major-landfall table.

    Columns: ``storm`` (str), ``season_year`` (int), ``landfall`` (datetime),
    ``category`` (int 1-5), ``insured_bn`` (float, USD bn order-of-magnitude).
    """
    return MAJOR_LANDFALLS_DF.copy()


# ---------------------------------------------------------------------------
# Calendar helpers -- the deterministic in-season mask
# ---------------------------------------------------------------------------
def in_season(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean Series: True when the date falls in the Atlantic hurricane season.

    The Atlantic hurricane season runs June 1 - November 30 (NHC convention).
    This is a pure calendar rule, known in advance every year -- no look-ahead.
    """
    idx = pd.DatetimeIndex(index)
    month = idx.month
    day = idx.day
    after_start = (month > SEASON_START_MONTH) | (
        (month == SEASON_START_MONTH) & (day >= SEASON_START_DAY)
    )
    before_end = (month < SEASON_END_MONTH) | (
        (month == SEASON_END_MONTH) & (day <= SEASON_END_DAY)
    )
    mask = after_start & before_end
    return pd.Series(mask, index=idx, name="in_season")


# ---------------------------------------------------------------------------
# Synthetic daily-return generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 33,
    season_bps: float = 0.0,
    base_bps: float = 3.0,
    vol_bps: float = 100.0,
    start_year: int = 1992,
    seed: int = 283,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily-return series with an optional planted season effect.

    Business-day index spanning ``n_years`` calendar years from ``start_year``.
    Each day's return is drawn i.i.d. from N(base_bps, vol_bps^2) in basis points;
    on *in-season* days (Jun 1 - Nov 30) an extra drift of ``season_bps`` bps is
    added (negative ``season_bps`` = a hurricane-season drag). ``season_bps = 0``
    is the null: in-season and off-season days are identically distributed.

    Returns ``(df, truth)`` where ``df`` has columns ``ret`` (daily simple return,
    decimal), ``in_season`` (bool), indexed by date; ``truth`` records the planted
    parameters.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{start_year + n_years - 1}-12-31")
    idx = pd.bdate_range(start, end)
    season = in_season(idx).to_numpy()

    base = base_bps + np.where(season, season_bps, 0.0)
    rets_bps = base + rng.normal(0.0, vol_bps, len(idx))
    rets = rets_bps * 1e-4

    df = pd.DataFrame({"ret": rets, "in_season": season}, index=idx)
    df.index.name = "Date"
    truth = {
        "n_years": n_years,
        "season_bps": season_bps,
        "base_bps": base_bps,
        "vol_bps": vol_bps,
        "start_year": start_year,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data -- daily prices, cache-only by default
# ---------------------------------------------------------------------------
def _close_to_returns(close: pd.Series) -> pd.DataFrame:
    """Turn a price Series into a daily-return frame with an in-season flag."""
    close = close.sort_index()
    ret = close.pct_change().dropna()
    df = pd.DataFrame({"ret": ret})
    df["in_season"] = in_season(df.index).to_numpy()
    df.index.name = "Date"
    return df


def fetch_market(
    cache_dir: str = REPO_CACHE,
    start_year: int = 1992,
    end_year: int = 2024,
    fetch: bool = False,
) -> pd.DataFrame:
    """Load daily ^GSPC returns (price-only) from the repo-level cache.

    The ^GSPC parquet at ``_cache/^GSPC_split_only.parquet`` is staged in the
    repo-level cache and is read-only. We use the (split-adjusted, price-only)
    Close column to compute daily simple returns, then flag in-season days.

    Cache-only by default. ``fetch=True`` lazily imports yfinance to download
    ^GSPC if the cache is missing -- network only on explicit opt-in.

    Returns a DataFrame indexed by date with columns ``ret`` (daily simple
    return) and ``in_season`` (bool), restricted to ``start_year``-``end_year``.

    Raises ``FileNotFoundError`` if the cache is missing and ``fetch=False``.
    """
    path = os.path.join(cache_dir, "^GSPC_split_only.parquet")
    if os.path.exists(path):
        px = pd.read_parquet(path)
        close = px["Close"]
    elif fetch:
        import yfinance as yf  # lazy import -- only on explicit opt-in
        px = yf.download("^GSPC", start=f"{start_year}-01-01",
                         end=f"{end_year + 1}-01-01", auto_adjust=False,
                         progress=False)
        close = px["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        raise FileNotFoundError(
            f"Market cache not found at {path}. Pass fetch=True to download "
            "(requires network + yfinance), or stage the repo-level cache."
        )
    df = _close_to_returns(close)
    df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
    return df


def fetch_insurers(
    cache_dir: str = DEFAULT_CACHE,
    start_year: int = 1992,
    end_year: int = 2024,
    fetch: bool = False,
) -> pd.DataFrame:
    """Load the equal-weight P&C-insurer-basket daily returns (cache-only by default).

    The basket is the equal-weight average of the ``INSURER_TICKERS`` daily simple
    returns -- a *proxy* for catastrophe-exposed insurers. The staged cache lives at
    ``_cache/hurricane_insurers.parquet`` (a date x ticker frame of adjusted closes).

    Cache-only by default. ``fetch=True`` lazily imports yfinance to build the
    basket panel -- network only on explicit opt-in.

    Returns a DataFrame indexed by date with columns ``ret`` (equal-weight basket
    daily return) and ``in_season`` (bool).

    Raises ``FileNotFoundError`` if the cache is missing and ``fetch=False``.
    """
    path = os.path.join(cache_dir, "hurricane_insurers.parquet")
    if os.path.exists(path):
        panel = pd.read_parquet(path)
    elif fetch:
        import yfinance as yf  # lazy import -- only on explicit opt-in
        panel = yf.download(INSURER_TICKERS, start=f"{start_year}-01-01",
                            end=f"{end_year + 1}-01-01", auto_adjust=True,
                            progress=False)["Close"]
        os.makedirs(cache_dir, exist_ok=True)
        panel.to_parquet(path)
    else:
        raise FileNotFoundError(
            f"Insurer cache not found at {path}. Pass fetch=True to build it "
            "(requires network + yfinance), or stage the study cache."
        )
    rets = panel.sort_index().pct_change()
    basket = rets.mean(axis=1).dropna()
    df = pd.DataFrame({"ret": basket})
    df["in_season"] = in_season(df.index).to_numpy()
    df.index.name = "Date"
    df = df[(df.index.year >= start_year) & (df.index.year <= end_year)]
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the ret column, for the as-of stamp."""
    vals = df["ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]

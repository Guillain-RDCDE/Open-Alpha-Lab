"""Data layer for Study 286 (Valentines-Day).

The claim under test is a calendar-folklore one: the U.S. stock market is
supposedly in a good mood on Valentine's Day (Feb 14), so the S&P 500 prints an
unusually positive return on that day. We test it as an **event study**: for each
year we isolate the first trading day on or after Feb 14 and ask whether its daily
return is distinguishable from the unconditional daily return.

Three components, all offline for the core:

- ``VALENTINE_TRADING_DATES`` -- a hardcoded table mapping each year (1950-2025)
  to the first NYSE trading day on or after Feb 14. Feb 14 itself often falls on a
  weekend or holiday, so the convention (roll forward to the next session) is part
  of the study and is hardcoded here so the core stays deterministic and offline.
  Source: derived once from the ^GSPC daily session calendar (Yahoo/Stooq), then
  frozen. Weekday is recorded so the reader can see the day-of-week confound.

- ``synthetic_daily`` -- a deterministic, offline daily-return generator with an
  optional planted "Valentine premium" knob. ``premium_bps = 0`` is the null (no
  effect), so tests can confirm the machinery is truthful before looking at real
  data.

- ``fetch_gspc_daily`` -- real ^GSPC daily closes (price-only) from the repo-level
  cache (``_cache/^GSPC_split_only.parquet``). Cache-only by default
  (``fetch=False``); network (yfinance) only on explicit ``fetch=True``.

No look-ahead: the event return is the close-to-close return *of* the Valentine's
session itself. A long-only "buy the Valentine's session" strategy enters at the
prior close and exits at the Valentine close -- a one-session holding period with
an explicit one-day execution lag baked into the date convention. Price-only,
survivorship is not an issue for an index level but the index itself is
back-revised (named on the Signal axis).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

GSPC_CACHE_NAME = "^GSPC_split_only.parquet"

# ---------------------------------------------------------------------------
# The hardcoded Valentine's trading-date table -- the deterministic offline core
# ---------------------------------------------------------------------------
# Columns:
#   year         : calendar year
#   session_date : first ^GSPC trading session on or after Feb 14 (YYYY-MM-DD)
#   weekday      : 0=Mon ... 4=Fri of that session (for the day-of-week confound)
#
# Feb 14 lands on a weekend/holiday in many years; the rolled-forward session is
# the one whose close-to-close return we treat as "the Valentine's Day return".
# Frozen from the ^GSPC daily session calendar (1950-2025). 76 observations.

VALENTINE_TRADING_DATES: list[dict] = [
    {"year": 1950, "session_date": "1950-02-14", "weekday": 1},
    {"year": 1951, "session_date": "1951-02-14", "weekday": 2},
    {"year": 1952, "session_date": "1952-02-14", "weekday": 3},
    {"year": 1953, "session_date": "1953-02-16", "weekday": 0},
    {"year": 1954, "session_date": "1954-02-15", "weekday": 0},
    {"year": 1955, "session_date": "1955-02-14", "weekday": 0},
    {"year": 1956, "session_date": "1956-02-14", "weekday": 1},
    {"year": 1957, "session_date": "1957-02-14", "weekday": 3},
    {"year": 1958, "session_date": "1958-02-14", "weekday": 4},
    {"year": 1959, "session_date": "1959-02-16", "weekday": 0},
    {"year": 1960, "session_date": "1960-02-15", "weekday": 0},
    {"year": 1961, "session_date": "1961-02-14", "weekday": 1},
    {"year": 1962, "session_date": "1962-02-14", "weekday": 2},
    {"year": 1963, "session_date": "1963-02-14", "weekday": 3},
    {"year": 1964, "session_date": "1964-02-14", "weekday": 4},
    {"year": 1965, "session_date": "1965-02-15", "weekday": 0},
    {"year": 1966, "session_date": "1966-02-14", "weekday": 0},
    {"year": 1967, "session_date": "1967-02-14", "weekday": 1},
    {"year": 1968, "session_date": "1968-02-14", "weekday": 2},
    {"year": 1969, "session_date": "1969-02-14", "weekday": 4},
    {"year": 1970, "session_date": "1970-02-16", "weekday": 0},
    {"year": 1971, "session_date": "1971-02-16", "weekday": 1},
    {"year": 1972, "session_date": "1972-02-14", "weekday": 0},
    {"year": 1973, "session_date": "1973-02-14", "weekday": 2},
    {"year": 1974, "session_date": "1974-02-14", "weekday": 3},
    {"year": 1975, "session_date": "1975-02-14", "weekday": 4},
    {"year": 1976, "session_date": "1976-02-17", "weekday": 1},
    {"year": 1977, "session_date": "1977-02-14", "weekday": 0},
    {"year": 1978, "session_date": "1978-02-14", "weekday": 1},
    {"year": 1979, "session_date": "1979-02-14", "weekday": 2},
    {"year": 1980, "session_date": "1980-02-14", "weekday": 3},
    {"year": 1981, "session_date": "1981-02-17", "weekday": 1},
    {"year": 1982, "session_date": "1982-02-16", "weekday": 1},
    {"year": 1983, "session_date": "1983-02-14", "weekday": 0},
    {"year": 1984, "session_date": "1984-02-14", "weekday": 1},
    {"year": 1985, "session_date": "1985-02-14", "weekday": 3},
    {"year": 1986, "session_date": "1986-02-14", "weekday": 4},
    {"year": 1987, "session_date": "1987-02-17", "weekday": 1},
    {"year": 1988, "session_date": "1988-02-16", "weekday": 1},
    {"year": 1989, "session_date": "1989-02-14", "weekday": 1},
    {"year": 1990, "session_date": "1990-02-14", "weekday": 2},
    {"year": 1991, "session_date": "1991-02-14", "weekday": 3},
    {"year": 1992, "session_date": "1992-02-14", "weekday": 4},
    {"year": 1993, "session_date": "1993-02-16", "weekday": 1},
    {"year": 1994, "session_date": "1994-02-14", "weekday": 0},
    {"year": 1995, "session_date": "1995-02-14", "weekday": 1},
    {"year": 1996, "session_date": "1996-02-14", "weekday": 2},
    {"year": 1997, "session_date": "1997-02-14", "weekday": 4},
    {"year": 1998, "session_date": "1998-02-17", "weekday": 1},
    {"year": 1999, "session_date": "1999-02-16", "weekday": 1},
    {"year": 2000, "session_date": "2000-02-14", "weekday": 0},
    {"year": 2001, "session_date": "2001-02-14", "weekday": 2},
    {"year": 2002, "session_date": "2002-02-14", "weekday": 3},
    {"year": 2003, "session_date": "2003-02-14", "weekday": 4},
    {"year": 2004, "session_date": "2004-02-17", "weekday": 1},
    {"year": 2005, "session_date": "2005-02-14", "weekday": 0},
    {"year": 2006, "session_date": "2006-02-14", "weekday": 1},
    {"year": 2007, "session_date": "2007-02-14", "weekday": 2},
    {"year": 2008, "session_date": "2008-02-14", "weekday": 3},
    {"year": 2009, "session_date": "2009-02-17", "weekday": 1},
    {"year": 2010, "session_date": "2010-02-16", "weekday": 1},
    {"year": 2011, "session_date": "2011-02-14", "weekday": 0},
    {"year": 2012, "session_date": "2012-02-14", "weekday": 1},
    {"year": 2013, "session_date": "2013-02-14", "weekday": 3},
    {"year": 2014, "session_date": "2014-02-14", "weekday": 4},
    {"year": 2015, "session_date": "2015-02-17", "weekday": 1},
    {"year": 2016, "session_date": "2016-02-16", "weekday": 1},
    {"year": 2017, "session_date": "2017-02-14", "weekday": 1},
    {"year": 2018, "session_date": "2018-02-14", "weekday": 2},
    {"year": 2019, "session_date": "2019-02-14", "weekday": 3},
    {"year": 2020, "session_date": "2020-02-14", "weekday": 4},
    {"year": 2021, "session_date": "2021-02-16", "weekday": 1},
    {"year": 2022, "session_date": "2022-02-14", "weekday": 0},
    {"year": 2023, "session_date": "2023-02-14", "weekday": 1},
    {"year": 2024, "session_date": "2024-02-14", "weekday": 2},
    {"year": 2025, "session_date": "2025-02-14", "weekday": 4},
]

VALENTINE_DF: pd.DataFrame = pd.DataFrame(VALENTINE_TRADING_DATES)


def valentine_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded Valentine's trading-date table.

    Columns: ``year`` (int), ``session_date`` (str 'YYYY-MM-DD'),
    ``weekday`` (int, 0=Mon..4=Fri). 76 rows (1950-2025).
    """
    return VALENTINE_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic daily-return generator -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    start_year: int = 1950,
    end_year: int = 2025,
    premium_bps: float = 0.0,
    base_bps: float = 3.6,
    vol_bps: float = 99.0,
    seed: int = 286,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily-return series with an optional planted Valentine premium.

    A synthetic Mon-Fri business-day calendar spanning ``start_year..end_year`` is
    generated. Each day's return is drawn i.i.d. from N(base_bps, vol_bps^2) in
    basis points. On the first business day on or after Feb 14 of each year, an
    extra drift of ``premium_bps`` bps is added. ``premium_bps = 0`` is the null:
    the Valentine session is statistically identical to every other day.

    Returns ``(df, truth)`` where ``df`` is indexed by date with columns ``ret``
    (daily simple return) and ``is_valentine`` (bool), and ``truth`` records the
    planted parameters. This is the study's null (and positive control) in a bottle.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(f"{start_year}-01-01", f"{end_year}-12-31")
    rets = rng.normal(base_bps * 1e-4, vol_bps * 1e-4, len(idx))

    df = pd.DataFrame({"ret": rets}, index=idx)
    df["year"] = df.index.year
    # First business day on or after Feb 14 each year.
    is_val = pd.Series(False, index=df.index)
    for y in range(start_year, end_year + 1):
        target = pd.Timestamp(y, 2, 14)
        sub = df.index[(df.index >= target) & (df.index <= pd.Timestamp(y, 2, 28))]
        if len(sub):
            is_val.loc[sub[0]] = True
    df["is_valentine"] = is_val.values
    df.loc[df["is_valentine"], "ret"] += premium_bps * 1e-4
    df = df.drop(columns=["year"])

    truth = {
        "start_year": start_year,
        "end_year": end_year,
        "premium_bps": premium_bps,
        "base_bps": base_bps,
        "vol_bps": vol_bps,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data -- ^GSPC daily closes (repo-level cache, cache-only by default)
# ---------------------------------------------------------------------------
def fetch_gspc_daily(
    fetch: bool = False,
    cache_dir: str = REPO_CACHE,
) -> pd.DataFrame:
    """Load ^GSPC daily close-to-close returns (price-only).

    Cache-only by default: reads ``_cache/^GSPC_split_only.parquet`` (staged in the
    repo). Network (yfinance) is touched only on explicit ``fetch=True``; the lazy
    import keeps the offline core network-free.

    Returns a DataFrame indexed by date with columns ``close`` and ``ret``
    (close-to-close simple return). Price-only -- no dividends. The index level is
    back-revised by the vendor, which we name on the Signal axis.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    cache_path = os.path.join(cache_dir, GSPC_CACHE_NAME)

    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"^GSPC cache not found at {cache_path}. "
                "Call fetch_gspc_daily(fetch=True) once, or stage the repo-level cache."
            )
        px = pd.read_parquet(cache_path)
    else:
        import yfinance as yf  # lazy import: network only on fetch=True

        raw = yf.download("^GSPC", start="1927-01-01", auto_adjust=False, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no ^GSPC data")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        px = raw[["Close"]].copy()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        px.to_parquet(cache_path)

    # Normalise to a 'close' column.
    if "Close" in px.columns:
        close = px["Close"]
    elif "close" in px.columns:
        close = px["close"]
    else:
        close = px.iloc[:, 0]
    out = pd.DataFrame({"close": close.astype(float)})
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out["ret"] = out["close"].pct_change()
    return out.dropna(subset=["ret"])


def build_event_frame(
    fetch: bool = False,
    cache_dir: str = REPO_CACHE,
    start_year: int = 1950,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Join the hardcoded Valentine session dates with the real ^GSPC daily returns.

    For each year the Valentine session return is the close-to-close return of the
    hardcoded ``session_date``. Returns a DataFrame indexed by ``year`` with
    columns: ``session_date`` (Timestamp), ``weekday`` (int), ``ret`` (the
    Valentine-session return), ``mkt_up`` (bool).

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    daily = fetch_gspc_daily(fetch=fetch, cache_dir=cache_dir)
    tbl = valentine_table()
    tbl = tbl[(tbl["year"] >= start_year) & (tbl["year"] <= end_year)].copy()
    tbl["session_date"] = pd.to_datetime(tbl["session_date"])

    rets = daily["ret"]
    tbl["ret"] = tbl["session_date"].map(lambda d: rets.get(d, np.nan))
    tbl = tbl.dropna(subset=["ret"]).copy()
    tbl["mkt_up"] = tbl["ret"] > 0
    return tbl.set_index("year")


def unconditional_daily(
    fetch: bool = False,
    cache_dir: str = REPO_CACHE,
    start_year: int = 1950,
    end_year: int = 2025,
) -> pd.Series:
    """The unconditional ^GSPC daily-return series over the study window (the baseline)."""
    daily = fetch_gspc_daily(fetch=fetch, cache_dir=cache_dir)
    yr = daily.index.year
    return daily.loc[(yr >= start_year) & (yr <= end_year), "ret"]


def fingerprint(rets) -> str:
    """A short content fingerprint of a return array/Series, for the as-of stamp."""
    vals = np.asarray(pd.Series(rets).dropna().to_numpy(dtype=float))
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]

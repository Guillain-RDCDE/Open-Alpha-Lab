"""Data layer for Study 280 (Solar-Eclipse).

Two components, both fully offline for the deterministic core:

- ``SOLAR_ECLIPSES`` — a hardcoded table of every **central** solar eclipse
  (total / annular / hybrid) from 1928 onward, the era covered by our cached
  ^GSPC daily series. Each row records the calendar date, the eclipse type, and
  a ``us_path`` flag = True when the central path (totality / annularity) crossed
  the contiguous United States — the natural "does it spook the *US* market" axis.
  This table is deterministic, small, well-known astronomy, and hardcoded here to
  keep the core fully offline and reproducible. Sources: NASA's Five Millennium
  Catalog of Solar Eclipses (Espenak & Meeus); Wikipedia "List of solar eclipses
  in the 20th/21st century."

- ``synthetic_panel`` — a deterministic, offline daily-return generator with an
  optional planted "eclipse-day premium/drag" knob. ``signal_bps = 0`` is the
  null (no effect), so tests confirm the machinery is truthful before we look at
  real data.

- ``fetch_gspc_daily`` — real S&P 500 (^GSPC) daily price returns from the
  repo-level cache (``_cache/^GSPC_split_only.parquet``). Cache-only by default
  (``fetch=False``); ``fetch=True`` triggers a lazy yfinance download. No network
  is touched in the default path.

No look-ahead: an event window is built around each eclipse *date*, and the
"eclipse-day return" is the close-to-close return realised on the first trading
day on or after the eclipse — the soonest a trader could have acted. The event
study is symmetric (``win`` trading days either side) and uses only realised
returns, never future information beyond the window the reader is shown.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

SEED = 280

# ---------------------------------------------------------------------------
# The hardcoded solar-eclipse table — the study's deterministic, offline core
# ---------------------------------------------------------------------------
# Columns:
#   date    : ISO calendar date of greatest eclipse (UTC day)
#   kind    : "total", "annular", or "hybrid" (central eclipses only — partials
#             are excluded because they have no path of totality/annularity)
#   us_path : True if the central path crossed the contiguous United States
#             (the obvious "spook the US market" axis), else False
#
# Source: NASA Five Millennium Canon of Solar Eclipses (Espenak & Meeus, 2006),
#   https://eclipse.gsfc.nasa.gov/ ; cross-checked against Wikipedia
#   "List of solar eclipses in the 20th century" and "...21st century."
# As-of: 2026-06-17. The list is restricted to central eclipses from 1928
# (start of our ^GSPC daily cache) through the 2026-08-12 total eclipse.

SOLAR_ECLIPSES: list[dict] = [
    {"date": "1930-04-28", "kind": "hybrid",  "us_path": True},
    {"date": "1930-10-21", "kind": "total",   "us_path": False},
    {"date": "1931-04-18", "kind": "annular", "us_path": False},
    {"date": "1932-08-31", "kind": "total",   "us_path": True},
    {"date": "1933-02-24", "kind": "annular", "us_path": False},
    {"date": "1934-02-14", "kind": "total",   "us_path": False},
    {"date": "1936-06-19", "kind": "total",   "us_path": False},
    {"date": "1937-06-08", "kind": "total",   "us_path": False},
    {"date": "1938-05-29", "kind": "total",   "us_path": False},
    {"date": "1940-04-07", "kind": "annular", "us_path": False},
    {"date": "1940-10-01", "kind": "total",   "us_path": False},
    {"date": "1941-09-21", "kind": "total",   "us_path": False},
    {"date": "1943-02-04", "kind": "total",   "us_path": False},
    {"date": "1945-07-09", "kind": "total",   "us_path": False},
    {"date": "1947-05-20", "kind": "total",   "us_path": False},
    {"date": "1947-11-12", "kind": "annular", "us_path": False},
    {"date": "1948-05-09", "kind": "annular", "us_path": False},
    {"date": "1948-11-01", "kind": "total",   "us_path": False},
    {"date": "1950-09-12", "kind": "total",   "us_path": False},
    {"date": "1952-02-25", "kind": "total",   "us_path": False},
    {"date": "1954-06-30", "kind": "total",   "us_path": False},
    {"date": "1955-06-20", "kind": "total",   "us_path": False},
    {"date": "1958-10-12", "kind": "total",   "us_path": False},
    {"date": "1959-10-02", "kind": "total",   "us_path": False},
    {"date": "1961-02-15", "kind": "total",   "us_path": False},
    {"date": "1962-07-31", "kind": "annular", "us_path": False},
    {"date": "1963-07-20", "kind": "total",   "us_path": True},
    {"date": "1965-05-30", "kind": "total",   "us_path": False},
    {"date": "1966-11-12", "kind": "total",   "us_path": False},
    {"date": "1970-03-07", "kind": "total",   "us_path": True},
    {"date": "1972-07-10", "kind": "total",   "us_path": False},
    {"date": "1973-06-30", "kind": "total",   "us_path": False},
    {"date": "1974-06-20", "kind": "total",   "us_path": False},
    {"date": "1976-10-23", "kind": "total",   "us_path": False},
    {"date": "1977-10-12", "kind": "total",   "us_path": False},
    {"date": "1979-02-26", "kind": "total",   "us_path": True},
    {"date": "1980-02-16", "kind": "total",   "us_path": False},
    {"date": "1981-07-31", "kind": "total",   "us_path": False},
    {"date": "1983-06-11", "kind": "total",   "us_path": False},
    {"date": "1984-05-30", "kind": "annular", "us_path": True},
    {"date": "1984-11-22", "kind": "total",   "us_path": False},
    {"date": "1985-11-12", "kind": "total",   "us_path": False},
    {"date": "1987-03-29", "kind": "hybrid",  "us_path": False},
    {"date": "1988-03-18", "kind": "total",   "us_path": False},
    {"date": "1990-07-22", "kind": "total",   "us_path": False},
    {"date": "1991-07-11", "kind": "total",   "us_path": False},
    {"date": "1992-06-30", "kind": "total",   "us_path": False},
    {"date": "1994-05-10", "kind": "annular", "us_path": True},
    {"date": "1994-11-03", "kind": "total",   "us_path": False},
    {"date": "1995-10-24", "kind": "total",   "us_path": False},
    {"date": "1997-03-09", "kind": "total",   "us_path": False},
    {"date": "1998-02-26", "kind": "total",   "us_path": False},
    {"date": "1999-08-11", "kind": "total",   "us_path": False},
    {"date": "2001-06-21", "kind": "total",   "us_path": False},
    {"date": "2002-06-10", "kind": "annular", "us_path": True},
    {"date": "2002-12-04", "kind": "total",   "us_path": False},
    {"date": "2003-05-31", "kind": "annular", "us_path": False},
    {"date": "2005-04-08", "kind": "hybrid",  "us_path": False},
    {"date": "2006-03-29", "kind": "total",   "us_path": False},
    {"date": "2008-08-01", "kind": "total",   "us_path": False},
    {"date": "2009-07-22", "kind": "total",   "us_path": False},
    {"date": "2010-07-11", "kind": "total",   "us_path": False},
    {"date": "2012-05-20", "kind": "annular", "us_path": True},
    {"date": "2012-11-13", "kind": "total",   "us_path": False},
    {"date": "2013-05-10", "kind": "annular", "us_path": False},
    {"date": "2013-11-03", "kind": "hybrid",  "us_path": False},
    {"date": "2015-03-20", "kind": "total",   "us_path": False},
    {"date": "2016-03-09", "kind": "total",   "us_path": False},
    {"date": "2017-02-26", "kind": "annular", "us_path": False},
    {"date": "2017-08-21", "kind": "total",   "us_path": True},
    {"date": "2019-07-02", "kind": "total",   "us_path": False},
    {"date": "2020-06-21", "kind": "annular", "us_path": False},
    {"date": "2020-12-14", "kind": "total",   "us_path": False},
    {"date": "2021-06-10", "kind": "annular", "us_path": False},
    {"date": "2021-12-04", "kind": "total",   "us_path": False},
    {"date": "2023-04-20", "kind": "hybrid",  "us_path": False},
    {"date": "2023-10-14", "kind": "annular", "us_path": True},
    {"date": "2024-04-08", "kind": "total",   "us_path": True},
    {"date": "2024-10-02", "kind": "annular", "us_path": False},
    {"date": "2026-02-17", "kind": "annular", "us_path": False},
    {"date": "2026-08-12", "kind": "total",   "us_path": False},
]

SOLAR_ECLIPSE_DF: pd.DataFrame = pd.DataFrame(SOLAR_ECLIPSES)
SOLAR_ECLIPSE_DF["date"] = pd.to_datetime(SOLAR_ECLIPSE_DF["date"])


def eclipse_table() -> pd.DataFrame:
    """Return a clean copy of the hardcoded solar-eclipse table.

    Columns: ``date`` (datetime64), ``kind`` ('total'|'annular'|'hybrid'),
    ``us_path`` (bool: central path crossed the contiguous US).
    """
    return SOLAR_ECLIPSE_DF.copy()


# ---------------------------------------------------------------------------
# Synthetic daily-return generator — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    start: str = "1928-01-01",
    end: str = "2026-09-30",
    signal_bps: float = 0.0,
    base_bps: float = 3.0,
    vol_bps: float = 100.0,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily-return series with an optional planted eclipse effect.

    Each *business* day is an i.i.d. draw N(base_bps, vol_bps^2) in basis points.
    On the first trading day on/after each eclipse in ``SOLAR_ECLIPSES``, an extra
    drift of ``signal_bps`` bps is added. ``signal_bps = 0`` is the null — eclipse
    days are statistically identical to every other day. This is the study's null
    in a bottle.

    Returns ``(df, truth)`` where ``df`` is indexed by date with a single ``ret``
    column (simple daily return), and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, end=end)
    rets = (base_bps + rng.normal(0.0, vol_bps, len(idx))) * 1e-4
    df = pd.DataFrame({"ret": rets}, index=idx)
    df.index.name = "Date"

    if signal_bps != 0.0:
        ecl = eclipse_table()
        for d in ecl["date"]:
            pos = df.index.searchsorted(d)  # first trading day on/after eclipse
            if 0 <= pos < len(df):
                df.iloc[pos, df.columns.get_loc("ret")] += signal_bps * 1e-4

    truth = {
        "start": start,
        "end": end,
        "signal_bps": signal_bps,
        "base_bps": base_bps,
        "vol_bps": vol_bps,
        "seed": seed,
        "n_days": len(df),
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — ^GSPC daily returns (repo-level cache, read-only by default)
# ---------------------------------------------------------------------------
def fetch_gspc_daily(
    cache_dir: str = REPO_CACHE,
    fetch: bool = False,
    start: str = "1928-01-01",
    end: str = "2026-09-30",
) -> pd.DataFrame:
    """Load ^GSPC daily close-to-close returns.

    Cache-only by default: reads ``_cache/^GSPC_split_only.parquet`` (staged in
    the repo-level cache, read-only). With ``fetch=True`` it lazily imports
    yfinance and downloads ^GSPC — the only path that touches the network.

    Returns a DataFrame indexed by ``Date`` (DatetimeIndex) with columns
    ``Close`` and ``ret`` (simple daily return).

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    path = os.path.join(cache_dir, "^GSPC_split_only.parquet")
    if os.path.exists(path):
        px = pd.read_parquet(path)
    elif fetch:
        import yfinance as yf  # lazy import — network only on explicit request

        raw = yf.download("^GSPC", start=start, end=end, auto_adjust=False,
                          progress=False)
        px = raw[["Close"]].copy()
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
    else:
        raise FileNotFoundError(
            f"^GSPC cache not found at {path}. Pass fetch=True to download "
            "from yfinance, or stage the repo-level _cache/ parquet."
        )

    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index)
    px = px[["Close"]].copy()
    px["ret"] = px["Close"].pct_change()
    px = px.loc[(px.index >= pd.Timestamp(start)) & (px.index <= pd.Timestamp(end))]
    return px.dropna()


def fingerprint(df: pd.DataFrame, col: str = "ret") -> str:
    """A short content fingerprint of a return column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]

"""Data layer for Study 191 (Leap-Year).

Two tapes, one shape (a calendar-year total-return series indexed by year):

- ``synthetic_annual`` — a *deterministic, offline* generator.  A ``leap_premium``
  knob adds a mean return differential on leap-year observations; ``leap_premium=0``
  is a pure random walk with no quadrennial seasonality.  This is the study's null
  in a bottle.
- ``load_shiller`` — annual calendar-year returns derived from the Shiller monthly
  S&P 500 dataset, staged at the shared repo cache
  ``_cache/shiller_sp500.parquet``.  No network call is ever made; we read the
  pre-staged parquet only when it exists (the CI runner will skip any test that
  guards on its presence).

No look-ahead: calendar year labels are assigned to the *actual* calendar year of
each observation.  December-to-December price change is the outcome — the year
label is known before January 1st.

Source: Shiller, R.J. (2005) *Irrational Exuberance*, 2nd ed. Updated data at
http://www.econ.yale.edu/~shiller/data.htm.  Cached at
``_cache/shiller_sp500.parquet`` in the repo root.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Per-study scratch cache (git-ignored).
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# Shared repo-level cache (read-only input — do NOT write here).
REPO_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_cache"))
SHILLER_PATH = os.path.join(REPO_CACHE, "shiller_sp500.parquet")


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------
def _is_leap(year: int) -> bool:
    """True iff *year* is a Gregorian leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def year_labels(years: pd.Index) -> pd.DataFrame:
    """Return a DataFrame of calendar-year boolean flags for *years*.

    Columns
    -------
    is_leap : leap year (e.g. 1928, 1932, …).
    is_election : US presidential-election year; in the modern era these coincide
        exactly with leap years (both are year % 4 == 0).
    term_year : 1 = election year, 2 = post-election, 3 = mid-term+1, 4 = mid-term.
        The four-year presidential cycle is the confound we must control for.
    """
    years_int = np.asarray(years, dtype=int)
    is_leap = np.array([_is_leap(int(y)) for y in years_int], dtype=bool)
    # Election years in US post-1789: always divisible by 4.
    is_election = (years_int % 4 == 0)
    term_year = (years_int % 4).copy()
    # Map: 0->1 (election), 1->2, 2->3, 3->4
    term_year = np.where(term_year == 0, 1, term_year + 1)
    return pd.DataFrame(
        {"is_leap": is_leap, "is_election": is_election, "term_year": term_year},
        index=years,
    )


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 100,
    annual_vol: float = 0.17,
    annual_drift: float = 0.08,
    leap_premium: float = 0.0,
    start_year: int = 1928,
    seed: int = 191,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual return series with a known leap-year effect (or none).

    Calendar-year log returns follow N(drift, vol²) with an additive bump of
    ``leap_premium`` applied *only* on leap years.

    - ``leap_premium = 0``  → pure random walk; leap years are just another year.
    - ``leap_premium > 0``  → positive return bias in leap years (the folk "good years").
    - ``leap_premium < 0``  → negative bias in leap years.

    Returns ``(df, truth)`` where ``df`` has columns
    ``year, ret, is_leap, is_election, term_year``.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, start_year + n_years)
    ret = rng.normal(annual_drift, annual_vol, n_years)

    labels = year_labels(pd.Index(years))
    # Apply the planted leap-year premium.
    ret[labels["is_leap"].to_numpy()] += leap_premium

    df = pd.DataFrame({"ret": ret}, index=years)
    df.index.name = "year"
    df = df.join(labels)

    truth = {
        "n_years": n_years,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "leap_premium": leap_premium,
        "start_year": int(start_year),
        "n_leap": int(labels["is_leap"].sum()),
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — Shiller monthly → annual calendar-year returns
# ---------------------------------------------------------------------------
def _build_annual(raw: pd.DataFrame, start_year: int = 1872) -> pd.DataFrame:
    """Convert the Shiller monthly parquet to a calendar-year return frame.

    We use the December monthly price (the last monthly observation for each year)
    and compute year-over-year price change: ret_Y = SP500_dec_Y / SP500_dec_{Y-1} - 1.
    This is a clean price return (no dividends re-invested) that matches the
    "headline" Shiller index, long enough to give ≥38 leap-year observations.

    No look-ahead: the December price is the *last* data point of the year.
    """
    raw = raw.copy()
    raw["Date"] = pd.to_datetime(raw["Date"])
    # Drop placeholder rows with SP500 = 0 or NaN.
    raw = raw[(raw["SP500"] > 0) & raw["SP500"].notna()]
    raw = raw.set_index("Date").sort_index()

    # Keep December rows only — one price per year.
    dec = raw[raw.index.month == 12]["SP500"].copy()
    dec.index = dec.index.year  # year as integer index
    dec.index.name = "year"

    # Year-over-year return.
    ret = dec.pct_change().dropna()
    ret.name = "ret"

    # Attach calendar flags.
    labels = year_labels(ret.index)
    df = pd.DataFrame({"ret": ret}).join(labels)
    return df[df.index >= start_year]


def load_shiller(
    start_year: int = 1872,
    path: str | None = None,
) -> pd.DataFrame:
    """Annual calendar-year S&P 500 return frame from the Shiller dataset.

    Reads the shared repo cache at ``_cache/shiller_sp500.parquet``; raises
    ``FileNotFoundError`` if it is absent (the CI runner sees this and skips the
    test via the ``requires_cache`` mark).

    Columns: ``ret, is_leap, is_election, term_year`` (index = year as int).
    """
    p = path or SHILLER_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Shiller cache not found at {p}.  "
            "This file is staged in the repo root _cache/ and must be present locally."
        )
    raw = pd.read_parquet(p)
    return _build_annual(raw, start_year=start_year)


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the return series, for the as-of stamp."""
    arr = df["ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

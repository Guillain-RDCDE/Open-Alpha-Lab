"""Data layer for Study 160 (Skyscraper-Curse).

Two tapes, one study:

- A **hardcoded event table** of world / US record-height skyscraper completions,
  sourced from Lawrence (1999), CTBUH, and Wikipedia. This is the canonical Skyscraper
  Index dataset — small, well-known, deterministic, and fully offline.
- A **synthetic generator** that plants a fake post-completion downturn so tests can
  verify the engine detects an effect when one exists, and reads near-zero when it does
  not. The ``crash_boost`` knob is the study's null in a bottle.
- A **real data loader** that fetches S&P 500 monthly returns from the Shiller dataset
  (already staged at _cache/shiller_sp500.parquet) so no network call is needed for
  the main analysis; yfinance daily ^GSPC is the supplementary real-tape.

The inference bar is set honestly from the start: with n ≈ 10 events (world records)
or n ≈ 7 (US-only records in the S&P era), a big t-stat is expected from sample
variance alone — we compute and report it, then state the obvious: a narrative, not a
signal.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_cache"))

# ---------------------------------------------------------------------------
# The hardcoded event table — world record-height completions (Lawrence 1999 +
# CTBUH Tall Buildings Database, accessed 2026-06).
#
# Each row: (year_completed, building, city, height_m, scope)
# scope = "world" for global record at the time, "us" if US-only record.
# "world" records are a strict subset: each must have been the TALLEST in the world.
# ---------------------------------------------------------------------------
SKYSCRAPER_EVENTS = pd.DataFrame([
    # --- Pre-S&P 500 era (included for completeness, excluded from S&P analysis) ---
    # Woolworth Building (NYC, 1913) — US and world record
    dict(year=1913, building="Woolworth Building",        city="New York",   height_m=241, scope="world"),
    # 40 Wall Street (NYC, 1930) — US and world record (briefly)
    dict(year=1930, building="40 Wall Street",            city="New York",   height_m=283, scope="world"),
    # Chrysler Building (NYC, 1930) — US and world record
    dict(year=1930, building="Chrysler Building",         city="New York",   height_m=319, scope="world"),
    # Empire State Building (NYC, 1931) — US and world record until 1972
    dict(year=1931, building="Empire State Building",     city="New York",   height_m=443, scope="world"),
    # --- S&P 500 era (1957 onward; S&P monthly data from Shiller starts 1871) ---
    # WTC North Tower (NYC, 1972) — world record (briefly surpassed Sears 1973)
    dict(year=1972, building="WTC North Tower",           city="New York",   height_m=417, scope="world"),
    # Sears Tower (Chicago, 1974) — world record until 1998
    dict(year=1974, building="Sears Tower",               city="Chicago",    height_m=442, scope="world"),
    # Petronas Twin Towers (KL, 1998) — world record until 2004
    dict(year=1998, building="Petronas Twin Towers",      city="Kuala Lumpur", height_m=452, scope="world"),
    # Taipei 101 (Taiwan, 2004) — world record until 2010
    dict(year=2004, building="Taipei 101",                city="Taipei",     height_m=508, scope="world"),
    # Burj Khalifa (Dubai, 2010) — world record (still standing as of 2026)
    dict(year=2010, building="Burj Khalifa",              city="Dubai",      height_m=828, scope="world"),
    # One World Trade Center (NYC, 2014) — US record; not world record
    dict(year=2014, building="One World Trade Center",    city="New York",   height_m=541, scope="us"),
    # Merdeka 118 (KL, 2023) — world record #2
    dict(year=2023, building="Merdeka 118",               city="Kuala Lumpur", height_m=679, scope="world"),
], columns=["year", "building", "city", "height_m", "scope"])

# Source citations embedded as a constant for traceability
_SOURCES = (
    "Lawrence, A. (1999). 'The Skyscraper Index: Faulty Towers!' Property Report, "
    "Dresdner Kleinwort Wasserstein Research. "
    "CTBUH (Council on Tall Buildings and Urban Habitat) database, ctbuh.org. "
    "Wikipedia: 'List of tallest buildings', accessed 2026-06."
)


def world_record_events(min_year: int = 1900, max_year: int = 2030) -> pd.DataFrame:
    """Return only global world-record completions in [min_year, max_year].

    Filters SKYSCRAPER_EVENTS to scope=='world' and the requested year range.
    """
    df = SKYSCRAPER_EVENTS[SKYSCRAPER_EVENTS["scope"] == "world"].copy()
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)].reset_index(drop=True)
    return df


def all_events(min_year: int = 1900, max_year: int = 2030) -> pd.DataFrame:
    """Return all events (world + US) in [min_year, max_year]."""
    df = SKYSCRAPER_EVENTS.copy()
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Shiller monthly S&P 500 — the main real-data source
# ---------------------------------------------------------------------------
def load_shiller(cache_dir: str = REPO_CACHE) -> pd.DataFrame:
    """Load the pre-staged Shiller S&P 500 monthly dataset.

    Columns of interest: SP500 (price index, nominal), Dividend, Earnings,
    'Consumer Price Index', PE10 (Shiller CAPE).

    The file has a 'Date' column (string 'YYYY-MM-DD') and a RangeIndex; we parse
    the Date column into a DatetimeIndex for time-series operations.

    The file lives at ``_cache/shiller_sp500.parquet`` (repo root _cache, not the
    study-local one). No network call needed — it is a read-only pre-staged input.
    """
    path = os.path.join(cache_dir, "shiller_sp500.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Shiller dataset not found at {path}. "
            "This is a pre-staged repo input — it should already be present."
        )
    df = pd.read_parquet(path)
    # Parse the Date column and use it as the index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df = df.set_index("Date")
    # Drop placeholder rows where SP500 is 0 or NaN
    df = df[df["SP500"].notna() & (df["SP500"] > 0)].copy()
    return df


# ---------------------------------------------------------------------------
# yfinance daily ^GSPC — supplementary real tape (cache-only by default)
# ---------------------------------------------------------------------------
def _gspc_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "gspc_daily_160.parquet")


def fetch_gspc(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Daily ^GSPC from Yahoo Finance; cache-only unless ``fetch=True``.

    Returns daily OHLCV from 1928 onward. Used for drawdown analysis around
    event years.
    """
    path = _gspc_cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached ^GSPC daily tape at {path}. "
                "Call fetch_gspc(fetch=True) once to populate the cache."
            )
        bars = pd.read_parquet(path)
    else:
        import yfinance as yf

        raw = yf.download(
            "^GSPC", start="1928-01-01", interval="1d",
            auto_adjust=True, progress=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no data for ^GSPC")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        bars.index.name = "date"
        if bars.index.tz is not None:
            bars.index = bars.index.tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        bars.to_parquet(path)

    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars.index.name = "date"
    return bars.sort_index()


# ---------------------------------------------------------------------------
# Synthetic generator — deterministic, offline, with a tunable crash knob
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 80,
    crash_boost: float = 0.0,
    crash_window: int = 3,
    annual_vol: float = 0.16,
    annual_drift: float = 0.07,
    event_years: list[int] | None = None,
    start_year: int = 1930,
    seed: int = 160,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual S&P return series with an optional post-event crash.

    Annual log-returns follow a drift-plus-noise process. In the ``crash_window`` years
    after each event year in ``event_years``, an extra annual drift of
    ``crash_boost / crash_window`` is added (negative = planted crash). With
    ``crash_boost=0`` the events are structurally present but carry zero extra signal —
    the study's null in a bottle.

    Returns ``(df, truth)`` where ``df`` has columns [year, log_ret, cum_log_ret] and
    ``truth`` records the planted parameters.
    """
    if event_years is None:
        event_years = [1931, 1972, 1974, 1998, 2004, 2010]

    rng = np.random.default_rng(seed)
    years = np.arange(start_year, start_year + n_years)
    daily_mu = annual_drift
    log_ret = rng.normal(daily_mu, annual_vol, n_years)

    crash_per_year = crash_boost / crash_window if crash_window > 0 else 0.0
    for ey in event_years:
        for k in range(crash_window):
            y = ey + k + 1
            mask = years == y
            log_ret[mask] += crash_per_year

    df = pd.DataFrame({"year": years, "log_ret": log_ret})
    df["cum_log_ret"] = df["log_ret"].cumsum()

    truth = {
        "crash_boost": crash_boost,
        "crash_window": crash_window,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "n_years": n_years,
        "event_years": event_years,
        "seed": seed,
    }
    return df, truth


def fingerprint(df: pd.DataFrame, col: str = "SP500") -> str:
    """Short content fingerprint for reproducibility stamps."""
    arr = df[col].to_numpy(dtype=float) if col in df.columns else df.iloc[:, 0].to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

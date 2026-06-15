"""Data layer for Study 159 (Presidential-Party).

Two tapes, both monthly / daily:

- ``synthetic_monthly`` — a *deterministic, offline* generator. Monthly log-returns with an
  optional per-party drift premium (``dem_premium_bps``). ``dem_premium_bps = 0`` is the null
  (no party effect) — the study's null in a bottle. A known random seed makes the series
  fully reproducible across machines and Python versions.
- ``load_shiller`` / ``load_yfinance`` — real market data from two complementary sources:
  Shiller's monthly S&P 500 dataset (1927-2026, real returns) and the Yahoo daily ^GSPC
  price index (1928-2026). Both are cache-only by default so the test-suite and the
  reproducible core never touch the network.

The presidential-party label is assigned by :func:`party_label`; it is a pure function of
the calendar date and the hardcoded ``PRESIDENTS`` table, which is known *before* any return
is observed — zero look-ahead risk.

The small-n problem is structural and explicit: from 1927 to 2026 there are only ~17
presidential terms, giving ~17 "Democrat" and ~8 "Republican" administration years.
No statistical test can paper over that — and the study says so loudly.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# The presidential-party table — hardcoded, fully offline.
#
# Source: U.S. National Archives, "Presidential Elections 1789-2020",
# Wikipedia "List of presidents of the United States".
# Dates are January 20 of inauguration year (or January 30 for FDR 1945,
# April 12 1945 for Truman, though for monthly precision Jan 20 is sufficient).
# We use the *party in the White House* convention: the month a president is
# inaugurated is assigned to the **new** party.
#
# Coverage starts 1927-01 to match the Shiller dataset's reliable real-return era.
# Each row: (name, party, term_start YYYY-MM-DD, term_end YYYY-MM-DD inclusive).
# "D" = Democrat, "R" = Republican.
# ---------------------------------------------------------------------------
PRESIDENTS: list[dict] = [
    # term_start is the first *full month* under that president's control
    # (we assign the inauguration month to the new president)
    dict(name="Coolidge",      party="R", start="1927-01-01", end="1929-02-28"),
    dict(name="Hoover",        party="R", start="1929-03-01", end="1933-02-28"),
    dict(name="Roosevelt FDR", party="D", start="1933-03-01", end="1945-03-31"),
    dict(name="Truman",        party="D", start="1945-04-01", end="1953-01-31"),
    dict(name="Eisenhower",    party="R", start="1953-02-01", end="1961-01-31"),
    dict(name="Kennedy",       party="D", start="1961-02-01", end="1963-11-30"),
    dict(name="Johnson LBJ",   party="D", start="1963-12-01", end="1969-01-31"),
    dict(name="Nixon",         party="R", start="1969-02-01", end="1974-07-31"),
    dict(name="Ford",          party="R", start="1974-08-01", end="1977-01-31"),
    dict(name="Carter",        party="D", start="1977-02-01", end="1981-01-31"),
    dict(name="Reagan",        party="R", start="1981-02-01", end="1989-01-31"),
    dict(name="Bush HW",       party="R", start="1989-02-01", end="1993-01-31"),
    dict(name="Clinton",       party="D", start="1993-02-01", end="2001-01-31"),
    dict(name="Bush W",        party="R", start="2001-02-01", end="2009-01-31"),
    dict(name="Obama",         party="D", start="2009-02-01", end="2017-01-31"),
    dict(name="Trump 1",       party="R", start="2017-02-01", end="2021-01-31"),
    dict(name="Biden",         party="D", start="2021-02-01", end="2025-01-31"),
    dict(name="Trump 2",       party="R", start="2025-02-01", end="2029-01-31"),
]

# Pre-build a lookup: for each president, the Timestamp range
_PRES_RECORDS = [
    {
        "name": p["name"],
        "party": p["party"],
        "start": pd.Timestamp(p["start"]),
        "end": pd.Timestamp(p["end"]),
    }
    for p in PRESIDENTS
]


# ---------------------------------------------------------------------------
# Calendar: assign party label to a DatetimeIndex
# ---------------------------------------------------------------------------
def party_label(index: pd.DatetimeIndex) -> pd.Series:
    """Assign each date a party label ('D', 'R', or NaN if outside coverage).

    The label is a pure function of the calendar date and the hardcoded
    ``PRESIDENTS`` table — no look-ahead risk whatsoever. Dates before 1927 or
    after the end of the latest defined term get ``NaN`` (dropped by the strategy).
    """
    out = pd.Series(index=index, dtype=object, name="party")
    for rec in _PRES_RECORDS:
        mask = (index >= rec["start"]) & (index <= rec["end"])
        out.loc[mask] = rec["party"]
    return out


# ---------------------------------------------------------------------------
# Synthetic tape — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_monthly(
    n_months: int = 1200,
    dem_premium_bps: float = 0.0,
    base_bps: float = 60.0,
    vol_ann: float = 0.16,
    seed: int = 159,
    start: str = "1927-01-01",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly return series with a planted Democrat premium.

    Monthly log-returns are i.i.d. normal with mean ``base_bps`` bps/month (drift)
    plus an additional ``dem_premium_bps`` bps/month under a Democrat president.

    To guarantee full coverage regardless of ``n_months``, the party sequence is
    constructed synthetically: terms of 48 months (4 years) alternate D/R, starting
    with R (Coolidge) — a stylised calendar that captures the block structure of real
    administrations without depending on the historical dates. This makes the premium
    recoverable for any ``n_months``, including large values used for power studies.

    The real tape uses the actual ``PRESIDENTS`` table via ``load_shiller()``; this
    synthetic generator is purely for testing the inference machinery under a known truth.

    ``dem_premium_bps = 0`` is the null (no party effect). Santa-Clara & Valkanov (2003)
    estimated a premium of roughly +1200 bps/yr (~100 bps/month) for Democrats.

    Returns ``(monthly_df, truth)`` where ``monthly_df`` has columns ``ret`` (monthly
    log-return) and ``party`` ('D' or 'R'), and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_months, freq="MS", name="date")

    vol_monthly = vol_ann / np.sqrt(12)
    base_monthly = base_bps * 1e-4  # bps -> decimal, already monthly

    # Extra drift per month under Democrat administrations
    dem_extra = dem_premium_bps * 1e-4  # bps -> decimal, monthly

    # Stylised alternating party sequence: 48-month terms, R then D then R...
    term_len = 48  # 4-year term in months
    party_seq = []
    start_party = "R"  # Coolidge first
    toggle = {"R": "D", "D": "R"}
    p = start_party
    while len(party_seq) < n_months:
        party_seq.extend([p] * term_len)
        p = toggle[p]
    parties = np.array(party_seq[:n_months])

    drift = np.where(parties == "D", base_monthly + dem_extra, base_monthly)
    noise = rng.normal(0.0, vol_monthly, n_months)
    ret = drift + noise

    df = pd.DataFrame({"ret": ret, "party": parties}, index=idx)
    truth = {
        "n_months": n_months,
        "dem_premium_bps": dem_premium_bps,
        "base_bps": base_bps,
        "vol_ann": vol_ann,
        "seed": seed,
        "term_len_months": term_len,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape 1 — Shiller monthly S&P 500 (real returns, long history)
# ---------------------------------------------------------------------------
_SHILLER_CACHE = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "_cache", "shiller_sp500.parquet")
)


def load_shiller(start_year: int = 1927) -> pd.DataFrame:
    """Load Shiller S&P 500 monthly data and compute real monthly returns.

    Uses the pre-staged cache at ``_cache/shiller_sp500.parquet``.  The Shiller
    dataset has monthly 'Real Price' (S&P 500 price divided by CPI), so returns
    here are *real* total-return proxies (price change only, no dividend reinvestment
    in the index column — see docs/results.md for the caveat).

    Adds ``party`` ('D' or 'R') and ``president`` columns from the hardcoded table,
    and drops rows outside the presidential-coverage window (NaN party).

    Returns a DataFrame with columns ``real_price``, ``ret``, ``party``, ``president``.
    """
    if not os.path.exists(_SHILLER_CACHE):
        raise FileNotFoundError(
            f"Shiller cache not found at {_SHILLER_CACHE}. "
            "Expected pre-staged file (read-only input — do not modify)."
        )
    raw = pd.read_parquet(_SHILLER_CACHE)

    # The Shiller dataset has a 'Date' column (string) and a RangeIndex.
    # Parse the date column and set it as index.
    if "Date" in raw.columns:
        raw = raw.copy()
        raw.index = pd.to_datetime(raw["Date"], errors="coerce")
        raw.index.name = "date"
        raw = raw[raw.index.notna()]

    # Real Price column (CPI-adjusted S&P level)
    price_col = "Real Price"
    if price_col not in raw.columns:
        # Fallback: use SP500 nominal
        price_col = "SP500"

    prices = raw[price_col].replace(0, np.nan).dropna()
    prices = prices[prices.index.year >= start_year].copy()
    # Normalise to month-start timestamps
    prices.index = prices.index.to_period("M").to_timestamp()

    ret = np.log(prices / prices.shift(1)).dropna()
    ret.name = "ret"

    # Align index to month-start timestamps for party_label
    ret_idx = pd.DatetimeIndex(ret.index)
    parties = party_label(ret_idx)

    # President name lookup
    pres_names = pd.Series(index=ret_idx, dtype=object, name="president")
    for rec in _PRES_RECORDS:
        mask = (ret_idx >= rec["start"]) & (ret_idx <= rec["end"])
        pres_names.loc[mask] = rec["name"]

    df = pd.DataFrame(
        {"real_price": prices.reindex(ret_idx), "ret": ret.values, "party": parties.values,
         "president": pres_names.values},
        index=ret_idx,
    )
    # Drop rows outside presidential coverage window
    df = df[df["party"].notna()].copy()
    return df


# ---------------------------------------------------------------------------
# Real tape 2 — Yahoo daily ^GSPC (price index, longer intraday resolution)
# ---------------------------------------------------------------------------
def _yf_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "pres_party_gspc_daily.parquet")


def fetch_gspc(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start_year: int = 1928,
) -> pd.DataFrame:
    """Daily ^GSPC close; cache-only unless ``fetch=True``.

    Returns a DataFrame with columns ``close``, ``ret``, ``party``, ``president``.
    Rows outside the presidential-coverage window are kept but have party = NaN;
    callers drop them as needed.
    """
    path = _yf_cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached ^GSPC tape at {path}. "
                "Call fetch_gspc(fetch=True) once to populate."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only on explicit fetch

        raw = yf.download("^GSPC", period="max", interval="1d", auto_adjust=True,
                          progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no data for ^GSPC")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"].dropna().squeeze()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        close = close[close.index < pd.Timestamp.now().normalize()]
        close.index.name = "date"
        ret = np.log(close / close.shift(1)).dropna()
        df = pd.DataFrame({"close": close, "ret": ret})
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    df = df[df.index.year >= start_year].copy()
    idx = pd.DatetimeIndex(df.index)
    df["party"] = party_label(idx).values
    pres_names = pd.Series(index=idx, dtype=object)
    for rec in _PRES_RECORDS:
        mask = (idx >= rec["start"]) & (idx <= rec["end"])
        pres_names.loc[mask] = rec["name"]
    df["president"] = pres_names.values
    return df


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """Short SHA-1 content fingerprint of the return column (as-of stamp)."""
    col = "ret" if "ret" in df.columns else df.columns[0]
    arr = df[col].to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

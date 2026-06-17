"""Data layer for Study 248 (Presidential-Honeymoon).

Two tapes, both daily:

- ``synthetic_daily`` — a *deterministic, offline* generator.  Returns follow i.i.d. normal
  noise plus an optional honeymoon-period drift ``honeymoon_premium_bp`` (extra bps per day
  in the first ``honeymoon_days`` calendar days after each inauguration).
  ``honeymoon_premium_bp = 0`` is the null (no honeymoon effect) — the test-suite's control arm.
- ``fetch_prices`` — real daily S&P 500 price-index (``^GSPC``, back to 1928), cache-only by
  default so the test-suite and the offline core never touch the network.

The 100-day honeymoon window is measured in *calendar* days from inauguration, assigned to a
``in_honeymoon`` boolean column.  Inauguration dates are hardcoded (no network needed).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

# ---------------------------------------------------------------------------
# Hardcoded inauguration table (US presidential inaugurations since 1929)
# Sources: US National Archives; Jan-20 is the constitutional date post-1937;
#          FDR 1933 was Mar-4 (last pre-20th-Amendment inauguration).
# Column: (president, inauguration_date)
# Each line = a NEW term start (re-elected incumbents count as a new term).
# ---------------------------------------------------------------------------
INAUGURATIONS: list[tuple[str, str]] = [
    ("Hoover",        "1929-03-04"),
    ("FDR",           "1933-03-04"),
    ("FDR (2nd)",     "1937-01-20"),
    ("FDR (3rd)",     "1941-01-20"),
    ("FDR (4th)",     "1945-01-20"),
    ("Truman",        "1949-01-20"),
    ("Eisenhower",    "1953-01-20"),
    ("Eisenhower (2nd)", "1957-01-20"),
    ("Kennedy",       "1961-01-20"),
    ("LBJ",           "1965-01-20"),
    ("Nixon",         "1969-01-20"),
    ("Nixon (2nd)",   "1973-01-20"),
    ("Carter",        "1977-01-20"),
    ("Reagan",        "1981-01-20"),
    ("Reagan (2nd)",  "1985-01-20"),
    ("Bush",          "1989-01-20"),
    ("Clinton",       "1993-01-20"),
    ("Clinton (2nd)", "1997-01-20"),
    ("Bush (2nd)",    "2001-01-20"),
    ("Bush (2nd/2)",  "2005-01-20"),
    ("Obama",         "2009-01-20"),
    ("Obama (2nd)",   "2013-01-20"),
    ("Trump",         "2017-01-20"),
    ("Biden",         "2021-01-20"),
    ("Trump (2nd)",   "2025-01-20"),
]

INAUGURATION_DATES: pd.DatetimeIndex = pd.DatetimeIndex(
    [pd.Timestamp(d) for _, d in INAUGURATIONS]
)

HONEYMOON_DAYS = 100  # calendar days (approx 4–5 trading months)


# ---------------------------------------------------------------------------
# Calendar: label honeymoon windows and term number
# ---------------------------------------------------------------------------
def honeymoon_labels(
    index: pd.DatetimeIndex,
    window_days: int = HONEYMOON_DAYS,
) -> pd.DataFrame:
    """For each date in ``index`` return whether it falls in *any* 100-day window.

    Returns a DataFrame with columns:
    - ``in_honeymoon``  (bool): True if the date is within ``window_days`` calendar
      days of an inauguration (inclusive of inauguration day, exclusive of day 100).
    - ``term_idx``      (int, nullable): index into INAUGURATIONS for the active
      term (0-based); NaN outside any term.

    The label is computed purely from the hardcoded inauguration list — no look-ahead.
    """
    in_hm = pd.Series(False, index=index, name="in_honeymoon")
    term_idx = pd.Series(np.nan, index=index, dtype="float64", name="term_idx")

    ts = pd.Series(index, dtype="datetime64[ns]")
    for i, inau in enumerate(INAUGURATION_DATES):
        end = inau + pd.Timedelta(days=window_days)
        mask = (ts.values >= inau.to_datetime64()) & (ts.values < end.to_datetime64())
        in_hm.iloc[mask] = True
        term_idx.iloc[mask] = float(i)

    return pd.DataFrame({"in_honeymoon": in_hm, "term_idx": term_idx}, index=index)


# ---------------------------------------------------------------------------
# Synthetic tape — deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_terms: int = 20,
    honeymoon_premium_bp: float = 0.0,
    base_bp: float = 3.5,
    vol_ann: float = 0.16,
    seed: int = 248,
    window_days: int = HONEYMOON_DAYS,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily return series with a planted honeymoon-period premium.

    ``n_terms`` presidential terms are simulated starting from a fixed anchor date.
    Each trading day gets ``base_bp`` bps of drift.  Days within ``window_days``
    calendar days of an inauguration get an additional ``honeymoon_premium_bp`` bps.

    ``honeymoon_premium_bp = 0`` is the null — no honeymoon effect exists.

    Returns ``(daily_df, truth)`` where ``daily_df`` has columns ``ret`` (simple daily
    return) and ``in_honeymoon`` (bool), and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    # Simulate n_terms * 4 years of daily returns (4-year average term length)
    n_years = n_terms * 4
    idx = pd.bdate_range("1929-01-01", periods=n_years * 252, name="date")

    labels = honeymoon_labels(pd.DatetimeIndex(idx), window_days=window_days)
    in_hm = labels["in_honeymoon"].values

    sig_d = vol_ann / np.sqrt(252)
    drift = np.full(len(idx), base_bp * 1e-4, dtype=float)
    drift[in_hm] += honeymoon_premium_bp * 1e-4

    ret = drift + sig_d * rng.standard_normal(len(idx))
    df = pd.DataFrame(
        {"ret": ret, "in_honeymoon": in_hm},
        index=idx,
    )
    truth = {
        "n_terms": n_terms,
        "honeymoon_premium_bp": honeymoon_premium_bp,
        "base_bp": base_bp,
        "vol_ann": vol_ann,
        "seed": seed,
        "window_days": window_days,
        "n_days": len(idx),
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("^", "").replace("=", "").replace("/", "").lower()
    return os.path.join(cache_dir, f"presidential_honeymoon_{safe}_daily.parquet")


def fetch_prices(
    ticker: str = "^GSPC",
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start_year: int = 1928,
    window_days: int = HONEYMOON_DAYS,
) -> pd.DataFrame:
    """Daily close and simple-return for ``ticker``, cache-first.

    ``^GSPC`` is the S&P 500 **price index** (no dividends, but available back to 1928 on
    Yahoo — all 25 inaugurations since Hoover). Returns a DataFrame with columns
    ``close``, ``ret``, plus ``in_honeymoon`` (bool) and ``term_idx`` (int, nullable).

    Network is touched only with ``fetch=True``; otherwise raises ``FileNotFoundError`` if
    the cache does not exist.
    """
    path = _cache_path(ticker, cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached daily tape for {ticker} at {path}. "
                f"Call fetch_prices({ticker!r}, fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only on explicit fetch

        raw = yf.download(ticker, period="max", interval="1d", auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError(f"yfinance returned no data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        close = raw["Close"].dropna().squeeze()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        close = close[close.index < pd.Timestamp.now().normalize()]
        close.index.name = "date"
        ret = close.pct_change().dropna()
        df = pd.DataFrame({"close": close, "ret": ret})
        df.index.name = "date"
        os.makedirs(cache_dir, exist_ok=True)
        df.to_parquet(path)

    df = df[df.index.year >= start_year].copy()
    labels = honeymoon_labels(pd.DatetimeIndex(df.index), window_days=window_days)
    df["in_honeymoon"] = labels["in_honeymoon"].values
    df["term_idx"] = labels["term_idx"].values
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the close or return column, for the as-of stamp."""
    col = "close" if "close" in df.columns else "ret"
    h = hashlib.sha1(np.ascontiguousarray(df[col].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]

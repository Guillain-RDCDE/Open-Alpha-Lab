"""Data layer for Study 163 (Friday-13th).

Two tapes, one shape (a daily return series for ^GSPC):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``f13_effect``
  knob lets us dial a mean return differential on Friday-the-13th vs all other
  Fridays. ``f13_effect = 0`` is the null — Friday the 13th is just another
  trading day — so tests can assert the strategy beats a placebo *only* when a
  genuine anomaly is planted.  This is the study's null in a bottle.
- ``fetch_gspc`` — real ^GSPC daily prices (``yfinance``), **cache-only** by
  default so the test-suite and the reproducible core never touch the network.
  We use the shared repo cache at
  ``_cache/^GSPC_split_only.parquet`` (available from 1927-12-30).

No look-ahead: all signals are formed from the *calendar date label alone* —
is today the 13th and a Friday? That fact is known before the open. The daily
close-to-close return is the outcome.

**Friday-the-13th dates are derived by pure date arithmetic** (no fetch, no
table): a date is a Friday-13th iff ``date.weekday() == 4`` *and*
``date.day == 13``. This is deterministic and verifiable. ~1.7 events/year in
the modern era, so the effective n over the full GSPC history is ~170.

Source: Kolb & Rodriguez (1987), *Friday the Thirteenth: 'Part VII' — A Note*,
Journal of Finance 42(5), 1385–1387.  Later work: Lucey (2001), *Friday the
13th and the Philosophical Basis of Financial Economics*, Applied Economics
Letters 8(9), 565–568.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Per-study scratch cache (for the study's own fetches if any).
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# Shared repo-level cache (read-only input — do NOT write here).
REPO_CACHE = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "_cache")
)

TICKER = "^GSPC"
# Path in the shared repo cache where the GSPC daily bars live.
_GSPC_SHARED = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")
# Fallback: per-study cache (populated by verify.py --fetch).
_GSPC_LOCAL = os.path.join(DEFAULT_CACHE, "gspc_daily.parquet")


# ---------------------------------------------------------------------------
# Calendar arithmetic — the only "event table" we need
# ---------------------------------------------------------------------------
def is_friday_13th(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True on every Friday-the-13th in *dates*.

    Pure date arithmetic — no network call, no hard-coded table needed because
    "Friday the 13th" is a deterministic calendar property:
        weekday() == 4  (Friday, Monday=0)
        day == 13
    """
    arr = (dates.weekday == 4) & (dates.day == 13)
    return pd.Series(arr, index=dates, name="is_f13")


def is_friday_6th(dates: pd.DatetimeIndex) -> pd.Series:
    """Placebo mask: True on every Friday-the-6th.

    Used as the superstition-free matched control: same weekday, same proximity
    to the prior Monday, no folk legend attached.
    """
    arr = (dates.weekday == 4) & (dates.day == 6)
    return pd.Series(arr, index=dates, name="is_f6")


def is_friday(dates: pd.DatetimeIndex) -> pd.Series:
    """All Fridays (including the 13th) — the broad baseline."""
    return pd.Series(dates.weekday == 4, index=dates, name="is_friday")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 75,
    annual_vol: float = 0.15,
    annual_drift: float = 0.07,
    f13_effect: float = 0.0,
    seed: int = 163,
    start: str = "1950-01-04",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily GSPC-like tape with a known Friday-13th anomaly.

    Daily log-returns follow N(drift/252, (vol/sqrt(252))**2) with an additive
    bump of ``f13_effect / sqrt(252)`` (expressed in annual vol units) applied
    *only* on Friday-the-13th trading days.

    - ``f13_effect = 0``   → pure random walk; the signal is a fair coin.
    - ``f13_effect < 0``   → negative return bias on Friday-13th (the folk fear).
    - ``f13_effect > 0``   → positive bias on Friday-13th (contrarian "good luck").

    Returns ``(df, truth)`` where ``df`` has columns ``date, ret, is_f13,
    is_f6, is_friday``.
    """
    rng = np.random.default_rng(seed)
    # Build a business-day calendar starting from start.
    idx = pd.bdate_range(start=start, periods=n_years * 252)
    n = len(idx)
    daily_mu = annual_drift / 252.0
    daily_sig = annual_vol / np.sqrt(252.0)
    ret = rng.normal(daily_mu, daily_sig, n)

    f13 = (idx.weekday == 4) & (idx.day == 13)
    # Apply the planted effect on Friday-the-13th days.
    ret[f13] += f13_effect * daily_sig

    df = pd.DataFrame(
        {
            "ret": ret,
            "is_f13": f13,
            "is_f6": (idx.weekday == 4) & (idx.day == 6),
            "is_friday": idx.weekday == 4,
        },
        index=idx,
    )
    df.index.name = "date"

    truth = {
        "n_years": n_years,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "f13_effect": f13_effect,
        "n_days": n,
        "n_f13": int(f13.sum()),
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC daily, cache-only by default
# ---------------------------------------------------------------------------
def _build_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw OHLCV frame to a daily return frame with calendar labels."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw.rename(columns=str.lower)["close"].sort_index()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index.name = "date"
    ret = np.log(close / close.shift(1))
    idx = close.index
    df = pd.DataFrame(
        {
            "close": close,
            "ret": ret,
            "is_f13": (idx.weekday == 4) & (idx.day == 13),
            "is_f6": (idx.weekday == 4) & (idx.day == 6),
            "is_friday": idx.weekday == 4,
        }
    )
    return df.dropna(subset=["ret"])


def fetch_gspc(
    start: str = "1928-01-01",
    fetch: bool = False,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Daily GSPC return frame with calendar labels; cache-only unless ``fetch=True``.

    Tries the shared repo cache first (``_cache/^GSPC_split_only.parquet``);
    falls back to the per-study cache; finally hits the network only when
    ``fetch=True``.

    Columns: ``close, ret, is_f13, is_f6, is_friday`` (``ret`` is log-return).
    """
    # 1. Shared repo cache (best — already covers 1927-onward).
    if os.path.exists(_GSPC_SHARED):
        raw = pd.read_parquet(_GSPC_SHARED)
        df = _build_daily(raw)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    # 2. Per-study local cache.
    local = _GSPC_LOCAL if cache_dir is None else os.path.join(cache_dir, "gspc_daily.parquet")
    if os.path.exists(local):
        raw = pd.read_parquet(local)
        df = _build_daily(raw)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    # 3. Network — only with explicit fetch=True.
    if not fetch:
        raise FileNotFoundError(
            f"No GSPC cache found at {_GSPC_SHARED} or {local}. "
            "Call fetch_gspc(fetch=True) to populate the local cache."
        )

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download(
        TICKER,
        start=start,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {TICKER}")

    cache_dir_use = cache_dir or DEFAULT_CACHE
    os.makedirs(cache_dir_use, exist_ok=True)
    raw.to_parquet(os.path.join(cache_dir_use, "gspc_daily.parquet"))

    df = _build_daily(raw)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (ret column), for the as-of stamp."""
    arr = df["ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

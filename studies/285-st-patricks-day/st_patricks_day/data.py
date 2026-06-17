"""Data layer for Study 285 (St-Patricks-Day).

The folk claim: the stock market gets a "St. Patrick's Day bump" — equities are
supposed to do unusually well on (or around) March 17 each year, sometimes
dressed up as the "wearin' o' the green" rally. We test it as a clean **event
study** on ^GSPC daily returns.

Two tapes, one shape (a daily return series for ^GSPC):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``stpat_effect``
  knob plants a mean-return differential on the St. Patrick's Day trading session
  vs every other day. ``stpat_effect = 0`` is the null — March 17 is just another
  trading day — so the test-suite can assert the engine reads ~zero on the null
  and lights up only when a genuine bump is planted. This is the study's null in
  a bottle.
- ``fetch_gspc`` — real ^GSPC daily prices (``yfinance``), **cache-only** by
  default so the test-suite and the reproducible core never touch the network.
  We use the shared repo cache at ``_cache/^GSPC_split_only.parquet`` (1927-onward).

**St. Patrick's Day trading dates are derived by pure date arithmetic, not a
hard-coded table.** St. Patrick's Day is fixed on the civil calendar (March 17),
but March 17 frequently falls on a weekend or a market holiday. We therefore
define the *St. Patrick's Day trading session* for a given year as the **first
trading day on or after March 17** (so a Saturday-17 maps to the following Monday).
The dates ARE listed in ``stpat_trading_dates`` for transparency / spot-checks,
but they are computed from the price index, not transcribed.

No look-ahead: the event is the *calendar-date label* (today is the St. Patrick's
session) which is known before the open. The outcome is that day's close-to-close
log-return.

Sources:
  - Folklore: "wearin' o' the green" / holiday-rally lore; no peer-reviewed
    "St. Patrick's Day effect" exists — this is calendar superstition, tested as
    one more day-of-year candidate against the multiple-comparisons bar.
  - Lakonishok & Smidt (1988), "Are Seasonal Anomalies Real? A Ninety-Year
    Perspective," Review of Financial Studies 1(4), 403–425 — the canonical
    reckoning for calendar-day anomalies and the data-snooping problem.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Per-study scratch cache (for the study's own fetches, if any).
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
# Shared repo-level cache (read-only input — do NOT write here).
REPO_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "_cache"))

TICKER = "^GSPC"
# Path in the shared repo cache where the GSPC daily bars live (1927-onward).
_GSPC_SHARED = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")
# Fallback: per-study cache (populated by verify.py / fetch=True).
_GSPC_LOCAL = os.path.join(DEFAULT_CACHE, "gspc_daily.parquet")

# St. Patrick's Day is March 17 on the civil calendar.
STPAT_MONTH = 3
STPAT_DAY = 17
# Placebo day-of-month: one week later (March 24) — same month, same "late-March"
# proximity, no folklore attached.
PLACEBO_DAY = 24


# ---------------------------------------------------------------------------
# Calendar arithmetic — building the event mask from the price index itself
# ---------------------------------------------------------------------------
def stpat_trading_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First trading day on or after March 17 for each year present in ``index``.

    Pure index arithmetic — no hard-coded table. If March 17 is a trading day we
    use it; otherwise we roll forward to the next trading day in the same March
    (a Saturday-17 maps to Monday-19, etc.).
    """
    idx = pd.DatetimeIndex(index)
    out = []
    for y in sorted(set(idx.year)):
        cand = idx[(idx.year == y) & (idx.month == STPAT_MONTH) & (idx.day >= STPAT_DAY)]
        if len(cand):
            out.append(cand.min())
    return pd.DatetimeIndex(out)


def first_trading_on_or_after(index: pd.DatetimeIndex, day: int) -> pd.DatetimeIndex:
    """First trading day on or after ``day`` of March, per year. Generic helper.

    Used for both the St. Patrick's session (``day=17``), the placebo
    (``day=24``), and the day-of-March multiple-comparisons sweep.
    """
    idx = pd.DatetimeIndex(index)
    out = []
    for y in sorted(set(idx.year)):
        cand = idx[(idx.year == y) & (idx.month == STPAT_MONTH) & (idx.day >= day)]
        if len(cand):
            out.append(cand.min())
    return pd.DatetimeIndex(out)


def _label(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Attach the event/control boolean masks to a daily index."""
    stpat = stpat_trading_dates(idx)
    plac = first_trading_on_or_after(idx, PLACEBO_DAY)
    is_stpat = idx.isin(stpat)
    is_placebo = idx.isin(plac)
    return pd.DataFrame(
        {
            "is_stpat": is_stpat,
            "is_placebo": is_placebo,
            "is_march": idx.month == STPAT_MONTH,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 99,
    annual_vol: float = 0.15,
    annual_drift: float = 0.06,
    stpat_effect: float = 0.0,
    seed: int = 285,
    start: str = "1928-01-03",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily GSPC-like tape with an optional planted St. Patrick bump.

    Daily returns follow N(drift/252, (vol/sqrt(252))**2) with an additive bump of
    ``stpat_effect * (vol/sqrt(252))`` applied *only* on the St. Patrick's Day
    trading session (first trading day on or after March 17 each year).

    - ``stpat_effect = 0``   → pure random walk; the bump is a fair coin.
    - ``stpat_effect > 0``   → positive bias on March 17 (the folk "green rally").
    - ``stpat_effect < 0``   → negative bias.

    Returns ``(df, truth)`` where ``df`` has columns ``ret, is_stpat, is_placebo,
    is_march`` indexed by date.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_years * 252)
    n = len(idx)
    daily_mu = annual_drift / 252.0
    daily_sig = annual_vol / np.sqrt(252.0)
    ret = rng.normal(daily_mu, daily_sig, n)

    labels = _label(idx)
    is_stpat = labels["is_stpat"].to_numpy()
    ret[is_stpat] += stpat_effect * daily_sig

    df = pd.DataFrame({"ret": ret}, index=idx)
    df["is_stpat"] = labels["is_stpat"].to_numpy()
    df["is_placebo"] = labels["is_placebo"].to_numpy()
    df["is_march"] = labels["is_march"].to_numpy()
    df.index.name = "date"

    truth = {
        "n_years": n_years,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "stpat_effect": stpat_effect,
        "n_days": n,
        "n_stpat": int(is_stpat.sum()),
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC daily, cache-only by default
# ---------------------------------------------------------------------------
def _build_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw OHLCV frame to a labelled daily log-return frame."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw.rename(columns=str.lower)["close"].sort_index()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close.index.name = "date"
    ret = np.log(close / close.shift(1))
    df = pd.DataFrame({"close": close, "ret": ret})
    df = df.dropna(subset=["ret"])
    labels = _label(pd.DatetimeIndex(df.index))
    df["is_stpat"] = labels["is_stpat"].to_numpy()
    df["is_placebo"] = labels["is_placebo"].to_numpy()
    df["is_march"] = labels["is_march"].to_numpy()
    return df


def fetch_gspc(
    start: str = "1928-01-01",
    fetch: bool = False,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Daily GSPC return frame with St. Patrick labels; cache-only unless ``fetch=True``.

    Tries the shared repo cache first (``_cache/^GSPC_split_only.parquet``); falls
    back to the per-study cache; finally hits the network only when ``fetch=True``
    (lazy ``yfinance`` import).

    Columns: ``close, ret, is_stpat, is_placebo, is_march`` (``ret`` is log-return).
    """
    if os.path.exists(_GSPC_SHARED):
        raw = pd.read_parquet(_GSPC_SHARED)
        df = _build_daily(raw)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    local = _GSPC_LOCAL if cache_dir is None else os.path.join(cache_dir, "gspc_daily.parquet")
    if os.path.exists(local):
        raw = pd.read_parquet(local)
        df = _build_daily(raw)
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    if not fetch:
        raise FileNotFoundError(
            f"No GSPC cache found at {_GSPC_SHARED} or {local}. "
            "Call fetch_gspc(fetch=True) to populate the local cache."
        )

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download(
        TICKER, start=start, interval="1d", auto_adjust=True, progress=False,
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
    """Short content fingerprint of a tape (ret column), for the as-of stamp."""
    arr = df["ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

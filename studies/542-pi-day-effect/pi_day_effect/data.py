"""Data layer for Study 542 (Pi-Day-Effect).

The claim is pure numerology: certain calendar dates encode a famous mathematical
constant, so does the market "know" and behave differently on those days?

- **Pi Day** — March 14 (3/14, for pi = 3.14159...). The canonical constant-date.
- **Euler's day** — February 7 (2/7, e = 2.71828...) or, in the other reading,
  the day-of-year that matches e. We use the digit reading 2/7.
- **Tau Day** — June 28 (6/28, tau = 2*pi = 6.28318...).
- **Golden-ratio day** — January 6 (1/6, phi ~= 1.618, digit reading 1/6).
- **Square-root-of-two day** — January 4 (1/4, sqrt(2) ~= 1.414, digit reading 1/4).
- **Feigenbaum day** — April 6 (4/6, delta ~= 4.669).

These are the "constant days". Everything else is an ordinary trading day. The test:
is the mean daily return on constant days different from ordinary days? The honest
placebo is a *random* set of the same number of calendar-date slots — if random date
sets look just as extreme, the constant-day result is noise.

Two tapes, one shape (a daily close-to-close return series):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``constant_effect``
  knob dials a mean-return bump applied only on constant days. ``= 0`` is the null
  (constant days are just trading days). The study's positive control and null in one
  bottle; tests never touch the network.
- ``fetch_series`` — real daily prices (SPY headline, ^GSPC long tape), **cache-first**
  from the shared repo cache so the reproducible core never needs the network. Network
  is touched only on an explicit ``fetch=True``.

No look-ahead: every signal is formed from the *calendar-date label alone* (is today
March 14?), which is known before the open. The daily close-to-close return is the
outcome. Constant-date membership is pure date arithmetic — no fetch, no table.

Survivorship / power caveat (stated on the SIGNAL axis): each constant date occurs at
most once per year, so even the ~98-year GSPC tape yields only ~1 event/constant/year.
SPY (1993-) gives ~33 Pi Days. Low event counts cap the statistical power; a numerology
anomaly this thin can never earn a robust real-tape t >= 2 — the study is capped at
WEAK/NONE by construction.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
# The shared repo cache lives at <repo>/_cache/_cache/ (read-only input).
SHARED_CACHE = os.path.join(REPO_ROOT, "_cache", "_cache")

# The constant dates: (name, month, day, constant value shown for flavour).
CONSTANT_DATES = [
    ("pi", 3, 14, 3.14159),          # Pi Day
    ("euler_e", 2, 7, 2.71828),      # e = 2.718..., digit reading 2/7
    ("tau", 6, 28, 6.28318),         # Tau Day (2*pi)
    ("golden_phi", 1, 6, 1.61803),   # phi ~ 1.618, digit reading 1/6
    ("sqrt2", 1, 4, 1.41421),        # sqrt(2) ~ 1.414, digit reading 1/4
    ("feigenbaum", 4, 6, 4.66920),   # Feigenbaum delta ~ 4.669, reading 4/6
]
CONSTANT_NAMES = [c[0] for c in CONSTANT_DATES]
_CONSTANT_MD = {(m, d) for (_, m, d, _v) in CONSTANT_DATES}
_PI_MD = (3, 14)


# ---------------------------------------------------------------------------
# Calendar arithmetic — the only "event table" we need
# ---------------------------------------------------------------------------
def is_pi_day(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True on every March 14 (Pi Day) in *dates* — pure date arithmetic."""
    arr = (dates.month == _PI_MD[0]) & (dates.day == _PI_MD[1])
    return pd.Series(arr, index=dates, name="is_pi")


def is_constant_day(dates: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True on any of the constant dates (Pi/e/tau/phi/sqrt2/Feigenbaum)."""
    md = list(zip(dates.month, dates.day))
    arr = np.array([(m, d) in _CONSTANT_MD for (m, d) in md])
    return pd.Series(arr, index=dates, name="is_constant")


def constant_mask(dates: pd.DatetimeIndex, name: str) -> pd.Series:
    """Boolean mask for a single named constant date."""
    m, d = next((mm, dd) for (nm, mm, dd, _v) in CONSTANT_DATES if nm == name)
    arr = (dates.month == m) & (dates.day == d)
    return pd.Series(arr, index=dates, name=f"is_{name}")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 60,
    annual_vol: float = 0.16,
    annual_drift: float = 0.07,
    constant_effect: float = 0.0,
    seed: int = 542,
    start: str = "1965-01-04",
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily market-like tape with a known Pi/constant-day anomaly.

    Daily returns ~ N(drift/252, (vol/sqrt(252))**2), with an additive bump of
    ``constant_effect * daily_sig`` applied *only* on the constant dates.

    - ``constant_effect = 0``  -> pure random walk; constant days are a fair coin.
    - ``constant_effect > 0``  -> constant days earn a positive bias (the "lucky number").
    - ``constant_effect < 0``  -> negative bias.

    Returns ``(df, truth)`` where ``df`` has columns ``ret, is_pi, is_constant``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_years * 252)
    n = len(idx)
    daily_mu = annual_drift / 252.0
    daily_sig = annual_vol / np.sqrt(252.0)
    ret = rng.normal(daily_mu, daily_sig, n)

    is_const = is_constant_day(idx).to_numpy()
    ret[is_const] += constant_effect * daily_sig

    df = pd.DataFrame(
        {
            "ret": ret,
            "is_pi": is_pi_day(idx).to_numpy(),
            "is_constant": is_const,
        },
        index=idx,
    )
    df.index.name = "date"
    truth = {
        "n_years": n_years,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "constant_effect": constant_effect,
        "n_days": int(n),
        "n_pi": int(df["is_pi"].sum()),
        "n_constant": int(df["is_constant"].sum()),
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — daily prices, cache-first
# ---------------------------------------------------------------------------
_TICKER_FILES = {
    "SPY": ["SPY_total_return.parquet", "SPY_split_only.parquet"],
    "^GSPC": ["^GSPC_split_only.parquet"],
}


def _build_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw OHLCV frame to a daily return frame with constant-date labels."""
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
            "is_pi": is_pi_day(idx).to_numpy(),
            "is_constant": is_constant_day(idx).to_numpy(),
        }
    )
    return df.dropna(subset=["ret"])


def fetch_series(
    ticker: str = "SPY",
    start: str = "1993-01-01",
    end: str = "2026-06-30",
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Daily return frame with constant-date labels; cache-first.

    Resolution order:
      1. This study's own ``_cache/`` copy (written on the one explicit fetch).
      2. The shared repo cache (``_cache/_cache/<TICKER>_*.parquet``, read-only).
      3. Network — only with explicit ``fetch=True`` (then cached into this study).

    Columns: ``close, ret, is_pi, is_constant`` (``ret`` is daily log-return).
    Returns an empty frame when ``fetch=False`` and no cache is found.
    """
    local = os.path.join(cache_dir, f"{ticker.replace('^', '')}_daily.parquet")

    # 1. per-study cache
    if os.path.exists(local):
        df = _build_daily(pd.read_parquet(local))
        return _clip(df, start, end)

    # 2. shared repo cache
    for fname in _TICKER_FILES.get(ticker, []):
        shared = os.path.join(SHARED_CACHE, fname)
        if os.path.exists(shared):
            df = _build_daily(pd.read_parquet(shared))
            # persist a study-local copy so the core is self-contained
            os.makedirs(cache_dir, exist_ok=True)
            pd.read_parquet(shared).to_parquet(local)
            return _clip(df, start, end)

    # 3. network
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()
    os.makedirs(cache_dir, exist_ok=True)
    raw.to_parquet(local)
    return _clip(_build_daily(raw), start, end)


def _clip(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (ret column), for the as-of stamp."""
    arr = df["ret"].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

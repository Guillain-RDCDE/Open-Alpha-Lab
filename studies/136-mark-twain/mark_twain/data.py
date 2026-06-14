"""Data layer for Study 136 (Mark-Twain).

Two tapes, one shape (a daily close-to-close return Series indexed by date):

- ``synthetic_daily`` — a *deterministic, offline* generator. A ``october_penalty``
  knob adds a downward drift to October returns; ``october_penalty=0`` is a pure
  random walk with no monthly seasonality. This is the study's null in a bottle.
- ``fetch_daily`` — the real S&P 500 (^GSPC) daily tape back to 1928, via
  Yahoo Finance, **cache-first by default**. The result is stored as parquet under
  ``_cache/`` so the test suite and the reproducible core never touch the network.

No look-ahead is baked in here — calendar labels are assigned to the *actual* calendar
month of each observation, not a forward-looking window.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")
STUDY_CACHE = os.path.join(_HERE, "..", "_cache")

_CACHE_FILE = "mark_twain_gspc_daily.parquet"

# Calendar months that famously carry the "October-effect" label.
OCTOBER = 10
ALL_MONTHS = list(range(1, 13))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_daily(
    n_years: int = 80,
    annual_drift_bp: float = 700.0,
    vol_ann: float = 0.16,
    october_penalty_bp: float = 0.0,
    seed: int = 136,
) -> tuple[pd.Series, dict]:
    """A reproducible daily return series with a planted October drag (or none).

    Log returns are i.i.d. normal with an annualised drift of ``annual_drift_bp`` basis
    points and annual vol ``vol_ann``, except October days which receive an additional
    ``october_penalty_bp`` per-day bias (negative = extra downside). The only forecastable
    structure is the calendar label: when ``october_penalty_bp = 0`` (the null) all months
    are identical.

    Returns ``(returns, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("1946-01-02")
    # Generate trading days: roughly 252 per year
    all_days = pd.bdate_range(start=start, periods=n_years * 252)
    n = len(all_days)
    daily_drift = annual_drift_bp * 1e-4 / 252
    daily_vol = vol_ann / np.sqrt(252)
    r = rng.normal(daily_drift, daily_vol, n)
    # Plant the October penalty on October trading days
    oct_mask = (all_days.month == OCTOBER)
    r[oct_mask] += october_penalty_bp * 1e-4
    ret = pd.Series(r, index=all_days, name="ret")
    truth = {
        "n_years": n_years,
        "annual_drift_bp": annual_drift_bp,
        "vol_ann": vol_ann,
        "october_penalty_bp": october_penalty_bp,
        "seed": seed,
        "n_obs": int(n),
    }
    return ret, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC daily, cache-first
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _CACHE_FILE)


def fetch_daily(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "1928-01-01",
) -> pd.Series:
    """Daily close-to-close return for ^GSPC since ``start``; cache-first unless ``fetch=True``.

    The result is a simple-return Series (not log-return) aligned to business days.
    Network is touched only when ``fetch=True``; the cached parquet persists so every
    downstream consumer — tests, notebooks, verify.py — is offline by default.
    """
    # First try study-local cache, then repo-wide _cache
    for cdir in (STUDY_CACHE, cache_dir):
        path = _cache_path(cdir)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            ret = df["ret"] if "ret" in df.columns else df.iloc[:, 0]
            ret.index = pd.DatetimeIndex(ret.index)
            if ret.index.tz is not None:
                ret.index = ret.index.tz_localize(None)
            return ret.sort_index()

    if not fetch:
        raise FileNotFoundError(
            f"No cached daily tape for ^GSPC. "
            f"Call fetch_daily(fetch=True) once to populate the cache."
        )

    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download("^GSPC", start=start, interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no data for ^GSPC")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    px = raw["Close"].dropna()
    px.index = pd.DatetimeIndex(px.index)
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    px = px.sort_index()
    ret = px.pct_change().dropna()
    ret.name = "ret"
    ret.index.name = "date"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_frame().to_parquet(_cache_path(cache_dir))
    return ret


def fingerprint(ret: pd.Series) -> str:
    """A short content fingerprint of the return series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(ret.to_numpy()).tobytes())
    return h.hexdigest()[:12]

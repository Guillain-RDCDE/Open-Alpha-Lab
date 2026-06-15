"""Data layer for Study 161 (Year-Ending-Five).

Two tapes, one shape (an annual return series for the S&P 500 index):

- ``synthetic_annual`` — a *deterministic, offline* generator.  A ``digit5_boost``
  knob injects an extra return in years whose last digit is 5, so tests can confirm
  the analysis correctly recovers a planted effect and finds nothing when the boost
  is zero.  ``digit5_boost = 0`` is the null — calendar digit and annual return are
  independent.

- ``load_shiller`` — real calendar-year returns derived from the Shiller S&P 500
  monthly dataset, already staged at the shared repo cache.  The study covers the
  full Shiller window (~1872–2025), giving n ≈ 15–16 observations per last-digit
  bucket — brutally small, as we say loudly.

No look-ahead is baked in here.  The calendar year ends on 31 December; the last
digit of the year is known at that date; any \"trade\" based on it enters on 1 January
of the following year.  Calendar-year returns are purely retrospective, so the study
is really about pattern-in-history rather than a live trading signal.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SHILLER_CACHE = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "_cache", "shiller_sp500.parquet")
)

# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_annual(
    n_years: int = 150,
    annual_vol: float = 0.18,
    annual_drift: float = 0.07,
    digit5_boost: float = 0.0,
    start_year: int = 1876,
    seed: int = 161,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible annual return series with an optional decennial digit-5 boost.

    Log returns are drawn i.i.d. from N(annual_drift, annual_vol**2).  In years
    whose last digit is 5, an extra ``digit5_boost`` is added to the log return.
    ``digit5_boost = 0`` is the pure null: the last digit carries zero information.
    ``digit5_boost > 0`` plants the effect the decennial-cycle folklore claims.

    Returns ``(df, truth)`` where ``df`` has columns ``year``, ``log_ret``,
    ``ret_pct``, and ``last_digit``.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, start_year + n_years)
    log_ret = rng.normal(annual_drift, annual_vol, n_years)

    # Inject the decennial boost for digit-5 years.
    last_digits = years % 10
    log_ret[last_digits == 5] += digit5_boost

    ret_pct = (np.exp(log_ret) - 1.0) * 100.0
    df = pd.DataFrame(
        {"year": years, "log_ret": log_ret, "ret_pct": ret_pct, "last_digit": last_digits}
    ).set_index("year")

    truth = {
        "n_years": n_years,
        "annual_vol": annual_vol,
        "annual_drift": annual_drift,
        "digit5_boost": digit5_boost,
        "start_year": start_year,
        "seed": seed,
        "n_digit5_years": int((last_digits == 5).sum()),
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — Shiller S&P 500 monthly, staged at the shared cache
# ---------------------------------------------------------------------------
def load_shiller(
    cache_path: str = SHILLER_CACHE,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Calendar-year simple returns derived from Shiller's monthly S&P 500 dataset.

    The shared cache lives at ``_cache/shiller_sp500.parquet`` (cols: Date, SP500,
    Dividend, ...; monthly, 1871-2026; placeholder zeros on recent rows).  We use
    the December-close SP500 price for each year to compute annual log-returns and
    convert to simple (percentage) returns for readability.

    Returns a DataFrame indexed by year with columns:
    ``log_ret, ret_pct, last_digit, decade``.

    n ≈ 154 years (1872–2025) with ~15–16 observations per last-digit bucket.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Shiller cache not found at {cache_path}.  "
            "It should be pre-staged in the repo's shared _cache/ directory."
        )
    raw = pd.read_parquet(cache_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw[raw["SP500"] > 0].copy()
    raw["year"] = raw["Date"].dt.year
    raw["month"] = raw["Date"].dt.month

    # December close for each year — end-of-year price.
    dec = raw[raw["month"] == 12].groupby("year")["SP500"].last()

    # Annual log return: ln(P_dec_t / P_dec_{t-1}).
    log_ret = np.log(dec / dec.shift(1)).dropna()
    log_ret = log_ret[log_ret.index <= end_year]

    ret_pct = (np.exp(log_ret) - 1.0) * 100.0
    df = pd.DataFrame({"log_ret": log_ret, "ret_pct": ret_pct})
    df.index.name = "year"
    df["last_digit"] = df.index % 10
    df["decade"] = (df.index // 10) * 10
    return df.sort_index()


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of the annual return series (ret_pct column)."""
    arr = df["ret_pct"].to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

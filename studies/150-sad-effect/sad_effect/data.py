"""Data layer for Study 150 (SAD-Effect).

Two tapes, one logical shape (a monthly return Series with calendar metadata):

- ``synthetic_monthly`` — a *deterministic, offline* generator. A ``sad_premium`` knob
  injects extra return in the SAD-recovery window (Oct–Mar, i.e. autumn depression
  lifting through winter-to-spring); ``sad_premium = 0`` is the null (no seasonal), so
  a test can assert the seasonal signal appears only when we plant it. Returns a
  ``(returns Series, truth dict)`` pair.

- ``fetch_shiller`` — the Kamstra–Kramer–Levi paper's actual data source: the Shiller
  monthly S&P 500 dataset, already staged at the repo's shared cache
  ``_cache/shiller_sp500.parquet``. Monthly total returns from 1871 (price change plus
  lagged dividend yield). We drop any row where SP500 is zero or NaN (the recent
  placeholder rows in the dataset). Returns a monthly simple-return Series.

Daylight proxy
~~~~~~~~~~~~~~
The SAD thesis posits that returns should track the *change in daylight duration*. We
compute a deterministic daylight proxy as follows: for Northern-Hemisphere investors at
40°N (roughly New York), the day-length in hours for each month is given by the standard
astronomical formula for the 15th of each month. The SAD signal is operationalised as
the *change in day-length* between consecutive months: negative in autumn (days getting
shorter → increasing seasonal risk aversion → lower returns in the original KKL story)
and positive in spring (days lengthening → risk aversion lifting → recovery). We expose
``daylight_proxy`` for use in the regression arm of the study.

No look-ahead: each monthly return is labelled with the calendar metadata of that
month; there is no forward reference to future returns.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

# Autumn months in KKL's definition: the SAD-onset window when days are shortening.
# Returns are predicted to be *lower* here (peak seasonal risk aversion).
# The recovery / spring window (Oct–Mar = the "SAD season" in KKL; note they call the
# autumn *reduction* phase the *risky period* and expect positive risk premia in winter).
# We test KKL's prediction: Sep–Dec = *lower* returns (onset), Jan–Mar = *higher* returns
# (recovery), vs the unconditional monthly mean.
SAD_ONSET = {9, 10, 11}      # Sep, Oct, Nov — days shortening toward winter
SAD_RECOVERY = {12, 1, 2, 3}  # Dec, Jan, Feb, Mar — shortest days → rebounding daylight


# ---------------------------------------------------------------------------
# Astronomical day-length proxy (Northern-Hemisphere, 40°N ≈ New York)
# ---------------------------------------------------------------------------
def _day_length_hours(month: int, lat_deg: float = 40.0) -> float:
    """Approximate day-length in hours on the 15th of ``month`` at ``lat_deg`` North.

    Uses the standard astronomical declination formula:
        delta = -23.45 * cos(2*pi * (DOY + 10) / 365)   [degrees]
        cos(omega0) = -tan(lat) * tan(delta)
        day_length = 2 * omega0 / (2*pi) * 24             [hours]

    Returns 24 if the sun doesn't set (polar summer), 0 if it never rises.
    """
    # Approximate day-of-year for the 15th of each month (not leap-year-precise,
    # which is fine for a monthly proxy).
    doy_15 = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    doy = doy_15[month - 1]
    lat = np.radians(lat_deg)
    decl = np.radians(-23.45 * np.cos(2.0 * np.pi * (doy + 10.0) / 365.0))
    arg = -np.tan(lat) * np.tan(decl)
    arg = max(-1.0, min(1.0, arg))  # clamp for polar regions
    omega0 = np.arccos(arg)
    return 2.0 * omega0 / (2.0 * np.pi) * 24.0


# Pre-compute the 12-month lookup (month 1..12).
_DAYLIGHT_HOURS: dict[int, float] = {m: _day_length_hours(m) for m in range(1, 13)}


def daylight_proxy(index: pd.DatetimeIndex) -> pd.Series:
    """Month-over-month change in day-length (hours) for each date in ``index``.

    Negative in autumn (shortening days → SAD onset), positive in spring (lengthening
    days → SAD recovery). This is the independent variable in the KKL regression: if SAD
    drives returns, the coefficient on this proxy should be **positive** (shorter days →
    lower returns, i.e. the change is negative and the return is low; longer days →
    higher returns).
    """
    idx = pd.DatetimeIndex(index)
    months = idx.month
    dl = pd.Series([_DAYLIGHT_HOURS[m] for m in months], index=index, name="daylight_h")
    # Month-over-month change in day-length (same month last year as baseline is wrong;
    # we want calendar-adjacent months for the continuous version).
    delta = dl - dl.shift(1)
    delta.name = "delta_daylight_h"
    return delta


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_monthly(
    n_years: int = 80,
    base_bp: float = 60.0,
    sad_premium_bp: float = 0.0,
    vol_ann: float = 0.15,
    seed: int = 150,
) -> tuple[pd.Series, dict]:
    """A reproducible monthly return series with a tunable SAD seasonal component.

    The SAD pattern (Kamstra, Kramer & Levi 2003): returns are *lower* in the SAD-onset
    window (Sep–Nov, days shortening) and *higher* in the recovery window (Dec–Mar).
    ``sad_premium_bp`` is the symmetric shift: onset months are deflated by this amount
    and recovery months are inflated by the same amount. Setting ``sad_premium_bp = 0``
    collapses to a pure random walk with no seasonal: the null hypothesis in a bottle.

    Returns ``(ret_series, truth)`` where ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range("1946-01-31", periods=n, freq="ME", name="date")

    months = pd.DatetimeIndex(idx).month
    onset_mask = np.isin(months, list(SAD_ONSET)).astype(float)
    recovery_mask = np.isin(months, list(SAD_RECOVERY)).astype(float)

    drift = base_bp * 1e-4
    # SAD signal: deflate onset, inflate recovery (symmetric around zero)
    adjustment = sad_premium_bp * 1e-4 * (recovery_mask - onset_mask)
    r = drift + adjustment + (vol_ann / np.sqrt(12)) * rng.standard_normal(n)

    truth = {
        "n_years": n_years,
        "base_bp": base_bp,
        "sad_premium_bp": sad_premium_bp,
        "vol_ann": vol_ann,
        "seed": seed,
        "has_sad": sad_premium_bp != 0.0,
    }
    return pd.Series(r, index=idx, name="ret"), truth


# ---------------------------------------------------------------------------
# Real tape — Shiller S&P 500 monthly, shared repo cache (read-only input)
# ---------------------------------------------------------------------------
def fetch_shiller(cache_dir: str = DEFAULT_CACHE) -> pd.Series:
    """Monthly total-return of the S&P 500 from the Shiller dataset (1871 – present).

    The dataset is already staged at ``_cache/shiller_sp500.parquet`` and is a
    **read-only** shared input — do not re-fetch or overwrite it. Recent months where
    SP500 is zero (placeholder rows) are dropped. Total return for month t is computed
    as the price change plus the dividend yield earned during the month:

        ret_t = (SP500_t / SP500_{t-1}) - 1  +  (Dividend_{t-1} / SP500_{t-1} / 12)

    This matches the Kamstra–Kramer–Levi (2003) approach of using total returns from
    the Shiller–Ibbotson dataset. Returns a monthly simple-return Series indexed by
    month-start dates (the Shiller dataset convention: 1871-01-01, 1871-02-01, …).
    """
    path = os.path.join(cache_dir, "shiller_sp500.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Shiller dataset not found at {path}. "
            "The shared cache at _cache/shiller_sp500.parquet is a read-only staged input."
        )
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Drop recent placeholder rows (SP500 == 0)
    df = df[df["SP500"] > 0].copy()

    # Total return: price appreciation + lagged dividend yield (monthly)
    price_ret = df["SP500"].pct_change()
    div_yield_monthly = (df["Dividend"] / df["SP500"]).shift(1) / 12.0
    total_ret = price_ret + div_yield_monthly

    total_ret = total_ret.dropna()
    total_ret.name = "ret"
    total_ret.index.name = "date"
    return total_ret


def fingerprint(ret: pd.Series) -> str:
    """A short content fingerprint of a return series, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(ret.to_numpy()).tobytes())
    return h.hexdigest()[:12]

"""Data layer for Study 148 (Lunar-Effect).

Two tapes, one logical shape (a daily return Series with a lunar-phase label):

- ``synthetic_lunar`` — a *deterministic, offline* generator. A ``full_discount``
  knob injects a negative premium on full-moon ±7-day windows relative to new-moon
  windows.  ``full_discount=0`` gives the null (pure random walk), so the
  full-vs-new return test scores zero — the study's null in a bottle.
- ``fetch_panel`` — loads S&P 500 daily returns from the shared repo cache (^GSPC,
  1928-2026) and augments with the lunar phase for each date.  Cache-only by default
  (``fetch=False`` raises if no cache file exists); ``fetch=True`` hits Yahoo Finance
  and repopulates the cache.

The lunar phase is computed entirely from pure astronomy — the synodic month of
~29.53059 days, anchored to a known new-moon epoch, with no external data fetch.
The canonical anchor is the J2000.0 epoch new moon: 6 January 2000 18:14 UTC,
Julian date 2451551.26 (per Meeus, *Astronomical Algorithms*, 2nd ed., Ch. 47).

Definition of windows (mirroring Yuan, Zheng & Zhu 2006):
  - ``NEW``  — within ±7 trading days of a new moon (phase ≈ 0 or ≈ 1).
  - ``FULL`` — within ±7 trading days of a full moon (phase ≈ 0.5).
  - ``OTHER`` — all other trading days.

YZZ (2006) report higher returns in new-moon windows, lower in full-moon windows.
We test the contrast (new − full mean return) with a HAC t-stat.

No look-ahead is baked in here — the lunar phase on day t is pure astronomy
and does not depend on any future return.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(REPO_ROOT, "_cache")

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Lunar-phase engine — pure astronomy, no external data
# ---------------------------------------------------------------------------
# Epoch: new moon at J2000.0 reference, Julian Date 2451551.26 (6 Jan 2000 18:14 UTC).
# Synodic month: 29.530588853 days (mean, Meeus 1998 Table 47.a).
_JD_EPOCH_NEW_MOON = 2451551.26
_SYNODIC_MONTH = 29.530588853

# Julian-date anchor for the Python datetime origin (noon 1 Jan 4713 BC, proleptic Julian).
# More usefully: JD of 2000-01-01 noon = 2451545.0.
_JD_J2000_NOON = 2451545.0


def _jd(date: "np.datetime64 | pd.Timestamp") -> float:
    """Julian date for a calendar date (treated as noon UTC for daily data)."""
    ts = pd.Timestamp(date)
    # Days since J2000.0 noon (2000-01-01 12:00 UTC)
    delta = (ts - pd.Timestamp("2000-01-01 12:00:00")).total_seconds() / 86400.0
    return _JD_J2000_NOON + delta


def _jd_array(index: pd.DatetimeIndex) -> np.ndarray:
    """Vectorised Julian dates for a DatetimeIndex (noon UTC per day)."""
    ref = pd.Timestamp("2000-01-01 12:00:00")
    deltas = (index - ref).total_seconds() / 86400.0
    return _JD_J2000_NOON + deltas


def lunar_phase(index: pd.DatetimeIndex) -> pd.Series:
    """Lunar phase in [0, 1) for each date in ``index``.

    Phase 0 = new moon, 0.5 = full moon.  Computed from the mean synodic month
    anchored to a known new-moon JD — no external data required.  The error
    relative to the true phase is typically < 0.5 day (~1.7% of the synodic
    month) and is well within the ±7-day window used by Yuan et al. (2006).
    """
    jds = _jd_array(index)
    phase = (jds - _JD_EPOCH_NEW_MOON) % _SYNODIC_MONTH / _SYNODIC_MONTH
    return pd.Series(phase, index=index, name="lunar_phase")


def lunar_label(
    index: pd.DatetimeIndex,
    window_days: int = 7,
) -> pd.Series:
    """Assign each date a lunar label: 'NEW', 'FULL', or 'OTHER'.

    A date is labelled 'NEW' if its fractional lunar phase corresponds to within
    ``window_days / _SYNODIC_MONTH`` of a new moon (phase near 0 or 1), and
    'FULL' if within that fraction of a full moon (phase near 0.5).  The window
    is a fraction of the synodic month, not calendar days, so it is uniform
    across the cycle regardless of trading-day density.
    """
    phase = lunar_phase(index).to_numpy()
    frac = window_days / _SYNODIC_MONTH  # ~0.237 for window=7

    # Distance to nearest new moon: min(phase, 1-phase) (phase wraps at 1=new moon)
    dist_new = np.minimum(phase, 1.0 - phase)
    # Distance to nearest full moon: |phase - 0.5|
    dist_full = np.abs(phase - 0.5)

    labels = np.where(
        dist_new <= frac, "NEW",
        np.where(dist_full <= frac, "FULL", "OTHER")
    )
    return pd.Series(labels, index=index, name="lunar_label")


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_lunar(
    n_days: int = 3000,
    full_discount: float = 0.0,
    new_premium: float = 0.0,
    base_ret: float = 4e-4,
    daily_vol: float = 0.011,
    seed: int = 148,
    start: str = "2000-01-03",
    window_days: int = 7,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily-return series with a tunable lunar premium/discount.

    Log returns are i.i.d. normal with mean ``base_ret`` and std ``daily_vol``,
    except on full-moon-window days where the mean is *reduced* by ``full_discount``
    and on new-moon-window days where it is *increased* by ``new_premium``.
    Setting both to 0 gives a pure random walk — the null — so a test can assert
    the new-vs-full gap appears only when we plant it.

    Returns ``(frame, truth)`` where ``frame`` has columns ``ret``, ``lunar_phase``,
    and ``lunar_label``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days, name="date")

    labels = lunar_label(idx, window_days=window_days)
    phase = lunar_phase(idx)

    rets = base_ret + daily_vol * rng.standard_normal(n_days)
    full_mask = labels == "FULL"
    new_mask = labels == "NEW"
    rets[full_mask.values] -= full_discount
    rets[new_mask.values] += new_premium

    frame = pd.DataFrame(
        {"ret": rets, "lunar_phase": phase.values, "lunar_label": labels.values},
        index=idx,
    )
    truth = {
        "full_discount": full_discount,
        "new_premium": new_premium,
        "base_ret": base_ret,
        "daily_vol": daily_vol,
        "n_days": n_days,
        "seed": seed,
        "window_days": window_days,
        "n_new": int(new_mask.sum()),
        "n_full": int(full_mask.sum()),
        "n_other": int((~new_mask & ~full_mask).sum()),
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tape — S&P 500 daily returns from the shared repo cache
# ---------------------------------------------------------------------------
def _gspc_cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "last_call_gspc.parquet")


def fetch_panel(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "1928-01-01",
    window_days: int = 7,
) -> pd.DataFrame:
    """S&P 500 daily returns (^GSPC, 1928-2026) with lunar-phase labels; cache-first.

    Returns a DataFrame with columns ``ret``, ``lunar_phase``, ``lunar_label``,
    indexed by trading date.  Only dates from ``start`` onward are returned.
    Network is touched only on an explicit ``fetch=True``; the shared repo cache at
    ``_cache/last_call_gspc.parquet`` already holds the full history.
    """
    cache = _gspc_cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"No cached ^GSPC data at {cache}. "
                "Call fetch_panel(fetch=True) once to populate, or run "
                "examples/verify.py --fetch."
            )
        raw = pd.read_parquet(cache)
        ret = raw["ret"] if "ret" in raw.columns else raw.iloc[:, 0]
    else:
        import yfinance as yf

        px = yf.download("^GSPC", period="max", auto_adjust=True, progress=False)
        if px.empty:
            raise RuntimeError("yfinance returned no data for ^GSPC")
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
        close_col = next(c for c in px.columns if str(c).lower() in ("close", "adj close"))
        ret = px[close_col].pct_change().dropna()
        ret.name = "ret"
        os.makedirs(cache_dir, exist_ok=True)
        ret.to_frame().to_parquet(cache)

    ret.index = pd.DatetimeIndex(ret.index).tz_localize(None)
    ret = ret[ret.index >= start].dropna()
    ret.index.name = "date"

    phase = lunar_phase(ret.index)
    labels = lunar_label(ret.index, window_days=window_days)

    frame = pd.DataFrame(
        {"ret": ret.values, "lunar_phase": phase.values, "lunar_label": labels.values},
        index=ret.index,
    )
    return frame


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of the return column, for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(frame["ret"].to_numpy()).tobytes())
    return h.hexdigest()[:12]

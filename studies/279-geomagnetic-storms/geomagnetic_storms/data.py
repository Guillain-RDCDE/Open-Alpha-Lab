"""Data layer for Study 279 (Geomagnetic-Storms).

Three components, the core two fully offline and deterministic:

- ``KP_MONTHLY_INDEX`` — a hardcoded monthly geomagnetic-activity index (an Ap-style
  planetary index) spanning 1932–2025, built from a documented model of the 11-year
  solar cycle.  The geomagnetic Ap index oscillates with the sunspot cycle: activity
  peaks a year or two *after* sunspot maximum (the "Russell-McPherron" / declining-phase
  effect), with elevated equinoctial activity (the March/September semi-annual peaks).
  The series is anchored to the canonical solar-cycle minima (cycles 17–25) and the
  well-known stormy years (1989, 2003, 2024).  It is deterministic, small, and reproduced
  here so the core is fully offline.  A monthly value at or above the high quantile is a
  "stormy" month; at or below the low quantile is a "calm" month — the Krivelyova-Robotti
  (2003) classification.

- ``synthetic_monthly`` — a deterministic, offline monthly equity-return generator with an
  optional planted "storm penalty" knob (``storm_bps``: bps subtracted from stormy-month
  returns).  ``storm_bps = 0`` is the null (storms carry no information), so the test-suite
  can confirm the machinery is truthful before looking at real data.

- ``fetch_gspc_monthly`` — real ^GSPC monthly (December/December-style) price returns from
  the repo-level cache (``_cache/^GSPC_split_only.parquet``).  Cache-only by default
  (``fetch=False``); the network (yfinance) is touched only on an explicit ``fetch=True``.

**No look-ahead**: the geomagnetic index for month *t* is known by the *end* of month *t*
(it is a contemporaneous physical measurement), and is matched to the return earned *in
that same month*.  Following Krivelyova-Robotti, the tradable interpretation uses the
**prior** month's storm classification to position for the current month — implemented with
an explicit one-month execution lag in ``strategy.long_short_storm``.

**Price-only / total-return**: the cached ^GSPC series is a *price* index (no dividends),
so all real-tape return numbers are price-only and labelled as such.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))

GSPC_CACHE = os.path.join(REPO_CACHE, "^GSPC_split_only.parquet")

# ---------------------------------------------------------------------------
# The geomagnetic Ap index — the study's deterministic, offline core
# ---------------------------------------------------------------------------
# Canonical solar-cycle minima (year.fraction), cycles 17–25.  Source: SILSO /
# NOAA SWPC solar-cycle tables.  Geomagnetic activity (the Ap index) tracks the
# sunspot cycle but peaks in the *declining* phase, ~1–2 years after sunspot
# maximum, and is amplified near the equinoxes (the semi-annual / Russell-
# McPherron effect).  We reconstruct a monthly Ap proxy from these anchors.
#
# Cycle minima (approx, decimal year):
_SOLAR_MINIMA = [
    1933.8, 1944.2, 1954.3, 1964.9, 1976.5, 1986.8,
    1996.4, 2008.9, 2019.9, 2030.5,
]
# Mean cycle length ~10.6 yr; mean amplitude of Ap activity scaled to ~16 (avg)
# with stormy peaks reaching ~30+.

_START_YEAR = 1932
_END_YEAR = 2025

# Documented exceptionally stormy months (great geomagnetic storms), added as
# spikes on top of the cyclical model.  (year, month): extra Ap boost.
# March 1989 (Quebec blackout), Oct/Nov 2003 (Halloween storms), May 2024
# (Gannon storm), Aug 1972, March 2015 (St Patrick's Day storm).
_STORM_SPIKES = {
    (1972, 8): 22.0,
    (1989, 3): 30.0,
    (1991, 3): 18.0,
    (2000, 7): 16.0,
    (2001, 3): 16.0,
    (2003, 10): 28.0,
    (2003, 11): 20.0,
    (2004, 11): 16.0,
    (2015, 3): 18.0,
    (2017, 9): 16.0,
    (2024, 5): 26.0,
}


def _build_kp_index() -> pd.Series:
    """Deterministically reconstruct a monthly Ap geomagnetic index, 1932–2025.

    The model superimposes:
      * an 11-year solar cycle (each cycle a half-sine between successive minima),
        with the geomagnetic peak shifted ~1.5 yr into the declining phase;
      * a semi-annual (equinoctial) modulation peaking near the March and
        September equinoxes (the Russell-McPherron effect);
      * documented great-storm spikes (1989, 2003, 2024, ...).
    No random noise: the series is a pure deterministic function of the date, so
    it is byte-for-byte reproducible.
    """
    dates = pd.date_range(f"{_START_YEAR}-01-31", f"{_END_YEAR}-12-31", freq="ME")
    vals = np.empty(len(dates), dtype=float)

    minima = np.array(_SOLAR_MINIMA)
    for i, d in enumerate(dates):
        yr = d.year + (d.month - 0.5) / 12.0
        # Find the bracketing cycle minima
        below = minima[minima <= yr]
        above = minima[minima > yr]
        if len(below) == 0:
            lo = minima[0] - 10.6
            hi = minima[0]
        elif len(above) == 0:
            lo = minima[-1]
            hi = minima[-1] + 10.6
        else:
            lo = below[-1]
            hi = above[0]
        length = hi - lo
        phase = (yr - lo) / length  # 0..1 within the cycle
        # Sunspot half-sine peaks at phase 0.4; geomagnetic activity lags by ~0.15
        geo_phase = phase - 0.15
        cyc = np.sin(np.pi * np.clip(geo_phase, 0.0, 1.0)) ** 0.8
        # Secondary declining-phase hump (coronal-hole high-speed streams)
        cyc += 0.45 * np.sin(np.pi * np.clip((phase - 0.55) / 0.45, 0.0, 1.0)) ** 2
        # Semi-annual equinoctial modulation (peaks ~March=3 and ~Sept=9)
        semi = 1.0 + 0.35 * np.cos(4.0 * np.pi * (d.month - 3.0) / 12.0)
        base = 8.0 + 18.0 * cyc * semi
        spike = _STORM_SPIKES.get((d.year, d.month), 0.0)
        vals[i] = base + spike

    s = pd.Series(vals, index=dates, name="ap")
    return s


_KP_SERIES = _build_kp_index()


def kp_monthly_index(
    start_year: int = _START_YEAR,
    end_year: int = _END_YEAR,
) -> pd.DataFrame:
    """Return the hardcoded monthly geomagnetic Ap index as a tidy frame.

    Columns: ``date`` (month-end DatetimeIndex), ``ap`` (float geomagnetic index).
    The series is deterministic — a pure function of the calendar — so every
    reader reconstructs identical values.
    """
    s = _KP_SERIES
    mask = (s.index.year >= start_year) & (s.index.year <= end_year)
    out = s[mask].to_frame()
    out.index.name = "date"
    return out.copy()


def storm_classification(
    ap: pd.Series,
    hi_q: float = 0.80,
    lo_q: float = 0.20,
) -> pd.Series:
    """Label each month 'stormy' / 'calm' / 'normal' from the Ap index quantiles.

    Following Krivelyova-Robotti (2003): months in the top ``hi_q`` quantile of
    geomagnetic activity are 'stormy'; months in the bottom ``lo_q`` quantile are
    'calm'; the middle band is 'normal'.  The thresholds are computed on the
    *whole* supplied series (an in-sample classification — named as such on the
    Signal axis; a walk-forward version would use only past data and is weaker).

    Returns a Series of strings aligned to ``ap``.
    """
    ap = pd.Series(ap).astype(float)
    hi = ap.quantile(hi_q)
    lo = ap.quantile(lo_q)
    out = pd.Series("normal", index=ap.index, name="regime")
    out[ap >= hi] = "stormy"
    out[ap <= lo] = "calm"
    return out


# ---------------------------------------------------------------------------
# Synthetic monthly return generator — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_monthly(
    n_months: int = 360,
    storm_bps: float = 0.0,
    base_ret: float = 0.007,
    vol_ann: float = 0.15,
    hi_q: float = 0.80,
    lo_q: float = 0.20,
    seed: int = 279,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible monthly-return series with an optional planted storm penalty.

    Months inherit the deterministic Ap index (reused, tiled if needed) and are
    classified into stormy / calm / normal.  Each month's return is drawn i.i.d.
    from N(base_ret, (vol_ann/sqrt(12))^2); in 'stormy' months an extra drift of
    ``-storm_bps`` bps is subtracted (storms depress returns, per the hypothesis).
    ``storm_bps = 0`` is the null — regime carries no information.

    Returns ``(df, truth)`` where ``df`` has columns ``date``, ``ret`` (monthly
    simple return), ``ap`` (geomagnetic index), ``regime`` ('stormy'/'calm'/
    'normal'), and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    base_idx = kp_monthly_index().index
    if n_months <= len(base_idx):
        idx = base_idx[:n_months]
        ap = _KP_SERIES.reindex(idx).to_numpy()
    else:
        reps = int(np.ceil(n_months / len(base_idx)))
        ap = np.tile(_KP_SERIES.to_numpy(), reps)[:n_months]
        idx = pd.date_range(base_idx[0], periods=n_months, freq="ME")

    ap_s = pd.Series(ap, index=idx)
    regime = storm_classification(ap_s, hi_q=hi_q, lo_q=lo_q).to_numpy()

    vol_m = vol_ann / np.sqrt(12.0)
    drift = np.where(regime == "stormy", base_ret - storm_bps * 1e-4, base_ret)
    rets = drift + rng.normal(0.0, vol_m, n_months)

    df = pd.DataFrame(
        {"date": idx, "ret": rets, "ap": ap, "regime": regime}
    ).set_index("date")

    truth = {
        "n_months": n_months,
        "storm_bps": storm_bps,
        "base_ret": base_ret,
        "vol_ann": vol_ann,
        "hi_q": hi_q,
        "lo_q": lo_q,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real data — ^GSPC monthly returns (repo-level cache, cache-only by default)
# ---------------------------------------------------------------------------
def fetch_gspc_monthly(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
    start_year: int = 1932,
    end_year: int = 2025,
    hi_q: float = 0.80,
    lo_q: float = 0.20,
) -> pd.DataFrame:
    """Load ^GSPC monthly price returns and join with the geomagnetic index.

    Cache-only by default: reads the daily ^GSPC parquet at
    ``_cache/^GSPC_split_only.parquet`` (price index, no dividends), resamples to
    month-end, computes the monthly simple return, and joins each month with its
    hardcoded geomagnetic Ap value and storm/calm/normal regime.

    Network (yfinance) is touched only on an explicit ``fetch=True``.

    Returns a DataFrame indexed by month-end date with columns:
      ``ret`` (monthly price return), ``ap`` (geomagnetic index), ``regime``
      ('stormy'/'calm'/'normal'), ``mkt_up`` (bool: ret > 0).

    Raises ``FileNotFoundError`` if cache absent and ``fetch=False``.
    """
    if fetch:
        import yfinance as yf  # lazy import: network only on fetch=True

        raw = yf.download(
            "^GSPC", start="1927-01-01", interval="1d",
            auto_adjust=False, progress=False,
        )
        if raw.empty:
            raise RuntimeError("yfinance returned no ^GSPC data")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        raw.to_parquet(cache_path)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"^GSPC cache not found at {cache_path}. "
            "Call fetch_gspc_monthly(fetch=True) once to populate it."
        )

    px = pd.read_parquet(cache_path)
    if "Close" not in px.columns:
        raise KeyError("expected a 'Close' column in the ^GSPC cache")
    close = px["Close"].astype(float)
    close.index = pd.to_datetime(close.index)

    # Month-end close and monthly simple return.
    me = close.resample("ME").last()
    ret = me.pct_change().dropna()
    ret.name = "ret"

    df = ret.to_frame()
    df = df[(df.index.year >= start_year) & (df.index.year <= end_year)].copy()

    # Join with the geomagnetic Ap index on month-end date.
    ap = _KP_SERIES.copy()
    ap.index = ap.index + pd.offsets.MonthEnd(0)
    df["ap"] = ap.reindex(df.index)
    df = df.dropna(subset=["ap"])

    df["regime"] = storm_classification(df["ap"], hi_q=hi_q, lo_q=lo_q)
    df["mkt_up"] = df["ret"] > 0
    return df


def fingerprint(df: pd.DataFrame, col: str = "ret") -> str:
    """A short content fingerprint of a return column, for the as-of stamp."""
    vals = pd.Series(df[col]).dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]

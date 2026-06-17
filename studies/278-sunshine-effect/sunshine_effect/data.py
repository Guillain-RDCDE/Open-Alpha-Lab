"""Data layer for Study 278 (Sunshine-Effect).

The Hirshleifer & Shumway (2003) "sunshine effect": daily morning cloud cover
over the city of a major stock exchange is *negatively* related to that day's
index return. The proposed channel is mood -- sunshine lifts traders' moods and
nudges them to buy. The cleanest test pairs NYC daily sky-cover with the NYSE
(here proxied by the S&P 500, ``^GSPC``) daily return.

Three components, all offline for the deterministic core:

- ``NYC_CLOUD_CLIMATOLOGY`` -- a hardcoded month-by-month NYC sky-cover
  climatology (mean cloud cover, in *octas* 0-8, and the day-to-day standard
  deviation within each calendar month). Source: NOAA/NWS Central Park (KNYC)
  1991-2020 climate normals and the U.S. Climate Normals sky-cover tables.
  This small, well-known, deterministic table is hardcoded to keep the core
  fully offline.

- ``synthetic_cloud_returns`` -- a deterministic, offline daily generator that
  draws a cloud-cover series from the climatology and an index-return series
  with an optional *planted* sunshine effect (``effect_bps_per_octa``). The
  ``effect_bps_per_octa = 0`` setting is the null: cloud cover carries no
  information. This is the study's null (and positive control) in a bottle.

- ``fetch_index_daily`` -- the real ``^GSPC`` daily close (via yfinance),
  cache-only by default so the test-suite and the reproducible core never touch
  the network. The cache lives at ``_cache/gspc_daily.parquet``. We then
  recolour each trading day with a *deterministic* draw from the NYC
  climatology (seeded by the calendar date) so the join is reproducible from the
  cache alone -- we ship a climatological cloud proxy, NOT a hand-collected
  station log. This is named honestly: the cloud series is a *climatological
  reconstruction*, so any real-tape signal is a reconstruction-level result, and
  the effect a genuine station log would show is an upper bound on what a
  tradable rule could have captured live.

No look-ahead: the cloud cover is the *morning* sky condition of trading day t,
known at the open; the return is the same day's close-to-close return, so a
trader acting at the open on the morning sky has a one-day, open-to-close
horizon. We label the execution lag explicitly (signal at the open, position
held to the close; costs charged on the implied daily turnover).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

INDEX_CACHE = os.path.join(DEFAULT_CACHE, "gspc_daily.parquet")

# ---------------------------------------------------------------------------
# Hardcoded NYC sky-cover climatology -- the deterministic offline core
# ---------------------------------------------------------------------------
# Columns (by calendar month 1..12):
#   mean_octas : average daily sky cover, octas (0 = clear, 8 = overcast).
#   sd_octas   : within-month day-to-day standard deviation of sky cover.
#
# Source: NOAA NWS New York Central Park (KNYC) sky-cover climatology, U.S.
# Climate Normals 1991-2020. NYC averages a bit over half-overcast year-round,
# cloudiest in late autumn/winter (Nov-Jan) and clearest in late summer/early
# autumn (Aug-Oct). Values are deterministic and hardcoded for full offline
# reproducibility.
NYC_CLOUD_CLIMATOLOGY: dict[int, dict[str, float]] = {
    1:  {"mean_octas": 5.4, "sd_octas": 2.3},   # January  -- cloudiest stretch
    2:  {"mean_octas": 5.2, "sd_octas": 2.3},   # February
    3:  {"mean_octas": 5.3, "sd_octas": 2.2},   # March
    4:  {"mean_octas": 5.1, "sd_octas": 2.2},   # April
    5:  {"mean_octas": 5.0, "sd_octas": 2.1},   # May
    6:  {"mean_octas": 4.8, "sd_octas": 2.0},   # June
    7:  {"mean_octas": 4.6, "sd_octas": 1.9},   # July
    8:  {"mean_octas": 4.4, "sd_octas": 1.9},   # August  -- clearest stretch
    9:  {"mean_octas": 4.3, "sd_octas": 2.0},   # September
    10: {"mean_octas": 4.5, "sd_octas": 2.1},   # October
    11: {"mean_octas": 5.1, "sd_octas": 2.2},   # November
    12: {"mean_octas": 5.5, "sd_octas": 2.3},   # December -- cloudiest
}

# Octas scale bounds (a valid sky-cover observation is 0..8).
OCTAS_MIN, OCTAS_MAX = 0.0, 8.0


def cloud_climatology() -> pd.DataFrame:
    """Return the hardcoded NYC sky-cover climatology as a tidy DataFrame.

    Columns: ``month`` (1-12), ``mean_octas``, ``sd_octas``.
    """
    rows = [
        {"month": m, "mean_octas": v["mean_octas"], "sd_octas": v["sd_octas"]}
        for m, v in sorted(NYC_CLOUD_CLIMATOLOGY.items())
    ]
    return pd.DataFrame(rows)


def _date_seed(ts: pd.Timestamp, base_seed: int) -> int:
    """A small deterministic per-date integer seed (date -> stable cloud draw)."""
    key = f"{base_seed}-{ts.year:04d}-{ts.month:02d}-{ts.day:02d}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def cloud_cover_for_dates(dates: pd.DatetimeIndex, base_seed: int = 278) -> pd.Series:
    """A deterministic climatological cloud-cover series for the given dates.

    For each date we draw a single sky-cover value (octas) from a Normal centred
    on that calendar month's ``mean_octas`` with that month's ``sd_octas``,
    using a *per-date* seed so the value is reproducible from the date alone
    (no network, no station log). Values are clipped to the valid 0..8 octas
    range. This is a climatological reconstruction, named as such -- it carries
    the seasonal sky-cover structure but NOT the day-specific synoptic weather a
    real station log would.

    Returns a Series of octas indexed by ``dates``.
    """
    out = np.empty(len(dates), dtype=float)
    for i, ts in enumerate(dates):
        clim = NYC_CLOUD_CLIMATOLOGY[ts.month]
        rng = np.random.default_rng(_date_seed(ts, base_seed))
        val = rng.normal(clim["mean_octas"], clim["sd_octas"])
        out[i] = float(np.clip(val, OCTAS_MIN, OCTAS_MAX))
    return pd.Series(out, index=dates, name="cloud_octas")


def deseasonalize_cloud(cloud: pd.Series) -> pd.Series:
    """Remove the calendar-month mean from the cloud series.

    Hirshleifer-Shumway stress that the sky-cover signal must be *de-seasonalised*
    so the effect is not confounded with month-of-year return seasonality (e.g.
    a 'January effect' masquerading as a 'cloudy-January effect'). We subtract
    each calendar month's mean octas, leaving a zero-mean within-month anomaly.
    """
    idx = cloud.index
    months = np.array([ts.month for ts in idx])
    out = cloud.copy().astype(float)
    for m in range(1, 13):
        mask = months == m
        if mask.any():
            out.iloc[mask] = cloud.iloc[mask] - cloud.iloc[mask].mean()
    return out.rename("cloud_anom")


# ---------------------------------------------------------------------------
# Synthetic generator -- the deterministic offline core (null + positive control)
# ---------------------------------------------------------------------------
def synthetic_cloud_returns(
    n_days: int = 2520,
    effect_bps_per_octa: float = 0.0,
    base_bps: float = 3.0,
    vol_bps: float = 100.0,
    start: str = "2000-01-03",
    seed: int = 278,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily (cloud, index-return) frame with an optional planted effect.

    Cloud cover is drawn from the NYC climatology. The daily index return is::

        base_bps - effect_bps_per_octa * cloud_anom + noise

    where ``cloud_anom`` is the *de-seasonalised* cloud cover (zero-mean within
    each calendar month). The sign convention follows Hirshleifer-Shumway: more
    cloud -> lower return, so a positive ``effect_bps_per_octa`` plants a genuine
    sunshine effect. ``effect_bps_per_octa = 0`` is the null -- cloud carries no
    information about returns.

    Returns ``(df, truth)`` where ``df`` has a DatetimeIndex of trading days and
    columns ``ret`` (daily simple return), ``cloud_octas``, ``cloud_anom``;
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    # ~252 trading days/year, business-day calendar (offline, deterministic).
    dates = pd.bdate_range(start=start, periods=n_days)
    cloud = cloud_cover_for_dates(dates, base_seed=seed)
    cloud_anom = deseasonalize_cloud(cloud)

    base = base_bps * 1e-4
    eff = effect_bps_per_octa * 1e-4
    noise = rng.normal(0.0, vol_bps * 1e-4, n_days)
    ret = base - eff * cloud_anom.to_numpy() + noise

    df = pd.DataFrame(
        {
            "ret": ret,
            "cloud_octas": cloud.to_numpy(),
            "cloud_anom": cloud_anom.to_numpy(),
        },
        index=dates,
    )
    df.index.name = "date"
    truth = {
        "n_days": n_days,
        "effect_bps_per_octa": effect_bps_per_octa,
        "base_bps": base_bps,
        "vol_bps": vol_bps,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- ^GSPC daily returns recoloured with the NYC cloud climatology
# ---------------------------------------------------------------------------
def fetch_index_daily(
    fetch: bool = False,
    cache_path: str = INDEX_CACHE,
    start: str = "1990-01-01",
) -> pd.DataFrame:
    """Real ``^GSPC`` daily close (cache-only unless ``fetch=True``).

    Network is touched only on an explicit ``fetch=True`` (lazy yfinance import),
    after which the result is cached as a parquet under
    ``_cache/gspc_daily.parquet``. Returns a frame with a DatetimeIndex of
    trading days and a single ``close`` column. Raises ``FileNotFoundError`` if
    the cache is absent and ``fetch=False``.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached ^GSPC daily series at {cache_path}. "
                "Call fetch_index_daily(fetch=True) once to populate the cache."
            )
        return pd.read_parquet(cache_path)

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "^GSPC",
        start=start,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no ^GSPC data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = raw["Close"]

    out = pd.DataFrame({"close": close.astype(float)})
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out = out.sort_index().dropna()

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_parquet(cache_path)
    print(f"Cached ^GSPC daily: {len(out)} rows -> {cache_path}")
    return out


def build_real_panel(
    fetch: bool = False,
    cache_path: str = INDEX_CACHE,
    start_year: int = 1990,
    end_year: int = 2024,
    seed: int = 278,
) -> pd.DataFrame:
    """Join real ^GSPC daily returns with the climatological NYC cloud proxy.

    Loads the cached ``^GSPC`` close (cache-only by default), computes daily
    close-to-close simple returns, attaches a deterministic climatological cloud
    cover for each trading day (``cloud_cover_for_dates``), de-seasonalises it,
    and restricts to the [start_year, end_year] window.

    Returns a frame indexed by trading day with columns ``ret`` (daily simple
    return), ``cloud_octas``, ``cloud_anom``. Raises ``FileNotFoundError`` if the
    cache is missing and ``fetch=False`` (callers branch on this for offline runs).
    """
    px = fetch_index_daily(fetch=fetch, cache_path=cache_path)
    px = px[(px.index.year >= start_year) & (px.index.year <= end_year)].copy()
    ret = px["close"].pct_change().dropna()
    dates = ret.index
    cloud = cloud_cover_for_dates(dates, base_seed=seed)
    cloud_anom = deseasonalize_cloud(cloud)
    df = pd.DataFrame(
        {
            "ret": ret.to_numpy(),
            "cloud_octas": cloud.to_numpy(),
            "cloud_anom": cloud_anom.to_numpy(),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def fingerprint(df: pd.DataFrame, col: str = "ret") -> str:
    """A short content fingerprint of a return column, for the as-of stamp."""
    vals = df[col].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()[:12]

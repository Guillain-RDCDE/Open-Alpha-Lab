"""Data layer for Study 269 (Baltic-Dry).

The claim under test: *does the Baltic Dry Index (BDI) -- a daily measure of the
cost of shipping dry bulk commodities (iron ore, coal, grain) -- lead the stock
market?*  The folk story is that freight rates are a clean, un-gameable read on
real global demand, so a rising BDI should foreshadow rising equities.

Three tapes, one shape (a monthly frame indexed by month-end with a ``bdi``
level column and a ``sp500`` level column):

- ``synthetic_series`` -- a *deterministic, offline* generator.  A ``beta`` knob
  plants an exact predictive relationship: next month's S&P log-return depends
  linearly on this month's BDI log-change.  ``beta = 0`` is the null (BDI carries
  no forward information).  This is the study's positive control in a bottle.

- ``CURATED_BDI_MONTHLY`` -- a hardcoded, documented monthly Baltic Dry Index
  series (1985-2026).  The BDI has no clean free API (the daily index is a
  licensed Baltic Exchange product), so we hardcode the well-known monthly
  trajectory anchored to widely reported waypoints: launch at 1000 (Jan 1985),
  the 2008 super-spike to ~11,440 (May 2008) and the crash to ~663 (Dec 2008),
  the 2016 trough near ~300 (Feb 2016), the 2021 rebound to ~5,650 (Oct 2021),
  and the 2022-2026 normalisation.  Levels between waypoints are filled by
  deterministic log-interpolation.  This is a *curated reconstruction*, not the
  official fixing -- it reproduces the index's shape and order of magnitude, not
  every daily print.  Treat real-tape results as indicative.

- ``fetch_real`` -- the real monthly tape from cache.  Pairs the curated BDI
  series (or, on ``fetch=True``, the BDRY freight ETF as a 2018+ proxy) with
  monthly S&P 500 closes.  The S&P side reads the repo-level Shiller parquet
  (cache-only, no network); ``fetch=True`` instead pulls ^GSPC + BDRY from
  yfinance and caches the result locally.

**No look-ahead / execution lag**: the signal at month-end *t* (a function of
BDI data through *t*) predicts the S&P return earned *during* month *t+1*.
Positions are formed at the close of month *t* and held through *t+1* -- one full
month of execution lag.  The BDI is published with a one-day lag at most, so a
month-end signal is comfortably actionable.

**Survivorship / look-back caveat**: the curated waypoints are chosen with
hindsight knowledge of the index's history.  They are public, well-reported
levels, but a curated series can flatter an in-sample fit.  This is named on the
Signal axis -- real-tape numbers here are indicative, not a live backtest of the
official fixing.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")
REPO_CACHE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "_cache"))

REAL_CACHE = os.path.join(DEFAULT_CACHE, "bdi_sp500_monthly.parquet")
SHILLER_CACHE = os.path.join(REPO_CACHE, "shiller_sp500.parquet")

SEED = 269

# ---------------------------------------------------------------------------
# Curated monthly Baltic Dry Index waypoints -- the deterministic offline core
# ---------------------------------------------------------------------------
# (year, month, level).  These are widely reported BDI levels; the series in
# between is filled by deterministic log-interpolation in ``curated_bdi``.
# The BDI is a licensed Baltic Exchange product with no clean free feed, so we
# reconstruct its monthly shape from public waypoints.  This is NOT the official
# daily fixing -- it reproduces magnitude and turning points, not every print.
_BDI_WAYPOINTS: list[tuple[int, int, float]] = [
    (1985, 1, 1000.0),    # index launched at 1000
    (1986, 7, 565.0),     # mid-80s shipping slump
    (1989, 12, 1500.0),
    (1992, 6, 1100.0),
    (1995, 6, 2200.0),
    (1998, 12, 850.0),    # Asian crisis demand shock
    (2000, 12, 1500.0),
    (2003, 12, 4700.0),   # China commodity boom begins
    (2004, 12, 4500.0),
    (2005, 8, 1800.0),    # 2005 pullback
    (2007, 12, 9100.0),   # pre-crisis surge
    (2008, 5, 11440.0),   # all-time peak (~May 2008)
    (2008, 12, 663.0),    # GFC freight collapse
    (2009, 11, 4500.0),   # stimulus-driven bounce
    (2010, 12, 1800.0),
    (2012, 2, 650.0),     # overcapacity glut
    (2013, 12, 2300.0),
    (2015, 12, 480.0),
    (2016, 2, 300.0),     # record low (~290-300, Feb 2016)
    (2017, 12, 1580.0),
    (2018, 12, 1270.0),
    (2019, 9, 2500.0),
    (2020, 5, 410.0),     # COVID demand air-pocket
    (2021, 10, 5650.0),   # post-COVID congestion spike
    (2022, 6, 2300.0),
    (2023, 2, 530.0),     # 2023 trough
    (2023, 12, 2100.0),
    (2024, 12, 1000.0),
    (2026, 5, 1400.0),    # study as-of
]


def curated_bdi(
    start: str = "1985-01-31",
    end: str = "2026-05-31",
) -> pd.Series:
    """Curated monthly Baltic Dry Index level (deterministic, offline).

    Builds a month-end series by log-linear interpolation between the public
    ``_BDI_WAYPOINTS``.  Index is a month-end ``DatetimeIndex``; values are the
    reconstructed BDI level.  This is a curated reconstruction of the index's
    shape -- not the official Baltic Exchange fixing.
    """
    months = pd.date_range(start, end, freq="ME")
    # anchor positions in "months since start" space
    base = months[0]
    way_dates = [pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) for y, m, _ in _BDI_WAYPOINTS]
    way_x = np.array([(d.year - base.year) * 12 + (d.month - base.month) for d in way_dates], dtype=float)
    way_logy = np.array([np.log(lvl) for *_, lvl in _BDI_WAYPOINTS])

    x = np.array([(d.year - base.year) * 12 + (d.month - base.month) for d in months], dtype=float)
    logy = np.interp(x, way_x, way_logy)
    levels = np.exp(logy)
    return pd.Series(levels, index=months, name="bdi")


CURATED_BDI_MONTHLY = curated_bdi()


# ---------------------------------------------------------------------------
# Synthetic series -- the deterministic positive control
# ---------------------------------------------------------------------------
def synthetic_series(
    n_months: int = 360,
    beta: float = 0.6,
    bdi_vol: float = 0.12,
    sp_base: float = 0.006,
    sp_vol: float = 0.043,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """A monthly (bdi, sp500) frame with a *known* BDI->equity lead-lag.

    The BDI log-change is an AR(1)-ish noise process.  Next month's S&P
    log-return is::

        sp_ret[t+1] = sp_base + beta * bdi_logchange[t] + noise[t+1]

    so a positive BDI change this month plants a positive expected equity return
    next month, scaled by ``beta``.  ``beta = 0`` is the null: the BDI carries no
    forward information and any measured predictability is pure luck.

    Returns ``(df, truth)`` where ``df`` has month-end index and columns
    ``bdi`` and ``sp500`` (cumulative levels from 1000), and ``truth`` records
    the planted parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("1990-01-31", periods=n_months, freq="ME")

    # BDI log-changes: mean-reverting noise
    bdi_chg = np.zeros(n_months)
    for t in range(1, n_months):
        bdi_chg[t] = 0.35 * bdi_chg[t - 1] + rng.normal(0.0, bdi_vol)
    bdi_level = 1000.0 * np.exp(np.cumsum(bdi_chg))

    # S&P log-returns: base + beta * (lagged BDI change) + noise
    sp_noise = rng.normal(0.0, sp_vol, n_months)
    sp_ret = np.full(n_months, sp_base) + sp_noise
    sp_ret[1:] += beta * bdi_chg[:-1]  # this-month BDI change drives next-month return
    sp_level = 1000.0 * np.exp(np.cumsum(sp_ret))

    df = pd.DataFrame({"bdi": bdi_level, "sp500": sp_level}, index=months)
    truth = {
        "beta": beta,
        "bdi_vol": bdi_vol,
        "sp_base": sp_base,
        "sp_vol": sp_vol,
        "n_months": n_months,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- curated BDI x Shiller S&P 500 (cache-only by default)
# ---------------------------------------------------------------------------
def _shiller_monthly_sp500(
    cache_dir_path: str = SHILLER_CACHE,
    start: str = "1985-01-31",
) -> pd.Series:
    """Monthly S&P 500 nominal price level from the repo-level Shiller cache."""
    if not os.path.exists(cache_dir_path):
        raise FileNotFoundError(
            f"Shiller cache not found at {cache_dir_path}. "
            "The repo-level _cache/shiller_sp500.parquet must be present."
        )
    sh = pd.read_parquet(cache_dir_path)
    if "Date" in sh.columns:
        idx = pd.to_datetime(sh["Date"], errors="coerce")
    else:
        idx = pd.to_datetime(sh.index)
    sp = pd.Series(sh["SP500"].to_numpy(dtype=float), index=idx, name="sp500")
    sp = sp[sp > 0].sort_index()
    sp.index = sp.index + pd.offsets.MonthEnd(0)
    sp = sp[~sp.index.duplicated(keep="last")]
    return sp[sp.index >= pd.Timestamp(start)]


def fetch_real(
    fetch: bool = False,
    cache_path: str = REAL_CACHE,
    use_proxy: bool = False,
) -> pd.DataFrame:
    """Monthly (bdi, sp500) real tape; cache-only unless ``fetch=True``.

    Default (``fetch=False``): pair the curated monthly BDI series with monthly
    S&P 500 closes from the repo Shiller cache.  No network.  If a previously
    fetched local cache exists at ``cache_path`` it is used instead.

    ``fetch=True``: pull monthly ^GSPC closes from yfinance, and -- if
    ``use_proxy`` -- the BDRY dry-bulk freight ETF (inception 2018) as a tradable
    BDI proxy; otherwise keep the curated BDI.  The merged frame is cached to
    ``cache_path``.

    Columns: ``bdi`` (level), ``sp500`` (level).  Index: month-end dates.
    """
    if not fetch:
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)
        # Build offline from curated BDI x Shiller S&P
        sp = _shiller_monthly_sp500()
        bdi = CURATED_BDI_MONTHLY
        df = pd.concat([bdi, sp], axis=1, join="inner").dropna()
        df.index.name = "date"
        return df

    # ------------------------------------------------------------------
    # Live fetch
    # ------------------------------------------------------------------
    import yfinance as yf  # lazy import: network only on fetch=True

    gspc = yf.download("^GSPC", start="1985-01-01", interval="1mo",
                       auto_adjust=True, progress=False)
    if gspc.empty:
        raise RuntimeError("yfinance returned no ^GSPC data")
    sp = gspc["Close"]
    if isinstance(sp, pd.DataFrame):
        sp = sp.iloc[:, 0]
    sp.index = pd.to_datetime(sp.index) + pd.offsets.MonthEnd(0)
    sp = sp.rename("sp500").sort_index()

    if use_proxy:
        bdry = yf.download("BDRY", start="2018-01-01", interval="1mo",
                           auto_adjust=True, progress=False)
        if bdry.empty:
            raise RuntimeError("yfinance returned no BDRY data")
        b = bdry["Close"]
        if isinstance(b, pd.DataFrame):
            b = b.iloc[:, 0]
        b.index = pd.to_datetime(b.index) + pd.offsets.MonthEnd(0)
        bdi = b.rename("bdi").sort_index()
    else:
        bdi = CURATED_BDI_MONTHLY

    df = pd.concat([bdi, sp], axis=1, join="inner").dropna()
    df.index.name = "date"

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path)
    print(f"Cached real tape: {df.shape[0]} months -> {cache_path}")
    return df


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a frame (last row), for the as-of stamp."""
    arr = df.iloc[-1].dropna().to_numpy(dtype=float)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

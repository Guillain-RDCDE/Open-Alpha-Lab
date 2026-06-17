"""Data layer for Study 267 (M2-Growth).

The claim under test: money-supply (M2) growth drives the stock market. The
monetarist intuition (Friedman) is that "inflation is always and everywhere a
monetary phenomenon" and that excess money creation eventually flows into asset
prices — so fast M2 growth should foreshadow strong equity returns, and a money
slowdown should foreshadow weak ones. Retail commentary in 2020-2023 leaned
hard on the chart of M2 YoY growth overlaid on the S&P 500.

Three components, the offline core fully deterministic:

- ``M2_YOY_ANCHORS`` — a hardcoded table of month-anchored M2 year-over-year
  growth (%) at a coarse monthly cadence, capturing the well-known historical
  contour of US M2 growth: the double-digit money growth of the 1970s, the
  near-zero readings of the early 1990s and 2010, the COVID spike to ~27% in
  early 2021, and the first-ever *negative* YoY prints in 2023. Source contour:
  FRED series ``M2SL`` (M2 money stock, seasonally adjusted) YoY % change.
  Hardcoded so the core is offline and reproducible; the series is monotone-
  interpolated to a clean monthly grid.

- ``synthetic_panel`` — a deterministic, offline generator that builds a monthly
  M2-YoY series and an S&P return series with an optional planted "M2 premium"
  knob. ``premium = 0`` is the null (M2 growth carries no forward information).

- ``fetch_gspc_monthly`` — real ^GSPC (S&P 500 price index) monthly closes via
  yfinance, cache-only by default (``fetch=False``). The network is touched only
  on an explicit ``fetch=True`` (lazy yfinance import). Cache lives at
  ``_cache/gspc_monthly.parquet``.

No look-ahead: M2 for month t is *released with a lag* and is only knowable
weeks after month-end. We impose a one-month execution lag — the M2 reading
dated month t is used to predict the S&P return *earned in month t+1* (and, for
the longer-horizon test, the next 12 months). Returns are price-only on ^GSPC
(no dividends), which we label on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(_HERE, "..", "_cache"))
GSPC_CACHE = os.path.join(DEFAULT_CACHE, "gspc_monthly.parquet")

SEED = 267

# ---------------------------------------------------------------------------
# Hardcoded M2 YoY growth anchors — the study's deterministic offline core
# ---------------------------------------------------------------------------
# (year, month) -> M2 year-over-year growth in PERCENT.
# These anchors trace the canonical historical contour of US M2 YoY growth
# (FRED M2SL, seasonally adjusted). They are deliberately at a ~yearly cadence
# with extra anchors around the inflection points (the 2020-2023 episode), and
# linearly interpolated onto a monthly grid in ``m2_yoy_series``.
#
# Source contour: FRED "M2SL" YoY % change. As-of 2026-06-16.
M2_YOY_ANCHORS: list[tuple[int, int, float]] = [
    (1960, 1, 4.9),
    (1962, 1, 8.0),
    (1965, 1, 8.1),
    (1968, 1, 7.8),
    (1971, 1, 13.5),   # early-70s money boom
    (1973, 1, 9.0),
    (1975, 1, 4.5),    # post-oil-shock slowdown
    (1976, 1, 13.7),   # peak 1970s money growth
    (1978, 1, 8.2),
    (1980, 1, 8.5),
    (1981, 1, 10.0),
    (1983, 1, 11.5),
    (1985, 1, 8.6),
    (1987, 1, 4.0),
    (1989, 1, 4.9),
    (1991, 1, 3.1),
    (1993, 1, 0.8),    # early-90s near-zero money growth
    (1995, 1, 1.5),
    (1997, 1, 4.8),
    (1999, 1, 7.7),
    (2001, 1, 9.0),    # post-dotcom easing
    (2003, 1, 7.4),
    (2005, 1, 4.4),
    (2007, 1, 5.8),
    (2009, 1, 9.5),    # GFC QE
    (2010, 1, 1.9),    # money growth collapse after GFC
    (2012, 1, 9.6),
    (2014, 1, 6.0),
    (2016, 1, 6.2),
    (2018, 1, 4.0),
    (2019, 1, 4.0),
    (2020, 1, 6.7),
    (2020, 6, 23.0),   # COVID money explosion begins
    (2021, 2, 27.0),   # all-time-high M2 YoY ~27%
    (2021, 12, 13.0),
    (2022, 6, 5.9),
    (2022, 12, 0.0),
    (2023, 6, -3.9),   # first sustained NEGATIVE M2 YoY in the modern record
    (2023, 12, -2.3),
    (2024, 6, 1.5),
    (2024, 12, 3.9),
    (2025, 6, 4.5),
    (2025, 12, 4.3),
]


def m2_yoy_series(
    start: str = "1960-01-31",
    end: str = "2025-12-31",
) -> pd.Series:
    """Monthly M2 year-over-year growth (%) interpolated from the hardcoded anchors.

    Returns a ``pd.Series`` indexed by month-end dates (``DatetimeIndex``) with
    name ``m2_yoy`` and values in percent. The anchors are linearly interpolated
    onto the monthly grid; the series is the deterministic, offline core of the
    study.
    """
    idx = pd.date_range(start, end, freq="ME")
    anchor_dates = [
        pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0) for (y, m, _) in M2_YOY_ANCHORS
    ]
    anchor_vals = [v for (_, _, v) in M2_YOY_ANCHORS]
    anchor = pd.Series(anchor_vals, index=pd.DatetimeIndex(anchor_dates)).sort_index()
    # Reindex to the union, interpolate by time, then restrict to the grid.
    union = anchor.index.union(idx)
    interp = anchor.reindex(union).interpolate(method="time").reindex(idx)
    interp = interp.ffill().bfill()
    interp.name = "m2_yoy"
    interp.index.name = "date"
    return interp


# ---------------------------------------------------------------------------
# Synthetic panel — deterministic offline generator with a planted M2 premium
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_months: int = 360,
    premium: float = 0.0,
    base_ret: float = 0.006,
    ret_vol: float = 0.042,
    m2_persistence: float = 0.97,
    m2_vol: float = 0.6,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    """A monthly (m2_yoy, sp_ret) panel with an optional planted M2 -> return link.

    The M2-YoY series is a persistent AR(1) process (so it looks like real money
    growth: slow-moving, autocorrelated). The S&P monthly return is::

        sp_ret_t = base_ret + premium * (m2_yoy_{t-1} - mean) / 100 + noise_t

    where the lagged, de-meaned M2 reading drives the *next* month's return with
    coefficient ``premium``. ``premium = 0`` is the null: M2 carries no forward
    information and the only structure is the unconditional equity drift.

    Returns ``(df, truth)`` with columns ``m2_yoy`` (percent), ``sp_ret`` (simple
    monthly return), indexed by month-end dates, and ``truth`` recording the
    planted parameters.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1995-01-31", periods=n_months, freq="ME")

    # Persistent AR(1) money-growth series, centered near 6% with realistic vol.
    m2 = np.empty(n_months)
    m2[0] = 6.0
    shocks = rng.normal(0.0, m2_vol, n_months)
    for t in range(1, n_months):
        m2[t] = 6.0 + m2_persistence * (m2[t - 1] - 6.0) + shocks[t]
    m2_mean = m2.mean()

    noise = rng.normal(0.0, ret_vol, n_months)
    sp = np.full(n_months, base_ret)
    # lagged M2 drives next month's return
    sp[1:] += premium * (m2[:-1] - m2_mean) / 100.0
    sp += noise

    df = pd.DataFrame({"m2_yoy": m2, "sp_ret": sp}, index=idx)
    df.index.name = "date"
    truth = {
        "premium": premium,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "m2_persistence": m2_persistence,
        "m2_vol": m2_vol,
        "n_months": n_months,
        "m2_mean": float(m2_mean),
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — ^GSPC monthly closes (yfinance), cache-only by default
# ---------------------------------------------------------------------------
def fetch_gspc_monthly(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
    start: str = "1960-01-01",
) -> pd.Series:
    """Monthly ^GSPC (S&P 500 price index) close; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (lazy yfinance import);
    the result is then cached as a parquet under ``_cache/gspc_monthly.parquet``.
    Returns a ``pd.Series`` of month-end closes named ``gspc`` indexed by a
    month-end ``DatetimeIndex``.

    Raises ``FileNotFoundError`` when the cache is absent and ``fetch=False`` so
    callers can branch on ``HAVE_REAL``. The series is **price only** (no
    dividends) — labeled on the Signal axis.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached ^GSPC monthly series at {cache_path}. "
                "Call fetch_gspc_monthly(fetch=True) once to populate the cache."
            )
        s = pd.read_parquet(cache_path)["gspc"]
        s.index = pd.to_datetime(s.index)
        return s

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "^GSPC",
        start=start,
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no ^GSPC data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]
    close.index = pd.to_datetime(close.index) + pd.offsets.MonthEnd(0)
    close = close.sort_index().dropna()
    close.name = "gspc"
    close.index.name = "date"

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    close.to_frame().to_parquet(cache_path)
    print(f"Cached ^GSPC monthly: {len(close)} months -> {cache_path}")
    return close


def build_real_panel(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
) -> pd.DataFrame:
    """Join the hardcoded M2-YoY series with ^GSPC monthly returns.

    Returns a monthly DataFrame with columns:
      - ``m2_yoy``    : M2 year-over-year growth (%), from the hardcoded anchors
      - ``gspc``      : ^GSPC month-end close (price index, no dividends)
      - ``sp_ret``    : current-month ^GSPC simple return
      - ``sp_fwd1``   : NEXT-month ^GSPC return (the look-ahead-free target)
      - ``sp_fwd12``  : forward 12-month ^GSPC return (compounded)

    No look-ahead: ``sp_fwd1`` and ``sp_fwd12`` are returns earned *after* the
    month-t M2 reading is known. The merge is an inner join on month-end dates.
    Raises ``FileNotFoundError`` if the ^GSPC cache is absent and ``fetch=False``.
    """
    gspc = fetch_gspc_monthly(fetch=fetch, cache_path=cache_path)
    m2 = m2_yoy_series()

    df = pd.DataFrame({"gspc": gspc}).join(pd.DataFrame({"m2_yoy": m2}), how="inner")
    df = df.sort_index()
    df["sp_ret"] = df["gspc"].pct_change()
    df["sp_fwd1"] = df["sp_ret"].shift(-1)
    # forward 12-month compounded return
    fwd12 = (1.0 + df["sp_ret"]).shift(-1)
    for k in range(2, 13):
        fwd12 = fwd12 * (1.0 + df["sp_ret"].shift(-k))
    df["sp_fwd12"] = fwd12 - 1.0
    df.index.name = "date"
    return df


def fingerprint(obj) -> str:
    """A short content fingerprint of a Series/array of floats, for the as-of stamp."""
    if isinstance(obj, pd.DataFrame):
        arr = obj.select_dtypes("number").to_numpy(dtype=float)
    elif isinstance(obj, pd.Series):
        arr = obj.dropna().to_numpy(dtype=float)
    else:
        arr = np.asarray(obj, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

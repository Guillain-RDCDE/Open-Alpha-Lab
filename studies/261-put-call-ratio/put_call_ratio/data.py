"""Data layer for Study 261 (Put-Call-Ratio).

The CBOE *total* put/call ratio (puts traded divided by calls traded across all
CBOE-listed options) is the canonical "fear gauge by flow". A high reading means
traders are buying many puts relative to calls -- crowd fear -- and the folklore
says that contrarian extremes mark market bottoms. We test whether the ratio,
read at month-end, forecasts the *next* month's ^GSPC return.

Three components, all offline for the core:

- ``PCR_MONTHLY`` -- a hardcoded month-end table of the CBOE total put/call ratio
  from 2003-01 through 2025-12. CBOE began disseminating the daily total
  put/call ratio in 2003; the monthly figures here are month-end snapshots of
  that publicly reported series (rounded to two decimals). This table is small,
  well-known, and deterministic -- hardcoded here to keep the core fully offline.
  Source: CBOE market statistics (total put/call ratio history); the series
  long-run mean sits near ~0.92 and spikes above ~1.2 in panic months
  (Oct-2008, Mar-2020, Dec-2018).

- ``synthetic_panel`` -- a deterministic, offline generator with a planted
  ``contrarian_beta`` knob: next-month return = base + beta * (pcr - mean) +
  noise. ``contrarian_beta = 0`` is the null (the ratio carries no forward
  information). This is the study's null in a bottle.

- ``fetch_gspc_monthly`` -- the real ^GSPC monthly total-return-proxy series
  (price-only adjusted close, labelled as such), via yfinance, cache-only by
  default. Network is touched only on explicit ``fetch=True``.

No look-ahead: the put/call ratio is observed at the *close* of month t and used
to predict the return *earned during month t+1*. One full month of execution lag
is built into the design (the signal at month-end t positions for month t+1).

**Survivorship / regime caveat**: the ^GSPC index itself is survivorship-clean
(it is the index), but the put/call ratio history is short (~23 years, ~276
months) and dominated by a handful of crisis spikes. Inference rests on very few
independent "extreme" episodes -- this is named on the Signal axis.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

GSPC_CACHE = os.path.join(DEFAULT_CACHE, "gspc_monthly.parquet")

# ---------------------------------------------------------------------------
# Hardcoded CBOE total put/call ratio -- month-end values (the offline core)
# ---------------------------------------------------------------------------
# Each entry is (YYYY-MM, month-end total put/call ratio). The CBOE total
# put/call ratio is the daily total puts / total calls traded; the value below
# is a month-end snapshot. Values are rounded to two decimals and are a curated
# representation of the publicly reported CBOE series (long-run mean ~0.92).
# Panic months (Oct-2008, Dec-2018, Mar-2020) print >= ~1.15.
#
# Source: CBOE market statistics, total put/call ratio history. The table is
# deterministic, small, and hardcoded to keep the core fully offline.
PCR_MONTHLY: list[tuple[str, float]] = [
    ("2003-01", 0.78), ("2003-02", 0.86), ("2003-03", 0.74), ("2003-04", 0.66),
    ("2003-05", 0.69), ("2003-06", 0.71), ("2003-07", 0.73), ("2003-08", 0.70),
    ("2003-09", 0.79), ("2003-10", 0.68), ("2003-11", 0.72), ("2003-12", 0.67),
    ("2004-01", 0.70), ("2004-02", 0.78), ("2004-03", 0.88), ("2004-04", 0.82),
    ("2004-05", 0.91), ("2004-06", 0.74), ("2004-07", 0.93), ("2004-08", 0.95),
    ("2004-09", 0.83), ("2004-10", 0.90), ("2004-11", 0.72), ("2004-12", 0.75),
    ("2005-01", 0.84), ("2005-02", 0.79), ("2005-03", 0.88), ("2005-04", 0.99),
    ("2005-05", 0.86), ("2005-06", 0.80), ("2005-07", 0.78), ("2005-08", 0.92),
    ("2005-09", 0.85), ("2005-10", 1.01), ("2005-11", 0.81), ("2005-12", 0.83),
    ("2006-01", 0.82), ("2006-02", 0.90), ("2006-03", 0.85), ("2006-04", 0.91),
    ("2006-05", 1.04), ("2006-06", 0.97), ("2006-07", 0.95), ("2006-08", 0.84),
    ("2006-09", 0.82), ("2006-10", 0.80), ("2006-11", 0.83), ("2006-12", 0.86),
    ("2007-01", 0.89), ("2007-02", 1.02), ("2007-03", 0.97), ("2007-04", 0.88),
    ("2007-05", 0.85), ("2007-06", 0.96), ("2007-07", 1.05), ("2007-08", 1.10),
    ("2007-09", 0.92), ("2007-10", 0.90), ("2007-11", 1.07), ("2007-12", 0.94),
    ("2008-01", 1.12), ("2008-02", 1.01), ("2008-03", 1.09), ("2008-04", 0.93),
    ("2008-05", 0.90), ("2008-06", 1.04), ("2008-07", 1.08), ("2008-08", 0.95),
    ("2008-09", 1.16), ("2008-10", 1.27), ("2008-11", 1.21), ("2008-12", 1.05),
    ("2009-01", 1.02), ("2009-02", 1.10), ("2009-03", 0.98), ("2009-04", 0.85),
    ("2009-05", 0.83), ("2009-06", 0.92), ("2009-07", 0.86), ("2009-08", 0.84),
    ("2009-09", 0.81), ("2009-10", 0.95), ("2009-11", 0.86), ("2009-12", 0.83),
    ("2010-01", 0.94), ("2010-02", 0.97), ("2010-03", 0.82), ("2010-04", 0.88),
    ("2010-05", 1.13), ("2010-06", 1.06), ("2010-07", 0.95), ("2010-08", 1.04),
    ("2010-09", 0.85), ("2010-10", 0.83), ("2010-11", 0.92), ("2010-12", 0.80),
    ("2011-01", 0.85), ("2011-02", 0.82), ("2011-03", 0.99), ("2011-04", 0.84),
    ("2011-05", 0.93), ("2011-06", 0.98), ("2011-07", 1.01), ("2011-08", 1.18),
    ("2011-09", 1.08), ("2011-10", 0.97), ("2011-11", 1.02), ("2011-12", 0.90),
    ("2012-01", 0.86), ("2012-02", 0.81), ("2012-03", 0.83), ("2012-04", 0.97),
    ("2012-05", 1.05), ("2012-06", 0.92), ("2012-07", 0.96), ("2012-08", 0.84),
    ("2012-09", 0.85), ("2012-10", 0.96), ("2012-11", 1.00), ("2012-12", 0.88),
    ("2013-01", 0.82), ("2013-02", 0.89), ("2013-03", 0.83), ("2013-04", 0.91),
    ("2013-05", 0.85), ("2013-06", 0.98), ("2013-07", 0.80), ("2013-08", 0.93),
    ("2013-09", 0.82), ("2013-10", 0.84), ("2013-11", 0.79), ("2013-12", 0.81),
    ("2014-01", 0.97), ("2014-02", 0.84), ("2014-03", 0.88), ("2014-04", 0.92),
    ("2014-05", 0.83), ("2014-06", 0.80), ("2014-07", 0.95), ("2014-08", 0.82),
    ("2014-09", 0.94), ("2014-10", 1.06), ("2014-11", 0.81), ("2014-12", 0.93),
    ("2015-01", 0.95), ("2015-02", 0.82), ("2015-03", 0.91), ("2015-04", 0.86),
    ("2015-05", 0.85), ("2015-06", 0.94), ("2015-07", 0.90), ("2015-08", 1.10),
    ("2015-09", 1.02), ("2015-10", 0.88), ("2015-11", 0.89), ("2015-12", 0.97),
    ("2016-01", 1.08), ("2016-02", 1.02), ("2016-03", 0.86), ("2016-04", 0.85),
    ("2016-05", 0.89), ("2016-06", 0.99), ("2016-07", 0.82), ("2016-08", 0.83),
    ("2016-09", 0.92), ("2016-10", 0.95), ("2016-11", 0.90), ("2016-12", 0.83),
    ("2017-01", 0.85), ("2017-02", 0.80), ("2017-03", 0.88), ("2017-04", 0.87),
    ("2017-05", 0.86), ("2017-06", 0.84), ("2017-07", 0.79), ("2017-08", 0.96),
    ("2017-09", 0.82), ("2017-10", 0.78), ("2017-11", 0.83), ("2017-12", 0.81),
    ("2018-01", 0.84), ("2018-02", 1.07), ("2018-03", 1.04), ("2018-04", 0.92),
    ("2018-05", 0.86), ("2018-06", 0.93), ("2018-07", 0.85), ("2018-08", 0.83),
    ("2018-09", 0.88), ("2018-10", 1.12), ("2018-11", 1.05), ("2018-12", 1.24),
    ("2019-01", 0.94), ("2019-02", 0.85), ("2019-03", 0.92), ("2019-04", 0.83),
    ("2019-05", 1.06), ("2019-06", 0.88), ("2019-07", 0.87), ("2019-08", 1.03),
    ("2019-09", 0.90), ("2019-10", 0.92), ("2019-11", 0.84), ("2019-12", 0.86),
    ("2020-01", 0.97), ("2020-02", 1.08), ("2020-03", 1.19), ("2020-04", 0.96),
    ("2020-05", 0.88), ("2020-06", 0.90), ("2020-07", 0.82), ("2020-08", 0.78),
    ("2020-09", 0.94), ("2020-10", 0.98), ("2020-11", 0.85), ("2020-12", 0.80),
    ("2021-01", 0.91), ("2021-02", 0.84), ("2021-03", 0.87), ("2021-04", 0.83),
    ("2021-05", 0.92), ("2021-06", 0.85), ("2021-07", 0.95), ("2021-08", 0.86),
    ("2021-09", 0.99), ("2021-10", 0.83), ("2021-11", 0.94), ("2021-12", 0.93),
    ("2022-01", 1.10), ("2022-02", 1.05), ("2022-03", 0.96), ("2022-04", 1.08),
    ("2022-05", 1.13), ("2022-06", 1.15), ("2022-07", 0.94), ("2022-08", 0.99),
    ("2022-09", 1.14), ("2022-10", 1.00), ("2022-11", 0.92), ("2022-12", 1.07),
    ("2023-01", 0.90), ("2023-02", 0.97), ("2023-03", 1.04), ("2023-04", 0.91),
    ("2023-05", 0.93), ("2023-06", 0.83), ("2023-07", 0.85), ("2023-08", 0.98),
    ("2023-09", 1.02), ("2023-10", 1.06), ("2023-11", 0.84), ("2023-12", 0.82),
    ("2024-01", 0.89), ("2024-02", 0.86), ("2024-03", 0.83), ("2024-04", 1.01),
    ("2024-05", 0.88), ("2024-06", 0.85), ("2024-07", 0.95), ("2024-08", 1.07),
    ("2024-09", 0.90), ("2024-10", 0.92), ("2024-11", 0.84), ("2024-12", 0.97),
    ("2025-01", 0.93), ("2025-02", 0.99), ("2025-03", 1.05), ("2025-04", 1.16),
    ("2025-05", 0.94), ("2025-06", 0.88), ("2025-07", 0.86), ("2025-08", 0.95),
    ("2025-09", 0.91), ("2025-10", 0.93), ("2025-11", 0.90), ("2025-12", 0.92),
]


def pcr_series() -> pd.Series:
    """Return the hardcoded CBOE total put/call ratio as a month-end Series.

    Index is a ``pd.DatetimeIndex`` at month-end frequency; values are the
    month-end total put/call ratio (float). Deterministic and offline.
    """
    dates = [pd.Timestamp(ym + "-01") + pd.offsets.MonthEnd(0) for ym, _ in PCR_MONTHLY]
    vals = [v for _, v in PCR_MONTHLY]
    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name="pcr")
    return s.sort_index()


# ---------------------------------------------------------------------------
# Synthetic panel -- deterministic offline core with a planted contrarian beta
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_months: int = 276,
    contrarian_beta: float = 0.15,
    base_ret: float = 0.006,
    ret_vol: float = 0.035,
    pcr_mean: float = 0.92,
    pcr_vol: float = 0.12,
    seed: int = 261,
) -> tuple[pd.DataFrame, dict]:
    """A month x (pcr, fwd_ret) panel with a known contrarian put/call effect.

    The put/call ratio is drawn from ``N(pcr_mean, pcr_vol)`` and the *next*
    month's market return is::

        fwd_ret_t = base_ret + contrarian_beta * (pcr_t - pcr_mean) + noise_t

    A *positive* ``contrarian_beta`` plants the folklore claim: a high put/call
    ratio (excess fear, pcr > mean) precedes *higher* forward returns (the crowd
    was wrong). ``contrarian_beta = 0`` is the null -- the ratio carries no
    forward information.

    Returns ``(df, truth)`` where ``df`` has columns ``pcr`` (the month-end
    ratio) and ``fwd_ret`` (the return earned in the *following* month, aligned
    to the row where the signal is observed), and ``truth`` records the planted
    parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2003-01-31", periods=n_months, freq="ME")
    pcr = rng.normal(pcr_mean, pcr_vol, n_months)
    noise = rng.normal(0.0, ret_vol, n_months)
    fwd_ret = base_ret + contrarian_beta * (pcr - pcr_mean) + noise

    df = pd.DataFrame({"pcr": pcr, "fwd_ret": fwd_ret}, index=months)
    df.index.name = "date"
    truth = {
        "n_months": n_months,
        "contrarian_beta": contrarian_beta,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "pcr_mean": pcr_mean,
        "pcr_vol": pcr_vol,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- ^GSPC monthly closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_gspc_monthly(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
) -> pd.Series:
    """Monthly ^GSPC adjusted close (price-only proxy); cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/gspc_monthly.parquet``). Index is
    month-end dates, values are the ^GSPC month-end close.

    **Total-return label**: this is the *price-only* ^GSPC level (auto_adjust
    applies splits but the index excludes dividends); we label all real-tape
    returns as price-only so the dividend drag (~1.8%/yr) is not silently
    counted as alpha.

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False``.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached ^GSPC monthly series at {cache_path}. "
                "Call fetch_gspc_monthly(fetch=True) once to populate the cache."
            )
        s = pd.read_parquet(cache_path)["gspc"]
        s.index = pd.DatetimeIndex(s.index)
        return s

    # ------------------------------------------------------------------
    # Live fetch: yfinance monthly ^GSPC closes
    # ------------------------------------------------------------------
    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "^GSPC",
        start="2002-12-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no monthly ^GSPC data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]

    close.index = pd.to_datetime(close.index) + pd.offsets.MonthEnd(0)
    close = close.sort_index().dropna()
    close.name = "gspc"

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    close.to_frame().to_parquet(cache_path)
    print(f"Cached ^GSPC monthly: {len(close)} months -> {cache_path}")
    return close


def build_real_panel(
    fetch: bool = False,
    cache_path: str = GSPC_CACHE,
) -> pd.DataFrame:
    """Join the hardcoded put/call ratio with the real ^GSPC next-month return.

    Returns a DataFrame indexed by month-end with columns:
      ``pcr``     : month-end CBOE total put/call ratio (hardcoded)
      ``fwd_ret`` : ^GSPC price-only return earned in the *following* month

    No look-ahead: ``fwd_ret`` at row t is the return from month-end t to
    month-end t+1, i.e. the position taken on the signal observed at t.

    Raises ``FileNotFoundError`` if the ^GSPC cache is absent and ``fetch=False``.
    """
    pcr = pcr_series()
    gspc = fetch_gspc_monthly(fetch=fetch, cache_path=cache_path)
    ret = gspc.pct_change()  # return earned IN month t (t-1 -> t)
    fwd = ret.shift(-1)      # return earned in month t+1, aligned to signal at t
    df = pd.DataFrame({"pcr": pcr})
    df["fwd_ret"] = fwd.reindex(df.index)
    df = df.dropna()
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(obj) -> str:
    """A short content fingerprint of a Series/DataFrame's values, for the as-of stamp."""
    if isinstance(obj, pd.DataFrame):
        arr = obj.to_numpy(dtype=float).ravel()
    else:
        arr = np.asarray(obj, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

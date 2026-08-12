"""Data layer for Study 848 — "Costco Hot-Dog Index" (COST vs CPI vs SPY).

Three ingredients, all offline-friendly once cached.

* **Real tape — COST, SPY, XLP (yfinance).** Month-end total-return (``auto_adjust=True``)
  closes for **Costco (`COST`)**, the market (**`SPY`**) and the consumer-staples sector
  (**`XLP`**, the "does pricing power beat a boring staples basket?" control). ``fetch()``
  pulls them into this study's OWN ``_cache/`` as a single parquet (retry up to 4×);
  ``have_real()`` / ``load_market()`` read the cache OFFLINE (no yfinance import). The
  in-progress month is dropped so the last bar is complete. ``AS_OF="2026-06-30"``.

* **CPI — real ``CPIAUCSL``, fetched from the BLS public API and cached.**
  ``CPIAUCSL`` (CPI-U, all items, U.S. city average, seasonally adjusted, 1982-84=100) is
  the FRED alias for BLS series ``CUSR0000SA0``. ``fetch_cpi()`` pulls the real monthly
  levels from the BLS public API (``api.bls.gov``, no key) in ≤10-year chunks and caches
  them under ``_cache/`` as parquet; ``load_cpi()`` reads that cache OFFLINE. FRED's own
  ``fredgraph.csv`` endpoint is unreachable from some build hosts, so BLS is the primary
  feed. As a last-resort offline fallback (and to keep the machinery tests network-free) we
  also embed the **exact same public-record values** (2000-01 → 2026-06) in ``_CPI`` — real
  BLS observations, not an approximation — plus the 1985 launch anchor (``CPI_1985 = 107.6``,
  the CPIAUCSL 1985 annual average the $1.50 combo was priced against). One point, **2025-10**,
  was not published on the SA series (the 2025 federal data-collection disruption) and is
  linearly interpolated between the published Sep/Nov levels — the single interpolated point,
  disclosed in ``docs/references.md``.

* **Synthetic positive control.** A deterministic fixed-seed monthly world with a TUNABLE
  planted inflation-surprise beta (``edge``): COST's return loads ``edge`` on the inflation
  surprise. ``edge = 0`` is the null. Proves the inference engine recovers a real
  inflation beta when one exists. No network.

Pure numpy/pandas + stdlib; the offline core never touches the network.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
CACHE_PATH = os.path.join(CACHE_DIR, "cost_spy_xlp_monthly.parquet")
CPI_CACHE_PATH = os.path.join(CACHE_DIR, "cpi_cpiaucsl.parquet")

# BLS public API (no key): CUSR0000SA0 = CPI-U all items, U.S. city average, SA (= CPIAUCSL)
BLS_SERIES = "CUSR0000SA0"
BLS_API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

COST = "COST"          # Costco Wholesale — the hot-dog company
BENCH = "SPY"          # the market
STAPLES = "XLP"        # Consumer Staples Select Sector SPDR — the "boring staples" control
TICKERS = [COST, BENCH, STAPLES]

START = "1993-01-01"   # SPY inception; COST lists earlier, XLP from 1998 (join binds later)
AS_OF = "2026-06-30"   # last complete calendar month at publication
PANEL_START = "2000-01-31"  # CPI series starts 2000-01; the stamped join begins here

# The frozen combo price and its 1985 anchor (public record — see docs/references.md).
COMBO_PRICE = 1.50     # $1.50 hot-dog + 20oz soda, nominally unchanged since ~1985
COMBO_YEAR = 1985
CPI_1985 = 107.6       # CPIAUCSL 1985 annual average (1982-84 = 100)

__all__ = [
    "COST", "BENCH", "STAPLES", "TICKERS", "START", "AS_OF", "PANEL_START",
    "COMBO_PRICE", "COMBO_YEAR", "CPI_1985", "CACHE_DIR", "CACHE_PATH", "CPI_CACHE_PATH",
    "fetch", "fetch_cpi", "have_real", "have_cpi", "load_market", "load_cpi",
    "build_panel", "synthetic_world", "fingerprint",
]

# --------------------------------------------------------------------------- #
# CPI — REAL CPIAUCSL values (BLS CUSR0000SA0), embedded as the offline fallback
# --------------------------------------------------------------------------- #
# CPI-U, all items, U.S. city average, seasonally adjusted, index 1982-84 = 100
# (FRED ``CPIAUCSL`` = BLS ``CUSR0000SA0``). These are the REAL published monthly levels
# fetched from the BLS public API (2000-01 → 2026-06) — public-record observations, not an
# approximation — embedded so the offline machinery tests never need the network. The one
# exception is 2025-10, not published on the SA series (2025 federal data-collection
# disruption), linearly interpolated between the published Sep/Nov levels (see references).
_CPI: dict[str, float] = {
    "2000-01": 169.3, "2000-02": 170, "2000-03": 171, "2000-04": 170.9,
    "2000-05": 171.2, "2000-06": 172.2, "2000-07": 172.7, "2000-08": 172.7,
    "2000-09": 173.6, "2000-10": 173.9, "2000-11": 174.2, "2000-12": 174.6,
    "2001-01": 175.6, "2001-02": 176, "2001-03": 176.1, "2001-04": 176.4,
    "2001-05": 177.3, "2001-06": 177.7, "2001-07": 177.4, "2001-08": 177.4,
    "2001-09": 178.1, "2001-10": 177.6, "2001-11": 177.5, "2001-12": 177.4,
    "2002-01": 177.7, "2002-02": 178, "2002-03": 178.5, "2002-04": 179.3,
    "2002-05": 179.5, "2002-06": 179.6, "2002-07": 180, "2002-08": 180.5,
    "2002-09": 180.8, "2002-10": 181.2, "2002-11": 181.5, "2002-12": 181.8,
    "2003-01": 182.6, "2003-02": 183.6, "2003-03": 183.9, "2003-04": 183.2,
    "2003-05": 182.9, "2003-06": 183.1, "2003-07": 183.7, "2003-08": 184.5,
    "2003-09": 185.1, "2003-10": 184.9, "2003-11": 185, "2003-12": 185.5,
    "2004-01": 186.3, "2004-02": 186.7, "2004-03": 187.1, "2004-04": 187.4,
    "2004-05": 188.2, "2004-06": 188.9, "2004-07": 189.1, "2004-08": 189.2,
    "2004-09": 189.8, "2004-10": 190.8, "2004-11": 191.7, "2004-12": 191.7,
    "2005-01": 191.6, "2005-02": 192.4, "2005-03": 193.1, "2005-04": 193.7,
    "2005-05": 193.6, "2005-06": 193.7, "2005-07": 194.9, "2005-08": 196.1,
    "2005-09": 198.8, "2005-10": 199.1, "2005-11": 198.1, "2005-12": 198.1,
    "2006-01": 199.3, "2006-02": 199.4, "2006-03": 199.7, "2006-04": 200.7,
    "2006-05": 201.3, "2006-06": 201.8, "2006-07": 202.9, "2006-08": 203.8,
    "2006-09": 202.8, "2006-10": 201.9, "2006-11": 202, "2006-12": 203.1,
    "2007-01": 203.437, "2007-02": 204.226, "2007-03": 205.288, "2007-04": 205.904,
    "2007-05": 206.755, "2007-06": 207.234, "2007-07": 207.603, "2007-08": 207.667,
    "2007-09": 208.547, "2007-10": 209.19, "2007-11": 210.834, "2007-12": 211.445,
    "2008-01": 212.174, "2008-02": 212.687, "2008-03": 213.448, "2008-04": 213.942,
    "2008-05": 215.208, "2008-06": 217.463, "2008-07": 219.016, "2008-08": 218.69,
    "2008-09": 218.877, "2008-10": 216.995, "2008-11": 213.153, "2008-12": 211.398,
    "2009-01": 211.933, "2009-02": 212.705, "2009-03": 212.495, "2009-04": 212.709,
    "2009-05": 213.022, "2009-06": 214.79, "2009-07": 214.726, "2009-08": 215.445,
    "2009-09": 215.861, "2009-10": 216.509, "2009-11": 217.234, "2009-12": 217.347,
    "2010-01": 217.488, "2010-02": 217.281, "2010-03": 217.353, "2010-04": 217.403,
    "2010-05": 217.29, "2010-06": 217.199, "2010-07": 217.605, "2010-08": 217.923,
    "2010-09": 218.275, "2010-10": 219.035, "2010-11": 219.59, "2010-12": 220.472,
    "2011-01": 221.187, "2011-02": 221.898, "2011-03": 223.046, "2011-04": 224.093,
    "2011-05": 224.806, "2011-06": 224.806, "2011-07": 225.395, "2011-08": 226.106,
    "2011-09": 226.597, "2011-10": 226.75, "2011-11": 227.169, "2011-12": 227.223,
    "2012-01": 227.842, "2012-02": 228.329, "2012-03": 228.807, "2012-04": 229.187,
    "2012-05": 228.713, "2012-06": 228.524, "2012-07": 228.59, "2012-08": 229.918,
    "2012-09": 231.015, "2012-10": 231.638, "2012-11": 231.249, "2012-12": 231.221,
    "2013-01": 231.679, "2013-02": 232.937, "2013-03": 232.282, "2013-04": 231.797,
    "2013-05": 231.893, "2013-06": 232.445, "2013-07": 232.9, "2013-08": 233.456,
    "2013-09": 233.544, "2013-10": 233.669, "2013-11": 234.1, "2013-12": 234.719,
    "2014-01": 235.288, "2014-02": 235.547, "2014-03": 236.028, "2014-04": 236.468,
    "2014-05": 236.918, "2014-06": 237.231, "2014-07": 237.498, "2014-08": 237.46,
    "2014-09": 237.477, "2014-10": 237.43, "2014-11": 236.983, "2014-12": 236.252,
    "2015-01": 234.747, "2015-02": 235.342, "2015-03": 235.976, "2015-04": 236.222,
    "2015-05": 237.001, "2015-06": 237.657, "2015-07": 238.034, "2015-08": 238.033,
    "2015-09": 237.498, "2015-10": 237.733, "2015-11": 238.017, "2015-12": 237.761,
    "2016-01": 237.652, "2016-02": 237.336, "2016-03": 238.08, "2016-04": 238.992,
    "2016-05": 239.557, "2016-06": 240.222, "2016-07": 240.101, "2016-08": 240.545,
    "2016-09": 241.176, "2016-10": 241.741, "2016-11": 242.026, "2016-12": 242.637,
    "2017-01": 243.618, "2017-02": 244.006, "2017-03": 243.892, "2017-04": 244.193,
    "2017-05": 244.004, "2017-06": 244.163, "2017-07": 244.243, "2017-08": 245.183,
    "2017-09": 246.435, "2017-10": 246.626, "2017-11": 247.284, "2017-12": 247.805,
    "2018-01": 248.859, "2018-02": 249.529, "2018-03": 249.577, "2018-04": 250.227,
    "2018-05": 250.792, "2018-06": 251.018, "2018-07": 251.214, "2018-08": 251.663,
    "2018-09": 252.182, "2018-10": 252.772, "2018-11": 252.594, "2018-12": 252.767,
    "2019-01": 252.561, "2019-02": 253.319, "2019-03": 254.277, "2019-04": 255.233,
    "2019-05": 255.296, "2019-06": 255.213, "2019-07": 255.802, "2019-08": 256.036,
    "2019-09": 256.43, "2019-10": 257.155, "2019-11": 257.879, "2019-12": 258.63,
    "2020-01": 259.127, "2020-02": 259.25, "2020-03": 258.076, "2020-04": 256.032,
    "2020-05": 255.802, "2020-06": 257.042, "2020-07": 258.352, "2020-08": 259.316,
    "2020-09": 259.997, "2020-10": 260.319, "2020-11": 260.911, "2020-12": 262.045,
    "2021-01": 262.687, "2021-02": 263.579, "2021-03": 264.961, "2021-04": 266.614,
    "2021-05": 268.383, "2021-06": 270.654, "2021-07": 271.903, "2021-08": 272.676,
    "2021-09": 273.91, "2021-10": 276.55, "2021-11": 278.919, "2021-12": 280.845,
    "2022-01": 282.543, "2022-02": 284.5, "2022-03": 287.674, "2022-04": 288.561,
    "2022-05": 291.298, "2022-06": 294.957, "2022-07": 294.913, "2022-08": 295.097,
    "2022-09": 296.349, "2022-10": 298.007, "2022-11": 298.786, "2022-12": 298.832,
    "2023-01": 300.42, "2023-02": 301.45, "2023-03": 301.821, "2023-04": 302.845,
    "2023-05": 303.334, "2023-06": 304.014, "2023-07": 304.609, "2023-08": 306.082,
    "2023-09": 307.276, "2023-10": 307.696, "2023-11": 308.148, "2023-12": 308.741,
    "2024-01": 309.698, "2024-02": 310.967, "2024-03": 312.345, "2024-04": 313.023,
    "2024-05": 313.175, "2024-06": 313.044, "2024-07": 313.569, "2024-08": 314.062,
    "2024-09": 314.732, "2024-10": 315.631, "2024-11": 316.528, "2024-12": 317.604,
    "2025-01": 318.961, "2025-02": 319.679, "2025-03": 319.785, "2025-04": 320.302,
    "2025-05": 320.62, "2025-06": 321.435, "2025-07": 322.169, "2025-08": 323.291,
    "2025-09": 324.245, "2025-10": 324.654, "2025-11": 325.063, "2025-12": 326.031,
    "2026-01": 326.588, "2026-02": 327.46, "2026-03": 330.293, "2026-04": 332.407,
    "2026-05": 333.979, "2026-06": 332.568,
}


def _cpi_from_dict() -> pd.Series:
    """The embedded real CPIAUCSL values as a month-end series (offline fallback)."""
    idx = pd.to_datetime([f"{k}-01" for k in _CPI]) + pd.offsets.MonthEnd(0)
    return pd.Series(list(_CPI.values()), index=idx, name="cpi").sort_index()


def have_cpi() -> bool:
    return os.path.exists(CPI_CACHE_PATH)


def fetch_cpi(retries: int = 4) -> pd.Series:
    """Download real CPIAUCSL (BLS ``CUSR0000SA0``) from the BLS public API; cache parquet.

    Network; runs once to build the cache. The BLS v2 public endpoint (no key) serves ≤10
    years per request, so we pull three overlapping windows (2000-2009, 2010-2019,
    2017-2026) and stitch them. Retries up to ``retries`` times per window. Writes a
    month-end ``pd.Series`` (index ``date``, name ``cpi``) to ``CPI_CACHE_PATH``. Any
    interior gap (e.g. the 2025-10 SA suspension) is linearly interpolated on the monthly
    grid so downstream YoY differencing sees a contiguous series.
    """
    import json
    import urllib.request

    def _one(sy: int, ey: int) -> dict[str, float]:
        body = json.dumps({"seriesid": [BLS_SERIES], "startyear": str(sy),
                           "endyear": str(ey)}).encode()
        last: Exception | None = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    BLS_API, data=body,
                    headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
                j = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
                out: dict[str, float] = {}
                for x in j["Results"]["series"][0]["data"]:
                    if not x["period"].startswith("M"):
                        continue
                    v = x["value"].replace(",", "").strip()
                    try:
                        out[f"{x['year']}-{x['period'][1:]}"] = float(v)
                    except ValueError:
                        continue
                return out
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt < retries - 1:
                    time.sleep(2.0 + 2.0 * attempt)
        raise RuntimeError(f"BLS fetch {sy}-{ey} failed after {retries} attempts: {last}")

    merged: dict[str, float] = {}
    for sy, ey in [(2000, 2009), (2010, 2019), (2017, 2026)]:
        merged.update(_one(sy, ey))
    keys = sorted(merged)
    idx = pd.to_datetime([f"{k}-01" for k in keys]) + pd.offsets.MonthEnd(0)
    s = pd.Series([merged[k] for k in keys], index=idx, name="cpi").sort_index()
    # fill any interior monthly gap (e.g. 2025-10 SA suspension) by time interpolation
    full = pd.date_range(s.index.min(), s.index.max(), freq="ME")
    s = s.reindex(full).interpolate(method="linear").rename("cpi")
    s.index.name = "date"
    if len(s) < 200:
        raise RuntimeError(f"suspiciously short CPI series ({len(s)} rows)")
    os.makedirs(CACHE_DIR, exist_ok=True)
    s.to_frame().to_parquet(CPI_CACHE_PATH)
    return s


def load_cpi(asof: str = AS_OF) -> pd.Series:
    """Month-end CPIAUCSL levels, sliced to ``<= asof``. OFFLINE.

    Prefers the real BLS cache under ``_cache/`` if present; otherwise falls back to the
    embedded real public-record values (``_CPI``). Either way the numbers are the real
    published ``CPIAUCSL`` levels (2025-10 interpolated — the SA suspension).
    """
    if have_cpi():
        s = pd.read_parquet(CPI_CACHE_PATH)["cpi"]
        s = s.sort_index()
    else:
        s = _cpi_from_dict()
    if asof is not None:
        s = s[s.index <= pd.Timestamp(asof)]
    return s


# --------------------------------------------------------------------------- #
# Real tape — yfinance month-end closes, cache-first
# --------------------------------------------------------------------------- #
def fetch(start: str = START, end: str = "2026-07-01", retries: int = 4,
          with_cpi: bool = True) -> pd.DataFrame:
    """Download COST + SPY + XLP month-end total-return closes (and real CPI); cache _cache/.

    Network; runs once to build the cache. Retries up to ``retries`` times with a short
    sleep on transient failure. Writes a single parquet with tz-naive month-end index and
    columns ``COST``, ``SPY``, ``XLP``. The in-progress month is dropped. With
    ``with_cpi=True`` it also pulls real CPIAUCSL via :func:`fetch_cpi` (best-effort — a CPI
    failure leaves the embedded real values as the offline fallback).
    """
    import yfinance as yf  # lazy — never imported on the offline path

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, interval="1mo",
                              auto_adjust=True, progress=False, threads=False)
            if raw is None or raw.empty:
                raise RuntimeError("yfinance returned no bars")
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            close = close.copy()
            close.index = pd.DatetimeIndex(close.index).tz_localize(None) + pd.offsets.MonthEnd(0)
            close = close[~close.index.duplicated(keep="last")]
            last_complete = pd.Timestamp.today().to_period("M") - 1
            close = close[close.index.to_period("M") <= last_complete]
            out = close[TICKERS].dropna(how="all")
            out.index.name = "date"
            if len(out) < 200:
                raise RuntimeError(f"suspiciously short tape ({len(out)} rows)")
            os.makedirs(CACHE_DIR, exist_ok=True)
            out.to_parquet(CACHE_PATH)
            if with_cpi:
                try:
                    fetch_cpi(retries=retries)
                except Exception:  # noqa: BLE001 — embedded real values remain the fallback
                    pass
            return out
        except Exception as exc:  # noqa: BLE001 — retry on any transient failure
            last_err = exc
            if attempt < retries - 1:
                time.sleep(2.0 + 2.0 * attempt)
    raise RuntimeError(f"yfinance fetch failed after {retries} attempts: {last_err}")


def have_real() -> bool:
    return os.path.exists(CACHE_PATH)


def load_market(asof: str = AS_OF) -> pd.DataFrame:
    """Cached month-end COST/SPY/XLP closes, sliced to ``<= asof``. OFFLINE (reads parquet)."""
    if not have_real():
        raise FileNotFoundError(
            f"No cached tape at {CACHE_PATH}. Call hotdog_index.data.fetch() once."
        )
    df = pd.read_parquet(CACHE_PATH)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[df.index <= pd.Timestamp(asof)]
    return df.sort_index()


# --------------------------------------------------------------------------- #
# The merged monthly panel
# --------------------------------------------------------------------------- #
def build_panel(asof: str = AS_OF) -> pd.DataFrame:
    """Merge COST/SPY/XLP (real tape) with the real CPI series into one monthly panel.

    Columns: ``cost``/``spy``/``xlp`` (total-return levels), ``cpi`` (index level), the
    monthly log returns ``cost_ret``/``spy_ret``/``xlp_ret``, monthly log inflation
    ``infl``, trailing 12-month YoY inflation ``yoy``, and the **inflation surprise**
    ``surprise`` = ΔYoY (the month-on-month change in YoY inflation — the "is inflation
    accelerating?" innovation). Sliced
    to ``asof`` (never a partial/future month). Returns an **empty** frame if the market
    cache is missing (so offline callers fall back to frozen numbers).
    """
    if not have_real():
        return pd.DataFrame()
    mkt = load_market(asof)
    cpi = load_cpi(asof)
    df = pd.DataFrame({"cost": mkt[COST], "spy": mkt[BENCH], "xlp": mkt[STAPLES],
                       "cpi": cpi})
    df = df[df.index >= pd.Timestamp(PANEL_START)].dropna()
    df["cost_ret"] = np.log(df["cost"]).diff()
    df["spy_ret"] = np.log(df["spy"]).diff()
    df["xlp_ret"] = np.log(df["xlp"]).diff()
    df["infl"] = np.log(df["cpi"]).diff()
    df["yoy"] = df["cpi"] / df["cpi"].shift(12) - 1.0
    df["surprise"] = df["yoy"].diff()          # ΔYoY inflation = the inflation surprise
    return df


# --------------------------------------------------------------------------- #
# Synthetic positive control (deterministic, fixed seed, no network)
# --------------------------------------------------------------------------- #
def synthetic_world(edge: float = 0.0, seed: int = 848, n_months: int = 312,
                    cost_vol: float = 0.06, spy_vol: float = 0.045,
                    xlp_vol: float = 0.035, infl_base: float = 0.0018,
                    infl_rho: float = 0.6, infl_sd: float = 0.0016) -> pd.DataFrame:
    """A monthly world with a TUNABLE planted inflation-surprise beta.

    A persistent monthly inflation ``infl`` (AR(1) around ``infl_base``) drives a CPI
    level; the inflation *surprise* is its ΔYoY innovation. COST's return loads
    ``edge`` on that surprise (``cost_ret = edge·surprise + noise``); SPY/XLP do not. With
    ``edge > 0`` the inflation-beta regression *should* recover a significant positive
    slope for COST (and the COST-vs-XLP beta gap should light up) — the machinery proof.
    ``edge = 0`` is the null. Deterministic given ``seed``; same columns as
    :func:`build_panel`. A ``PeriodIndex`` (monthly) is kept — never ``.to_timestamp()`` —
    to stay clear of the pandas ns-Timestamp overflow horizon. No network.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000-01", periods=n_months, freq="M")

    infl = np.empty(n_months)
    infl[0] = infl_base
    for t in range(1, n_months):
        infl[t] = infl_base + infl_rho * (infl[t - 1] - infl_base) + rng.normal(0.0, infl_sd)
    infl = np.maximum(infl, -0.004)
    cpi = 170.0 * np.exp(np.cumsum(infl))

    yoy = np.full(n_months, np.nan)
    yoy[12:] = cpi[12:] / cpi[:-12] - 1.0
    surprise = np.concatenate([[np.nan], np.diff(yoy)])   # ΔYoY (NaN for first 13)
    surp_filled = np.nan_to_num(surprise, nan=0.0)

    cost_ret = 0.010 + edge * surp_filled + cost_vol * rng.standard_normal(n_months)
    spy_ret = 0.006 + spy_vol * rng.standard_normal(n_months)
    xlp_ret = 0.005 + xlp_vol * rng.standard_normal(n_months)

    cost = 40.0 * np.exp(np.cumsum(cost_ret))
    spy = 150.0 * np.exp(np.cumsum(spy_ret))
    xlp = 25.0 * np.exp(np.cumsum(xlp_ret))

    df = pd.DataFrame({"cost": cost, "spy": spy, "xlp": xlp, "cpi": cpi,
                       "cost_ret": cost_ret, "spy_ret": spy_ret, "xlp_ret": xlp_ret,
                       "infl": infl, "yoy": yoy, "surprise": surprise}, index=idx)
    return df


# --------------------------------------------------------------------------- #
# Content fingerprint (for the as-of stamp)
# --------------------------------------------------------------------------- #
def fingerprint(df: pd.DataFrame) -> str:
    """Short content hash of the COST/SPY/CPI columns (the as-of stamp)."""
    cols = [c for c in ("cost", "spy", "cpi") if c in df.columns]
    mat = np.ascontiguousarray(df[cols].fillna(0.0).to_numpy(dtype=float))
    return hashlib.sha1(mat.tobytes()).hexdigest()[:12]

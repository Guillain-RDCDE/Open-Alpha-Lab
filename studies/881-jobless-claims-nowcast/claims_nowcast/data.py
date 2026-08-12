"""Data layer for Study 881 — Jobless-Claims Sector Rotation.

Three ingredients; the claims tape and the synthetic control are fully offline.

* **Real claims tape (documented public snapshot).** ``ICSA_4WK_BY_YEAR`` is a monthly
  snapshot of U.S. **initial unemployment claims, seasonally adjusted, 4-week moving
  average** (thousands) — FRED series ``IC4WSA`` (source: U.S. Dept. of Labor / ETA).
  The 4-week MA is the canonical noise-smoothed claims gauge, and its month-on-month
  change is the "4-week change in initial claims" this study rotates on. FRED's CSV
  endpoint (``fred.stlouisfed.org``) is **DNS-unreachable in this build**, so — exactly
  as Study 385 (jobless-claims-momentum) and Study 268 (Sahm-rule) hardcode their FRED
  pulls — we encode a public, never-revised-in-this-snapshot monthly table with the
  source cited. ``fetch()`` still *attempts* the live FRED CSV (retry + DBnomics
  fallback) and documents the firewall on failure.

* **Real sector tape (fetched & cached).** Daily total-return closes for the four
  rotation ETFs — **XLY, XLI** (cyclicals) and **XLP, XLU** (defensives) — plus **SPY**,
  pulled with yfinance (``auto_adjust=True``) and cached as a parquet under this study's
  own ``_cache/``. ``fetch_etfs()`` (network) rebuilds the cache; ``load_etfs()`` reads
  the parquet directly, OFFLINE, and is what the notebooks' offline cells use.

* **Synthetic control (the machinery proof).** :func:`synthetic_frame` is a
  deterministic, fixed-seed generator producing a monthly claims path and four
  sector-ETF price paths with a *planted* link: when claims rise, next month's
  **cyclical** returns are knocked down by a tunable ``edge`` (so the
  cyclical-minus-defensive spread loads negatively on the claims change — the claim,
  by construction). ``edge = 0`` is the null (claims carry no rotation information) and
  must NOT manufacture significance.

Pure numpy + pandas + stdlib for the offline path; yfinance is imported lazily only
inside ``fetch_etfs``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
ETF_CACHE = os.path.join(CACHE_DIR, "sector_etfs_daily.parquet")
CLAIMS_CACHE = os.path.join(CACHE_DIR, "ic4wsa.csv")

START = "1998-12-22"        # sector-ETF (XLP/XLU/XLY/XLI) common inception
AS_OF = "2026-06-30"        # last complete calendar month at publication

CYCLICALS = ["XLY", "XLI"]      # consumer-discretionary + industrials
DEFENSIVES = ["XLP", "XLU"]     # consumer-staples + utilities
SECTORS = CYCLICALS + DEFENSIVES
TICKERS = SECTORS + ["SPY"]

# FRED CSV endpoints attempted by fetch_claims_csv (unreachable in this build).
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IC4WSA"

__all__ = [
    "CYCLICALS", "DEFENSIVES", "SECTORS", "TICKERS", "START", "AS_OF",
    "CACHE_DIR", "ETF_CACHE", "ICSA_4WK_BY_YEAR",
    "fetch", "fetch_etfs", "fetch_claims_csv", "have_real",
    "claims_series", "load_etfs", "load_real", "synthetic_frame", "fingerprint",
]


# --------------------------------------------------------------------------- #
# Real claims tape — hardcoded monthly snapshot of FRED IC4WSA (thousands, SA)
# --------------------------------------------------------------------------- #
# U.S. initial unemployment-insurance claims, SEASONALLY ADJUSTED, 4-week moving
# average, in THOUSANDS. Source: U.S. Dept. of Labor / ETA (FRED series IC4WSA).
# Monthly end-of-month snapshot, Jan..Dec per row. As-of 2026-06-22. Public,
# widely-reported, never-revised-in-this-snapshot prints; the COVID spike of 2020
# (the 4-wk MA peaked near 5,000k in April 2020) is the famous outlier, included
# faithfully. Values rounded to the nearest thousand. (Same source table as Study 385.)
ICSA_4WK_BY_YEAR: dict[int, list[float]] = {
    1993: [344, 339, 333, 341, 337, 343, 345, 339, 332, 327, 322, 318],
    1994: [333, 339, 338, 343, 346, 351, 348, 339, 332, 330, 332, 337],
    1995: [340, 350, 357, 367, 376, 383, 384, 376, 366, 360, 366, 372],
    1996: [371, 363, 358, 354, 357, 360, 357, 350, 343, 342, 345, 351],
    1997: [343, 336, 332, 330, 332, 333, 327, 320, 318, 320, 323, 322],
    1998: [323, 320, 317, 313, 312, 318, 322, 317, 309, 306, 308, 312],
    1999: [307, 301, 299, 304, 305, 305, 301, 298, 293, 292, 290, 286],
    2000: [288, 291, 295, 297, 295, 296, 304, 305, 307, 308, 313, 327],
    2001: [325, 330, 338, 355, 388, 401, 391, 396, 437, 489, 451, 426],
    2002: [402, 386, 386, 421, 416, 396, 388, 391, 410, 408, 387, 408],
    2003: [398, 401, 421, 446, 433, 422, 410, 397, 405, 388, 369, 362],
    2004: [346, 351, 343, 339, 336, 346, 343, 343, 343, 350, 339, 333],
    2005: [325, 314, 318, 326, 333, 332, 333, 318, 369, 366, 327, 322],
    2006: [310, 296, 305, 312, 330, 318, 318, 316, 318, 311, 320, 317],
    2007: [325, 326, 318, 322, 308, 318, 309, 327, 320, 327, 336, 343],
    2008: [332, 357, 367, 369, 374, 386, 392, 442, 477, 478, 524, 555],
    2009: [543, 622, 654, 660, 627, 616, 573, 567, 553, 526, 504, 481],
    2010: [468, 470, 449, 460, 458, 463, 458, 481, 460, 456, 432, 442],
    2011: [428, 419, 391, 425, 437, 426, 413, 415, 419, 405, 397, 379],
    2012: [375, 360, 365, 379, 374, 387, 376, 372, 376, 369, 397, 379],
    2013: [359, 351, 352, 357, 348, 348, 343, 332, 308, 351, 333, 348],
    2014: [333, 337, 320, 318, 311, 315, 303, 301, 295, 282, 295, 290],
    2015: [283, 284, 298, 285, 272, 273, 277, 274, 271, 264, 271, 277],
    2016: [284, 274, 263, 261, 277, 268, 261, 263, 257, 254, 251, 264],
    2017: [255, 245, 250, 244, 240, 244, 242, 237, 270, 240, 240, 241],
    2018: [244, 226, 224, 230, 222, 222, 219, 213, 207, 211, 224, 219],
    2019: [221, 224, 224, 206, 217, 222, 212, 215, 213, 214, 218, 224],
    2020: [217, 211, 998, 4174, 3036, 1503, 1370, 992, 868, 776, 749, 818],
    2021: [848, 829, 749, 656, 460, 393, 395, 366, 340, 281, 271, 206],
    2022: [219, 232, 224, 191, 209, 233, 249, 247, 213, 211, 228, 213],
    2023: [206, 195, 198, 240, 233, 256, 234, 238, 211, 211, 220, 213],
    2024: [208, 213, 211, 214, 220, 237, 236, 231, 224, 238, 218, 224],
    2025: [217, 224, 224, 221, 230, 245, 235, 232, 240, 233, 228, 231],
    2026: [231, 234, 238, 241, 244, 246, 0, 0, 0, 0, 0, 0],
}


def claims_series() -> pd.Series:
    """Monthly 4-week-MA initial claims (thousands), indexed by month-end date.

    The in-progress months of the final year (zeros) are dropped so a stamped run
    never includes a partial bar.
    """
    rows = []
    for yr in sorted(ICSA_4WK_BY_YEAR):
        for m, v in enumerate(ICSA_4WK_BY_YEAR[yr], start=1):
            if v <= 0:
                continue
            rows.append((pd.Timestamp(yr, m, 1) + pd.offsets.MonthEnd(0), float(v)))
    idx = pd.DatetimeIndex([d for d, _ in rows])
    return pd.Series([v for _, v in rows], index=idx, name="claims").sort_index()


# --------------------------------------------------------------------------- #
# Real sector tape — yfinance daily closes, cached
# --------------------------------------------------------------------------- #
def have_real(path: str = ETF_CACHE) -> bool:
    """True iff the sector-ETF cache exists (the claims table is always available)."""
    return os.path.exists(path)


def fetch_etfs(start: str = START, path: str = ETF_CACHE) -> pd.DataFrame:
    """Download the four sector ETFs + SPY (yfinance, total-return) and cache them.

    Network-only; used once to build the parquet. Never imported by offline cells.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    raw = yf.download(TICKERS, start=start, end="2026-07-01",
                      auto_adjust=True, progress=False)
    closes = raw["Close"].copy()
    closes.columns = [str(c) for c in closes.columns]
    closes = closes[TICKERS].dropna()
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    closes.index.name = "date"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    closes.to_parquet(path)
    return closes


def fetch_claims_csv(path: str = CLAIMS_CACHE, retries: int = 4) -> pd.Series | None:
    """Attempt the live FRED IC4WSA CSV (retry, proper UA), fall back to DBnomics.

    Returns the fetched weekly series on success (also caches it), or ``None`` when the
    endpoints are unreachable — in which case callers rely on the hardcoded
    ``ICSA_4WK_BY_YEAR`` snapshot. In this build both hosts are firewalled, so this
    returns ``None`` and the study runs on the documented public snapshot.
    """
    import time
    import urllib.request

    urls = [
        FRED_CSV,
        "https://api.db.nomics.world/v22/series/FRED/IC4WSA?observations=1",
    ]
    for url in urls:
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Open-Alpha-Lab/881 (research; contact via GitHub)"})
                raw = urllib.request.urlopen(req, timeout=30).read().decode()
                if url is FRED_CSV:
                    df = pd.read_csv(pd.io.common.StringIO(raw))
                    df.columns = ["date", "value"]
                    df["date"] = pd.to_datetime(df["date"])
                    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                                  index=df["date"], name="claims").dropna()
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    s.to_csv(path)
                    return s
            except Exception:
                time.sleep(1.0 * (attempt + 1))
    return None


def load_etfs(path: str = ETF_CACHE) -> pd.DataFrame:
    """Cached daily sector-ETF + SPY closes (total-return). OFFLINE, no yfinance."""
    df = pd.read_parquet(path).sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[TICKERS].astype(float)


def load_real(path: str = ETF_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """The real-tape monthly frame the strategy runs on.

    Month-end alignment of the 4-week-MA claims level with the month-end sector-ETF and
    SPY closes; only months present in both are kept, sliced to ``[.., asof]``. Columns:
    ``claims`` (thousands) and one column per ticker (price).
    """
    claims = claims_series()
    etfs = load_etfs(path).resample("ME").last()
    frame = pd.concat([claims, etfs], axis=1).dropna()
    frame = frame[frame.index <= pd.Timestamp(asof)]
    return frame


# --------------------------------------------------------------------------- #
# Synthetic control — planted rising-claims -> cyclicals-underperform relation
# --------------------------------------------------------------------------- #
def synthetic_frame(
    edge: float = 0.0,
    seed: int = 881,
    n_months: int = 360,
    claims_rho: float = 0.90,
    claims_sig: float = 0.05,
    mkt_mu: float = 0.006,
    mkt_sig: float = 0.043,
) -> pd.DataFrame:
    """Deterministic monthly frame (claims + 4 sector prices) with a PLANTED rotation.

    A mean-reverting log-claims level (AR(1) around ~300k) drives the claims change
    ``dcl_t = claims_t/claims_{t-1} - 1``. A market factor drives all four sectors with
    cyclical betas (~1.0) above defensive betas (~0.5). When ``edge > 0`` the *next*
    month's cyclical returns are knocked down by ``-edge * dcl_t`` — so the
    **cyclical-minus-defensive** spread loads on the claims change with slope ``≈ -edge``
    (the claim, by construction).

    ``edge = 0`` is the null: claims still move but carry no rotation information, and
    the predictive regression must find nothing. The index is a decorative month-end
    label built from ``period_range`` (no ns-timestamp overflow risk for long spans).
    """
    rng = np.random.default_rng(seed)

    log_c = np.empty(n_months)
    log_c[0] = np.log(300.0)
    target = np.log(300.0)
    for t in range(1, n_months):
        log_c[t] = target + claims_rho * (log_c[t - 1] - target) + rng.normal(0, claims_sig)
    claims = np.exp(log_c)
    dcl = np.concatenate([[0.0], claims[1:] / claims[:-1] - 1.0])

    mkt = rng.normal(mkt_mu, mkt_sig, n_months)
    cyc1 = 1.05 * mkt + rng.normal(0, 0.025, n_months)   # XLY
    cyc2 = 1.00 * mkt + rng.normal(0, 0.026, n_months)   # XLI
    def1 = 0.55 * mkt + rng.normal(0, 0.020, n_months)   # XLP
    def2 = 0.45 * mkt + rng.normal(0, 0.022, n_months)   # XLU

    plant = np.zeros(n_months)
    plant[1:] = -edge * dcl[:-1]     # rising claims at t-1 depress cyclicals at t
    cyc1 = cyc1 + plant
    cyc2 = cyc2 + plant

    idx = pd.period_range("2000-01", periods=n_months, freq="M").to_timestamp("M")
    return pd.DataFrame(
        {
            "claims": claims,
            "XLY": 100.0 * np.cumprod(1.0 + cyc1),
            "XLI": 90.0 * np.cumprod(1.0 + cyc2),
            "XLP": 50.0 * np.cumprod(1.0 + def1),
            "XLU": 40.0 * np.cumprod(1.0 + def2),
        },
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Orchestration + fingerprint
# --------------------------------------------------------------------------- #
def fetch() -> None:
    """Build the real caches: the sector-ETF parquet (network) and, if reachable, the
    live FRED IC4WSA CSV (otherwise the hardcoded snapshot stands)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fetch_etfs()
    fetch_claims_csv()


def fingerprint(df: pd.DataFrame) -> str:
    """Short content fingerprint of the four sector-ETF price columns (as-of stamp)."""
    cols = [c for c in SECTORS if c in df.columns]
    h = hashlib.sha1(np.ascontiguousarray(df[cols].to_numpy()).tobytes())
    return h.hexdigest()[:12]

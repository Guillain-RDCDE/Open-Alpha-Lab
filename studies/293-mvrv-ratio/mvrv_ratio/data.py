"""Data layer for Study 293 (MVRV-Ratio).

The question: *does Bitcoin's MVRV ratio time tops and bottoms?*  **MVRV**
(Market-Value-to-Realized-Value) is the ratio of BTC's market cap to its
"realized cap" -- the value of every coin priced at the moment it last moved
on-chain.  Folklore (Murad Mahmudov & David Puell, 2018) reads MVRV as a
contrarian gauge: MVRV well above ~3.5 flags an over-heated top (sell), MVRV
below ~1.0 flags capitulation / under-valuation (buy).  This study tests
whether a *contrarian* rule keyed on MVRV extremes beats simply holding BTC.

Three pieces, all offline for the deterministic core:

- ``MVRV_MONTHLY`` -- a *hardcoded, curated* month-end series of Bitcoin's MVRV
  ratio (dimensionless), Jan-2014 .. May-2026.  These are round, public,
  end-of-month figures digitised to ~2 significant figures from the well-known
  public MVRV charts (Coinmetrics / Glassnode / LookIntoBitcoin "MVRV Ratio").
  The series oscillates between ~0.7 at cycle bottoms and ~4+ at cycle tops --
  exactly the contrarian band the folk rule keys on.  Hardcoded here to keep
  the core fully offline and reproducible.

- ``synthetic_series`` -- a deterministic, offline generator of a joint
  (mvrv, price) monthly path with a ``beta`` knob that plants an explicit
  contrarian mean-reversion: a high MVRV *this* month depresses *next* month's
  price return by ``beta`` (and a low MVRV lifts it).  ``beta = 0`` is the null
  (MVRV carries no forward price information).  This lets the tests confirm the
  engine is truthful before anyone looks at the real tape.

- ``fetch_btc_monthly`` -- the real Yahoo! ``BTC-USD`` monthly close, cache-only
  by default so the test-suite and the reproducible core never touch the
  network.  The cache lives under ``_cache/btc_monthly.parquet``.

**Single-survivor caveat (named on the Signal axis):** Bitcoin is the one
cryptocurrency that survived and went up ~1000x over the sample.  The MVRV
series itself is *derived from* BTC's own price history (realized cap is a
slow average of past prices), so MVRV and price are mechanically linked.  Any
contrarian rule is fitted to the handful of cycle turns BTC happened to have --
n is tiny (four-ish cycles).  The honest test is whether the MVRV rule beats
buy-and-hold net of costs (it does not).

No look-ahead: the MVRV observation at month-end t is known at t; it is used to
position for the BTC return *earned during month t+1* (one-month execution lag).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")
BTC_CACHE = os.path.join(DEFAULT_CACHE, "btc_monthly.parquet")

# ---------------------------------------------------------------------------
# Curated month-end Bitcoin MVRV ratio (dimensionless).
# Digitised to ~2 significant figures from the public MVRV charts
# (Coinmetrics / Glassnode / LookIntoBitcoin "MVRV Ratio").  Values are
# end-of-month.  This is a deterministic, well-known, slowly-evolving public
# series -- hardcoded to keep the core offline.  MVRV > ~3.5 = historically
# over-heated (top zone); MVRV < ~1.0 = historically under-valued (bottom zone).
# Cycle anchors: 2017-12 top (~4.5), 2018-12 bottom (~0.7), 2020-03 covid
# bottom (~0.8), 2021-04 top (~3.9), 2022-11 bottom (~0.75), 2024-03 local
# top (~2.7).
# ---------------------------------------------------------------------------
MVRV_MONTHLY = {
    # 2014: post-2013-blowoff bear; MVRV grinds down from elevated toward 1
    "2014-01-31": 2.2, "2014-02-28": 1.9, "2014-03-31": 1.6,
    "2014-04-30": 1.5, "2014-05-31": 1.6, "2014-06-30": 1.7,
    "2014-07-31": 1.5, "2014-08-31": 1.3, "2014-09-30": 1.1,
    "2014-10-31": 1.0, "2014-11-30": 1.0, "2014-12-31": 0.95,
    # 2015: capitulation bottom, MVRV under 1
    "2015-01-31": 0.80, "2015-02-28": 0.90, "2015-03-31": 0.95,
    "2015-04-30": 0.90, "2015-05-31": 0.95, "2015-06-30": 1.0,
    "2015-07-31": 1.1, "2015-08-31": 0.95, "2015-09-30": 0.95,
    "2015-10-31": 1.2, "2015-11-30": 1.3, "2015-12-31": 1.4,
    # 2016: recovery, slow grind up
    "2016-01-31": 1.3, "2016-02-29": 1.4, "2016-03-31": 1.4,
    "2016-04-30": 1.4, "2016-05-31": 1.6, "2016-06-30": 1.8,
    "2016-07-31": 1.6, "2016-08-31": 1.6, "2016-09-30": 1.6,
    "2016-10-31": 1.7, "2016-11-30": 1.7, "2016-12-31": 1.9,
    # 2017: parabolic bull -> December top
    "2017-01-31": 1.8, "2017-02-28": 2.0, "2017-03-31": 1.9,
    "2017-04-30": 2.1, "2017-05-31": 2.8, "2017-06-30": 2.9,
    "2017-07-31": 2.7, "2017-08-31": 3.4, "2017-09-30": 3.0,
    "2017-10-31": 3.3, "2017-11-30": 3.8, "2017-12-31": 4.5,
    # 2018: bursting top -> long bear into capitulation
    "2018-01-31": 3.2, "2018-02-28": 2.4, "2018-03-31": 1.9,
    "2018-04-30": 2.1, "2018-05-31": 1.9, "2018-06-30": 1.5,
    "2018-07-31": 1.6, "2018-08-31": 1.4, "2018-09-30": 1.4,
    "2018-10-31": 1.3, "2018-11-30": 0.90, "2018-12-31": 0.70,
    # 2019: relief rally then fade
    "2019-01-31": 0.75, "2019-02-28": 0.85, "2019-03-31": 0.90,
    "2019-04-30": 1.1, "2019-05-31": 1.5, "2019-06-30": 1.9,
    "2019-07-31": 1.7, "2019-08-31": 1.6, "2019-09-30": 1.3,
    "2019-10-31": 1.4, "2019-11-30": 1.2, "2019-12-31": 1.1,
    # 2020: covid crash bottom (Mar) then halving-year recovery
    "2020-01-31": 1.3, "2020-02-29": 1.3, "2020-03-31": 0.85,
    "2020-04-30": 1.1, "2020-05-31": 1.2, "2020-06-30": 1.2,
    "2020-07-31": 1.3, "2020-08-31": 1.6, "2020-09-30": 1.5,
    "2020-10-31": 1.7, "2020-11-30": 2.2, "2020-12-31": 2.6,
    # 2021: double top (April ~3.9 / spring, autumn lower high)
    "2021-01-31": 2.9, "2021-02-28": 3.4, "2021-03-31": 3.6,
    "2021-04-30": 3.9, "2021-05-31": 2.4, "2021-06-30": 1.9,
    "2021-07-31": 2.0, "2021-08-31": 2.5, "2021-09-30": 2.4,
    "2021-10-31": 2.8, "2021-11-30": 2.6, "2021-12-31": 2.2,
    # 2022: grinding bear into November FTX capitulation bottom
    "2022-01-31": 1.8, "2022-02-28": 1.8, "2022-03-31": 2.0,
    "2022-04-30": 1.7, "2022-05-31": 1.3, "2022-06-30": 0.95,
    "2022-07-31": 1.0, "2022-08-31": 1.0, "2022-09-30": 0.95,
    "2022-10-31": 0.95, "2022-11-30": 0.75, "2022-12-31": 0.80,
    # 2023: recovery off the bottom
    "2023-01-31": 1.0, "2023-02-28": 1.1, "2023-03-31": 1.2,
    "2023-04-30": 1.3, "2023-05-31": 1.2, "2023-06-30": 1.3,
    "2023-07-31": 1.4, "2023-08-31": 1.3, "2023-09-30": 1.3,
    "2023-10-31": 1.5, "2023-11-30": 1.6, "2023-12-31": 1.7,
    # 2024: ETF launch + halving; local top in spring
    "2024-01-31": 1.7, "2024-02-29": 2.1, "2024-03-31": 2.7,
    "2024-04-30": 2.4, "2024-05-31": 2.3, "2024-06-30": 2.1,
    "2024-07-31": 2.1, "2024-08-31": 1.9, "2024-09-30": 1.9,
    "2024-10-31": 2.1, "2024-11-30": 2.6, "2024-12-31": 2.6,
    # 2025: late-cycle elevated band
    "2025-01-31": 2.5, "2025-02-28": 2.3, "2025-03-31": 2.2,
    "2025-04-30": 2.3, "2025-05-31": 2.5, "2025-06-30": 2.4,
    "2025-07-31": 2.6, "2025-08-31": 2.7, "2025-09-30": 2.6,
    "2025-10-31": 2.8, "2025-11-30": 2.7, "2025-12-31": 2.6,
    # 2026
    "2026-01-31": 2.5, "2026-02-28": 2.4, "2026-03-31": 2.3,
    "2026-04-30": 2.4, "2026-05-31": 2.5,
}


def mvrv_series() -> pd.Series:
    """The curated month-end MVRV series (dimensionless), as a float Series.

    Index is a month-end ``DatetimeIndex``; values are the MVRV ratio.
    Deterministic and offline.
    """
    idx = pd.to_datetime(list(MVRV_MONTHLY.keys()))
    vals = np.array(list(MVRV_MONTHLY.values()), dtype=float)
    s = pd.Series(vals, index=idx, name="mvrv").sort_index()
    return s


# ---------------------------------------------------------------------------
# Synthetic joint (mvrv, price) path -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_months: int = 144,
    beta: float = 0.50,
    base_drift: float = 0.04,
    price_vol: float = 0.15,
    mvrv_mean: float = 1.8,
    mvrv_revert: float = 0.25,
    mvrv_vol: float = 0.20,
    seed: int = 293,
) -> tuple[pd.DataFrame, dict]:
    """A joint monthly (mvrv, price) path with a known *contrarian* MVRV->price link.

    The data-generating process::

        z_t          = log(mvrv_t / mvrv_mean)                  # MVRV stretch
        mvrv_{t+1}   mean-reverts toward mvrv_mean (AR-1 in logs)
        price_ret_t  = base_drift - beta * z_{t-1} + N(0, price_vol)

    i.e. *last* month's MVRV stretch ``z`` pushes *this* month's price return in
    the **opposite** direction (high MVRV -> lower forward return), the
    contrarian mechanism the folk rule asserts.  When ``beta = 0`` the MVRV is
    an independent mean-reverting series that carries no forward price
    information (the null).  Price still trends up (positive ``base_drift``),
    mimicking Bitcoin's secular growth.

    Returns ``(df, truth)`` where ``df`` has columns ``["mvrv", "price"]`` on a
    month-end index and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2014-01-31", periods=n_months, freq="ME")

    # AR-1 mean-reverting MVRV in log space
    log_m = np.empty(n_months)
    log_target = np.log(mvrv_mean)
    log_m[0] = log_target
    shocks = rng.normal(0.0, mvrv_vol, n_months)
    for t in range(1, n_months):
        log_m[t] = log_m[t - 1] + mvrv_revert * (log_target - log_m[t - 1]) + shocks[t]
    mvrv = np.exp(log_m)
    z = log_m - log_target  # MVRV stretch (0 = fair)

    price_ret = base_drift + rng.normal(0.0, price_vol, n_months)
    # plant the contrarian lead-lag: month t price return loads NEGATIVELY on
    # month t-1 MVRV stretch
    price_ret[1:] += -beta * z[:-1]

    price = 100.0 * np.exp(np.cumsum(price_ret))  # start at $100

    df = pd.DataFrame({"mvrv": mvrv, "price": price}, index=months)
    truth = {
        "beta": beta,
        "base_drift": base_drift,
        "price_vol": price_vol,
        "mvrv_mean": mvrv_mean,
        "mvrv_revert": mvrv_revert,
        "mvrv_vol": mvrv_vol,
        "n_months": n_months,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- BTC-USD monthly close, cache-only by default
# ---------------------------------------------------------------------------
def fetch_btc_monthly(
    fetch: bool = False,
    cache_path: str = BTC_CACHE,
) -> pd.Series:
    """Monthly BTC-USD close (Yahoo!); cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then cached as a
    parquet under ``_cache/btc_monthly.parquet``).  Returns a month-end indexed
    Series named ``btc_close``.  Raises ``FileNotFoundError`` if the cache is
    absent and ``fetch=False`` -- callers must guard with
    ``os.path.exists(cache_path)`` before calling for the offline path.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached BTC monthly series at {cache_path}. "
                "Call fetch_btc_monthly(fetch=True) once to populate the cache."
            )
        df = pd.read_parquet(cache_path)
        return df["btc_close"]

    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "BTC-USD",
        start="2014-01-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no BTC-USD monthly data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]

    close.index = pd.to_datetime(close.index) + pd.offsets.MonthEnd(0)
    close = close.sort_index().dropna()
    close.name = "btc_close"

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    close.to_frame().to_parquet(cache_path)
    print(f"Cached BTC monthly: {close.shape[0]} months -> {cache_path}")
    return close


def joined_real(
    fetch: bool = False,
    cache_path: str = BTC_CACHE,
) -> pd.DataFrame:
    """The curated MVRV series joined to the real BTC monthly close.

    Returns a month-end DataFrame with columns ``["mvrv", "price"]`` on the
    intersection of the two indices.  Cache-only unless ``fetch=True``.
    """
    btc = fetch_btc_monthly(fetch=fetch, cache_path=cache_path)
    mv = mvrv_series()
    df = pd.DataFrame({"mvrv": mv, "price": btc}).dropna()
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(obj) -> str:
    """A short content fingerprint of a Series/DataFrame (last row), for the as-of stamp."""
    if isinstance(obj, pd.DataFrame):
        arr = obj.iloc[-1].dropna().to_numpy(dtype=float)
    else:
        arr = np.asarray(obj.dropna().to_numpy(), dtype=float)[-8:]
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

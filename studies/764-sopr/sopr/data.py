"""Data layer for Study 764 (SOPR).

The question: *does Bitcoin's SOPR time capitulation and greed?*  **SOPR**
(Spent Output Profit Ratio) is the aggregate ratio of the value at which coins
are *spent* (moved on-chain) to the value at which they were *created* -- in
plain terms, the average "sale price / cost basis" of the coins that changed
hands.  Folklore (Renato Shirakashi, 2019; popularised by Glassnode) reads SOPR
as a *momentum / regime* gauge keyed on the number **1**: SOPR **> 1** means the
coins moving are, on aggregate, *in profit* (holders realising gains, a healthy
uptrend); SOPR **< 1** means coins move at a *loss* (capitulation, a downtrend).
The famous chart-lore: in bull markets SOPR bounces off 1 as *support*, in bear
markets it caps at 1 as *resistance*.  This study tests whether the "be long
when SOPR > 1, step aside when SOPR < 1" rule beats simply holding BTC.

Three pieces, all offline for the deterministic core:

- ``SOPR_MONTHLY`` -- a *hardcoded, curated* month-end series of Bitcoin's
  adjusted SOPR (dimensionless), Jan-2014 .. Jun-2026.  These are round, public,
  end-of-month figures digitised to ~2 decimal places from the well-known public
  aSOPR charts (Glassnode "Adjusted SOPR" / cited on-chain dashboards).  The
  series oscillates tightly around **1** -- dipping to ~0.93 at capitulation
  bottoms (2018-12, 2020-03 covid, 2022-11 FTX) and rising to ~1.05 in bull
  euphoria (2017-12, 2021, 2024).  Hardcoded here to keep the core fully offline
  and reproducible.  **This is a LABELLED PROXY, not a live Glassnode feed.**

- ``synthetic_series`` -- a deterministic, offline generator of a joint
  (sopr, price) monthly path with a ``beta`` knob that plants an explicit
  *momentum* lead-lag: a high SOPR *this* month lifts *next* month's price
  return by ``beta`` (and a low SOPR depresses it) -- exactly the trend story the
  ">1 is bullish" rule asserts.  ``beta = 0`` is the null (SOPR carries no
  forward price information).  This lets the tests confirm the engine is truthful
  before anyone looks at the real tape.

- ``fetch_btc_monthly`` -- the real Yahoo! ``BTC-USD`` monthly close, cache-only
  by default so the test-suite and the reproducible core never touch the
  network.  The cache lives under ``_cache/btc_monthly.parquet``.

**Single-survivor caveat (named on the Signal axis):** Bitcoin is the one
cryptocurrency that survived and went up ~150x over the aligned sample.  SOPR is
computed *from* BTC's own on-chain spending against past prices, so SOPR and
price are mechanically entangled.  Any regime rule is fitted to the handful of
cycle turns BTC happened to have.  The honest test is whether the SOPR rule
beats buy-and-hold net of costs (it does not).

No look-ahead: the SOPR observation at month-end t is known at t; it is used to
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
# Curated month-end Bitcoin adjusted-SOPR (dimensionless).
# Digitised to ~2 decimal places from the public aSOPR charts (Glassnode
# "Adjusted SOPR").  Values are end-of-month.  This is a deterministic,
# well-known, slowly-evolving public series -- a LABELLED PROXY hardcoded to
# keep the core offline.  SOPR > 1 = coins move in profit (folk "bullish
# regime"); SOPR < 1 = coins move at a loss (folk "capitulation").
# Cycle anchors: 2017-12 top (~1.05), 2018-12 capitulation (~0.93), 2020-03
# covid crash (~0.93), 2021-02 top (~1.04), 2022-11 FTX bottom (~0.94),
# 2024-03 local top (~1.03).
# ---------------------------------------------------------------------------
SOPR_MONTHLY = {
    # 2014: post-2013-blowoff bear; losses realised, SOPR grinds under 1
    "2014-01-31": 0.99, "2014-02-28": 0.98, "2014-03-31": 0.97,
    "2014-04-30": 0.98, "2014-05-31": 0.99, "2014-06-30": 1.00,
    "2014-07-31": 0.99, "2014-08-31": 0.98, "2014-09-30": 0.97,
    "2014-10-31": 0.96, "2014-11-30": 0.97, "2014-12-31": 0.96,
    # 2015: capitulation bottom early, then slow recovery back over 1
    "2015-01-31": 0.95, "2015-02-28": 0.98, "2015-03-31": 0.99,
    "2015-04-30": 0.99, "2015-05-31": 1.00, "2015-06-30": 1.00,
    "2015-07-31": 1.01, "2015-08-31": 0.98, "2015-09-30": 0.99,
    "2015-10-31": 1.01, "2015-11-30": 1.02, "2015-12-31": 1.02,
    # 2016: slow grind-up bull, SOPR mostly above 1
    "2016-01-31": 1.00, "2016-02-29": 1.01, "2016-03-31": 1.01,
    "2016-04-30": 1.01, "2016-05-31": 1.02, "2016-06-30": 1.03,
    "2016-07-31": 1.00, "2016-08-31": 1.01, "2016-09-30": 1.01,
    "2016-10-31": 1.01, "2016-11-30": 1.01, "2016-12-31": 1.02,
    # 2017: parabolic bull -> December euphoric top
    "2017-01-31": 1.01, "2017-02-28": 1.02, "2017-03-31": 1.01,
    "2017-04-30": 1.02, "2017-05-31": 1.03, "2017-06-30": 1.02,
    "2017-07-31": 1.01, "2017-08-31": 1.03, "2017-09-30": 1.01,
    "2017-10-31": 1.03, "2017-11-30": 1.03, "2017-12-31": 1.05,
    # 2018: bursting top -> long bear into capitulation
    "2018-01-31": 1.02, "2018-02-28": 0.99, "2018-03-31": 0.98,
    "2018-04-30": 1.00, "2018-05-31": 0.99, "2018-06-30": 0.98,
    "2018-07-31": 1.00, "2018-08-31": 0.98, "2018-09-30": 0.99,
    "2018-10-31": 0.98, "2018-11-30": 0.94, "2018-12-31": 0.93,
    # 2019: relief rally then fade
    "2019-01-31": 0.97, "2019-02-28": 0.99, "2019-03-31": 1.00,
    "2019-04-30": 1.02, "2019-05-31": 1.03, "2019-06-30": 1.03,
    "2019-07-31": 1.00, "2019-08-31": 0.99, "2019-09-30": 0.98,
    "2019-10-31": 1.00, "2019-11-30": 0.98, "2019-12-31": 0.98,
    # 2020: covid crash bottom (Mar ~0.93) then halving-year recovery
    "2020-01-31": 1.01, "2020-02-29": 1.00, "2020-03-31": 0.93,
    "2020-04-30": 1.01, "2020-05-31": 1.01, "2020-06-30": 1.00,
    "2020-07-31": 1.02, "2020-08-31": 1.02, "2020-09-30": 0.99,
    "2020-10-31": 1.02, "2020-11-30": 1.03, "2020-12-31": 1.03,
    # 2021: double top (Feb-Apr, autumn lower high), May crash
    "2021-01-31": 1.03, "2021-02-28": 1.04, "2021-03-31": 1.03,
    "2021-04-30": 1.03, "2021-05-31": 0.97, "2021-06-30": 0.98,
    "2021-07-31": 1.00, "2021-08-31": 1.02, "2021-09-30": 1.00,
    "2021-10-31": 1.03, "2021-11-30": 1.02, "2021-12-31": 0.99,
    # 2022: grinding bear into November FTX capitulation bottom
    "2022-01-31": 0.98, "2022-02-28": 0.99, "2022-03-31": 1.01,
    "2022-04-30": 0.99, "2022-05-31": 0.96, "2022-06-30": 0.94,
    "2022-07-31": 0.98, "2022-08-31": 0.99, "2022-09-30": 0.98,
    "2022-10-31": 0.99, "2022-11-30": 0.94, "2022-12-31": 0.96,
    # 2023: recovery off the bottom
    "2023-01-31": 1.01, "2023-02-28": 1.02, "2023-03-31": 1.02,
    "2023-04-30": 1.02, "2023-05-31": 0.99, "2023-06-30": 1.01,
    "2023-07-31": 1.01, "2023-08-31": 0.99, "2023-09-30": 0.99,
    "2023-10-31": 1.02, "2023-11-30": 1.02, "2023-12-31": 1.02,
    # 2024: ETF launch + halving; local top in spring
    "2024-01-31": 1.01, "2024-02-29": 1.03, "2024-03-31": 1.03,
    "2024-04-30": 1.01, "2024-05-31": 1.01, "2024-06-30": 0.99,
    "2024-07-31": 1.00, "2024-08-31": 0.98, "2024-09-30": 1.00,
    "2024-10-31": 1.02, "2024-11-30": 1.03, "2024-12-31": 1.02,
    # 2025: late-cycle elevated but choppy
    "2025-01-31": 1.02, "2025-02-28": 0.99, "2025-03-31": 0.99,
    "2025-04-30": 1.00, "2025-05-31": 1.02, "2025-06-30": 1.01,
    "2025-07-31": 1.02, "2025-08-31": 1.01, "2025-09-30": 1.00,
    "2025-10-31": 1.02, "2025-11-30": 1.00, "2025-12-31": 0.99,
    # 2026: fade
    "2026-01-31": 1.00, "2026-02-28": 0.99, "2026-03-31": 0.98,
    "2026-04-30": 0.99, "2026-05-31": 0.98, "2026-06-30": 0.96,
}


def sopr_series() -> pd.Series:
    """The curated month-end adjusted-SOPR series (dimensionless), as a Series.

    Index is a month-end ``DatetimeIndex``; values are the aSOPR ratio.
    Deterministic and offline.
    """
    idx = pd.to_datetime(list(SOPR_MONTHLY.keys()))
    vals = np.array(list(SOPR_MONTHLY.values()), dtype=float)
    s = pd.Series(vals, index=idx, name="sopr").sort_index()
    return s


# ---------------------------------------------------------------------------
# Synthetic joint (sopr, price) path -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_months: int = 144,
    beta: float = 0.50,
    base_drift: float = 0.04,
    price_vol: float = 0.15,
    sopr_center: float = 1.0,
    sopr_amp: float = 0.03,
    sopr_revert: float = 0.25,
    sopr_vol: float = 0.02,
    seed: int = 764,
) -> tuple[pd.DataFrame, dict]:
    """A joint monthly (sopr, price) path with a known *momentum* SOPR->price link.

    The data-generating process::

        z_t          = log(sopr_t / sopr_center)               # SOPR stretch
        sopr_{t+1}   mean-reverts toward sopr_center (AR-1 in logs)
        price_ret_t  = base_drift + beta * z_{t-1} + N(0, price_vol)

    i.e. *last* month's SOPR stretch ``z`` pushes *this* month's price return in
    the **same** direction (SOPR > 1 -> higher forward return), the *momentum /
    regime* mechanism the ">1 is bullish" rule asserts.  When ``beta = 0`` the
    SOPR is an independent mean-reverting series that carries no forward price
    information (the null).  Price still trends up (positive ``base_drift``),
    mimicking Bitcoin's secular growth.

    Returns ``(df, truth)`` where ``df`` has columns ``["sopr", "price"]`` on a
    month-end index and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2014-01-31", periods=n_months, freq="ME")

    # AR-1 mean-reverting SOPR stretch in log space (tight band around 1)
    log_s = np.empty(n_months)
    log_s[0] = 0.0
    shocks = rng.normal(0.0, sopr_vol, n_months)
    for t in range(1, n_months):
        log_s[t] = (1.0 - sopr_revert) * log_s[t - 1] + shocks[t]
    z = log_s  # SOPR stretch (0 = fair, at 1.0)
    sopr = sopr_center * np.exp(z * (sopr_amp / max(sopr_vol, 1e-9)))

    price_ret = base_drift + rng.normal(0.0, price_vol, n_months)
    # plant the momentum lead-lag: month t price return loads POSITIVELY on
    # month t-1 SOPR stretch
    price_ret[1:] += beta * z[:-1]

    price = 100.0 * np.exp(np.cumsum(price_ret))  # start at $100

    df = pd.DataFrame({"sopr": sopr, "price": price}, index=months)
    truth = {
        "beta": beta,
        "base_drift": base_drift,
        "price_vol": price_vol,
        "sopr_center": sopr_center,
        "sopr_revert": sopr_revert,
        "sopr_vol": sopr_vol,
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
    """The curated SOPR series joined to the real BTC monthly close.

    Returns a month-end DataFrame with columns ``["sopr", "price"]`` on the
    intersection of the two indices.  The intersection stops at the last SOPR
    month (2026-06-30), so any in-progress current-month BTC bar is dropped
    automatically (no partial-month contamination).  Cache-only unless
    ``fetch=True``.
    """
    btc = fetch_btc_monthly(fetch=fetch, cache_path=cache_path)
    sp = sopr_series()
    df = pd.DataFrame({"sopr": sp, "price": btc}).dropna()
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

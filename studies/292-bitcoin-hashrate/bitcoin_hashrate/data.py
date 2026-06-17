"""Data layer for Study 292 (Bitcoin-Hashrate).

The question: *does Bitcoin's hash rate predict its price?*  "Hash rate" is the
total computational power miners point at the network -- a proxy for how much
real capital is committed to securing (and betting on) the chain.  The folk
thesis ("price follows hashrate", a.k.a. the "Hash Ribbons" / "miner
capitulation" school) says a rising hash rate is a leading bullish signal.

Three pieces, all offline for the deterministic core:

- ``HASHRATE_MONTHLY`` -- a *hardcoded, curated* month-end series of the Bitcoin
  network hash rate in EH/s (exahashes per second), Jan-2014 .. May-2026.  These
  are round, public, end-of-month figures of the Blockchain.com 7-day-average
  hash-rate chart (https://www.blockchain.com/explorer/charts/hash-rate),
  digitised to two significant figures.  The series is monotone-ish but very
  noisy month to month, which is exactly the point: it is *trending* (mining
  capacity only grows secularly) so a naive "is hashrate up?" rule is almost
  always long.  Hardcoded here to keep the core fully offline and reproducible.

- ``synthetic_series`` -- a deterministic, offline generator of a joint
  (hashrate, price) monthly path with a ``beta`` knob that plants an explicit
  lead-lag: this month's hash-rate *growth* nudges *next* month's price return
  by ``beta``.  ``beta = 0`` is the null (hashrate carries no forward price
  information).  This lets the tests confirm the engine is truthful before
  anyone looks at the real tape.

- ``fetch_btc_monthly`` -- the real Yahoo! ``BTC-USD`` monthly close, cache-only
  by default so the test-suite and the reproducible core never touch the
  network.  The cache lives under ``_cache/btc_monthly.parquet``.

**Survivorship / regime caveat (named on the Signal axis):** Bitcoin is a single
surviving asset that went up ~1000x over the sample.  Hash rate and price both
trend up over the sample, so *any* "trend-following" rule will look brilliant by
construction.  The honest test is whether hashrate growth predicts price
*beyond* price's own momentum and beyond a buy-and-hold benchmark.

No look-ahead: the hashrate observation at month-end t is known at t; it is used
to predict the BTC return *earned during month t+1* (one-month execution lag).
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
# Curated month-end Bitcoin network hash rate, EH/s (exahashes/sec).
# Digitised from the public Blockchain.com 7-day-average hash-rate chart to ~2
# significant figures.  Values are end-of-month.  This is a deterministic,
# well-known, slowly-evolving public series -- hardcoded to keep the core
# offline.  Units: EH/s.  (1 EH/s = 1e18 hashes/sec.)
# ---------------------------------------------------------------------------
HASHRATE_MONTHLY = {
    # 2014: network in the low single-digit EH/s
    "2014-01-31": 0.010, "2014-02-28": 0.020, "2014-03-31": 0.035,
    "2014-04-30": 0.050, "2014-05-31": 0.080, "2014-06-30": 0.110,
    "2014-07-31": 0.140, "2014-08-31": 0.190, "2014-09-30": 0.250,
    "2014-10-31": 0.270, "2014-11-30": 0.290, "2014-12-31": 0.310,
    # 2015
    "2015-01-31": 0.320, "2015-02-28": 0.330, "2015-03-31": 0.360,
    "2015-04-30": 0.380, "2015-05-31": 0.350, "2015-06-30": 0.380,
    "2015-07-31": 0.420, "2015-08-31": 0.420, "2015-09-30": 0.480,
    "2015-10-31": 0.560, "2015-11-30": 0.620, "2015-12-31": 0.780,
    # 2016
    "2016-01-31": 0.900, "2016-02-29": 1.10, "2016-03-31": 1.20,
    "2016-04-30": 1.40, "2016-05-31": 1.40, "2016-06-30": 1.50,
    "2016-07-31": 1.60, "2016-08-31": 1.70, "2016-09-30": 1.80,
    "2016-10-31": 2.00, "2016-11-30": 2.10, "2016-12-31": 2.50,
    # 2017
    "2017-01-31": 3.00, "2017-02-28": 3.40, "2017-03-31": 3.70,
    "2017-04-30": 3.80, "2017-05-31": 4.80, "2017-06-30": 5.50,
    "2017-07-31": 6.40, "2017-08-31": 7.50, "2017-09-30": 9.30,
    "2017-10-31": 11.0, "2017-11-30": 11.0, "2017-12-31": 14.0,
    # 2018: ATH then bear
    "2018-01-31": 16.0, "2018-02-28": 21.0, "2018-03-31": 25.0,
    "2018-04-30": 27.0, "2018-05-31": 34.0, "2018-06-30": 40.0,
    "2018-07-31": 39.0, "2018-08-31": 51.0, "2018-09-30": 53.0,
    "2018-10-31": 51.0, "2018-11-30": 42.0, "2018-12-31": 41.0,
    # 2019
    "2019-01-31": 41.0, "2019-02-28": 43.0, "2019-03-31": 46.0,
    "2019-04-30": 52.0, "2019-05-31": 56.0, "2019-06-30": 66.0,
    "2019-07-31": 72.0, "2019-08-31": 78.0, "2019-09-30": 90.0,
    "2019-10-31": 95.0, "2019-11-30": 90.0, "2019-12-31": 97.0,
    # 2020: halving in May, China still dominant
    "2020-01-31": 110.0, "2020-02-29": 120.0, "2020-03-31": 110.0,
    "2020-04-30": 120.0, "2020-05-31": 110.0, "2020-06-30": 120.0,
    "2020-07-31": 125.0, "2020-08-31": 130.0, "2020-09-30": 135.0,
    "2020-10-31": 130.0, "2020-11-30": 140.0, "2020-12-31": 150.0,
    # 2021: China mining ban mid-year -> big drawdown then recovery
    "2021-01-31": 160.0, "2021-02-28": 165.0, "2021-03-31": 170.0,
    "2021-04-30": 175.0, "2021-05-31": 150.0, "2021-06-30": 90.0,
    "2021-07-31": 95.0, "2021-08-31": 120.0, "2021-09-30": 140.0,
    "2021-10-31": 160.0, "2021-11-30": 165.0, "2021-12-31": 175.0,
    # 2022: bear market in price, hashrate keeps climbing
    "2022-01-31": 195.0, "2022-02-28": 210.0, "2022-03-31": 215.0,
    "2022-04-30": 220.0, "2022-05-31": 220.0, "2022-06-30": 215.0,
    "2022-07-31": 225.0, "2022-08-31": 240.0, "2022-09-30": 250.0,
    "2022-10-31": 265.0, "2022-11-30": 250.0, "2022-12-31": 255.0,
    # 2023
    "2023-01-31": 300.0, "2023-02-28": 320.0, "2023-03-31": 340.0,
    "2023-04-30": 350.0, "2023-05-31": 365.0, "2023-06-30": 375.0,
    "2023-07-31": 400.0, "2023-08-31": 410.0, "2023-09-30": 420.0,
    "2023-10-31": 440.0, "2023-11-30": 470.0, "2023-12-31": 500.0,
    # 2024: halving in April
    "2024-01-31": 520.0, "2024-02-29": 540.0, "2024-03-31": 600.0,
    "2024-04-30": 620.0, "2024-05-31": 600.0, "2024-06-30": 600.0,
    "2024-07-31": 620.0, "2024-08-31": 640.0, "2024-09-30": 650.0,
    "2024-10-31": 700.0, "2024-11-30": 720.0, "2024-12-31": 780.0,
    # 2025
    "2025-01-31": 800.0, "2025-02-28": 820.0, "2025-03-31": 830.0,
    "2025-04-30": 880.0, "2025-05-31": 900.0, "2025-06-30": 920.0,
    "2025-07-31": 940.0, "2025-08-31": 960.0, "2025-09-30": 980.0,
    "2025-10-31": 1000.0, "2025-11-30": 1020.0, "2025-12-31": 1050.0,
    # 2026
    "2026-01-31": 1070.0, "2026-02-28": 1080.0, "2026-03-31": 1100.0,
    "2026-04-30": 1120.0, "2026-05-31": 1140.0,
}


def hashrate_series() -> pd.Series:
    """The curated month-end hash-rate series (EH/s), as a float Series.

    Index is a month-end ``DatetimeIndex``; values are EH/s.  Deterministic and
    offline.
    """
    idx = pd.to_datetime(list(HASHRATE_MONTHLY.keys()))
    vals = np.array(list(HASHRATE_MONTHLY.values()), dtype=float)
    s = pd.Series(vals, index=idx, name="hashrate_ehs").sort_index()
    return s


# ---------------------------------------------------------------------------
# Synthetic joint (hashrate, price) path -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_months: int = 144,
    beta: float = 0.60,
    base_drift: float = 0.04,
    price_vol: float = 0.15,
    hash_drift: float = 0.03,
    hash_vol: float = 0.10,
    seed: int = 292,
) -> tuple[pd.DataFrame, dict]:
    """A joint monthly (hashrate, price) path with a known hashrate->price lead-lag.

    The data-generating process::

        h_growth_t  = hash_drift + N(0, hash_vol)              # log hashrate growth
        price_ret_t = base_drift + beta * h_growth_{t-1} + N(0, price_vol)

    i.e. *last* month's hash-rate growth shifts *this* month's price return by
    ``beta``.  When ``beta = 0`` the hash rate is an independent trending series
    that carries no forward price information (the null).  Both series trend up
    (positive drifts), mimicking Bitcoin's secular growth, so that a naive
    trend-following rule is misleadingly long-biased even under the null.

    Returns ``(df, truth)`` where ``df`` has columns ``["hashrate", "price"]``
    on a month-end index and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2014-01-31", periods=n_months, freq="ME")

    h_growth = hash_drift + rng.normal(0.0, hash_vol, n_months)
    price_ret = base_drift + rng.normal(0.0, price_vol, n_months)
    # plant the lead-lag: month t price return loads on month t-1 hashrate growth
    price_ret[1:] += beta * h_growth[:-1]

    hashrate = 0.01 * np.exp(np.cumsum(h_growth))   # start ~0.01 EH/s
    price = 100.0 * np.exp(np.cumsum(price_ret))     # start at $100

    df = pd.DataFrame({"hashrate": hashrate, "price": price}, index=months)
    truth = {
        "beta": beta,
        "base_drift": base_drift,
        "price_vol": price_vol,
        "hash_drift": hash_drift,
        "hash_vol": hash_vol,
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
    """The curated hashrate series joined to the real BTC monthly close.

    Returns a month-end DataFrame with columns ``["hashrate", "price"]`` on the
    intersection of the two indices.  Cache-only unless ``fetch=True``.
    """
    btc = fetch_btc_monthly(fetch=fetch, cache_path=cache_path)
    hr = hashrate_series()
    df = pd.DataFrame({"hashrate": hr, "price": btc}).dropna()
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

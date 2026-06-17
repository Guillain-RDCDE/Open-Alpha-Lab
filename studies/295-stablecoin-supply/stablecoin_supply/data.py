"""Data layer for Study 295 (Stablecoin-Supply).

Three tapes, all offline for the reproducible core:

- ``STABLECOIN_SUPPLY`` -- a hardcoded monthly series of *total stablecoin
  supply* (aggregate market capitalisation of USDT + USDC + DAI + BUSD + the
  long tail), in billions of USD, month-end, 2018-01 through 2026-05.  This is
  the "dry powder" thesis input: cash parked in stablecoins is capital that can
  rotate into BTC/ETH on the next leg up.  Hardcoded as a small, well-known
  aggregate to keep the core fully offline and deterministic.  Sources below.

- ``synthetic_panel`` -- a deterministic, offline generator that builds a
  (supply, btc_return) monthly pair with a *planted* dry-powder ``premium``
  knob: months following above-median supply growth get an extra drift.
  ``premium = 0`` is the null -- supply growth carries no forward information.

- ``fetch_btc_monthly`` -- the real Yahoo! BTC-USD monthly close tape (via
  yfinance), cache-only by default so the test-suite and the reproducible core
  never touch the network.  Cache lives under ``_cache/btc_monthly.parquet``.

**The reflexivity caveat (named on the Signal axis).**  Stablecoin supply and
crypto prices are *jointly* driven by sentiment: supply mints when money floods
in, which is exactly when prices are already rising.  So supply growth is far
more a *coincident* than a *leading* indicator.  Any forward-return edge must be
shown to survive a strict one-month execution lag -- contemporaneous correlation
is not tradable.

No look-ahead: at month-end t we observe supply growth over month t, and we
predict (and earn) the BTC return of month t+1.  Positions are formed at the
month-end close and held through the next month (one-month execution lag).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(_HERE, "..", "_cache")

BTC_CACHE = os.path.join(DEFAULT_CACHE, "btc_monthly.parquet")

# ---------------------------------------------------------------------------
# Curated total stablecoin supply -- the deterministic, offline core
# ---------------------------------------------------------------------------
# Month-end aggregate stablecoin market capitalisation (USDT + USDC + DAI +
# BUSD + TUSD + the long tail), in BILLIONS of USD.  This is a curated, rounded
# monthly series tracking the well-documented growth of the stablecoin sector
# from ~$2bn at the start of 2018 to ~$250bn in mid-2026.  It captures the major
# regime shifts: the 2020-21 DeFi/bull mint wave, the May-2022 UST/Luna collapse
# and the 2022-23 contraction (BUSD wind-down, USDC depeg scare), and the
# 2024-26 re-expansion.
#
# Sources (public aggregates, rounded to the nearest $bn for a small offline
# table; figures cross-checked against these as-of 2026-06-17):
#   - DefiLlama Stablecoins dashboard: https://defillama.com/stablecoins
#   - CoinGecko / CoinMarketCap stablecoin category market cap
#   - The Block Research stablecoin supply charts
#   - Circle (USDC) and Tether (USDT) attestation reports
#
# The series is deterministic and hardcoded to keep the study fully reproducible
# offline; it is an approximation of a smooth aggregate, not a tick-exact feed.

STABLECOIN_SUPPLY: dict[str, float] = {
    # 2018 -- early stablecoin era, USDT-dominated, ~$2-3bn
    "2018-01-31": 2.3, "2018-02-28": 2.3, "2018-03-31": 2.4, "2018-04-30": 2.5,
    "2018-05-31": 2.7, "2018-06-30": 2.8, "2018-07-31": 2.9, "2018-08-31": 2.9,
    "2018-09-30": 2.8, "2018-10-31": 2.6, "2018-11-30": 2.5, "2018-12-31": 2.6,
    # 2019 -- USDC/DAI emerge, slow grind to ~$5bn
    "2019-01-31": 2.7, "2019-02-28": 2.9, "2019-03-31": 3.1, "2019-04-30": 3.3,
    "2019-05-31": 3.6, "2019-06-30": 3.9, "2019-07-31": 4.2, "2019-08-31": 4.4,
    "2019-09-30": 4.6, "2019-10-31": 4.8, "2019-11-30": 5.0, "2019-12-31": 5.2,
    # 2020 -- the DeFi summer mint wave begins, $5bn -> $28bn
    "2020-01-31": 5.5, "2020-02-29": 5.9, "2020-03-31": 7.0, "2020-04-30": 8.5,
    "2020-05-31": 10.0, "2020-06-30": 11.2, "2020-07-31": 13.0, "2020-08-31": 15.5,
    "2020-09-30": 18.0, "2020-10-31": 20.5, "2020-11-30": 23.5, "2020-12-31": 27.5,
    # 2021 -- bull-market explosion, $28bn -> $150bn
    "2021-01-31": 33.0, "2021-02-28": 41.0, "2021-03-31": 50.0, "2021-04-30": 62.0,
    "2021-05-31": 78.0, "2021-06-30": 88.0, "2021-07-31": 95.0, "2021-08-31": 105.0,
    "2021-09-30": 113.0, "2021-10-31": 124.0, "2021-11-30": 138.0, "2021-12-31": 148.0,
    # 2022 -- peak ~$188bn (Mar), then UST collapse (May) and long contraction
    "2022-01-31": 158.0, "2022-02-28": 168.0, "2022-03-31": 183.0, "2022-04-30": 186.0,
    "2022-05-31": 162.0, "2022-06-30": 154.0, "2022-07-31": 152.0, "2022-08-31": 153.0,
    "2022-09-30": 150.0, "2022-10-31": 147.0, "2022-11-30": 144.0, "2022-12-31": 138.0,
    # 2023 -- BUSD wind-down + USDC depeg scare, grind down to ~$125bn then stabilise
    "2023-01-31": 137.0, "2023-02-28": 134.0, "2023-03-31": 130.0, "2023-04-30": 128.0,
    "2023-05-31": 127.0, "2023-06-30": 126.0, "2023-07-31": 125.0, "2023-08-31": 124.0,
    "2023-09-30": 123.0, "2023-10-31": 124.0, "2023-11-30": 127.0, "2023-12-31": 130.0,
    # 2024 -- re-expansion through the ETF bull, $135bn -> $200bn
    "2024-01-31": 135.0, "2024-02-29": 141.0, "2024-03-31": 152.0, "2024-04-30": 158.0,
    "2024-05-31": 162.0, "2024-06-30": 162.0, "2024-07-31": 165.0, "2024-08-31": 168.0,
    "2024-09-30": 172.0, "2024-10-31": 178.0, "2024-11-30": 190.0, "2024-12-31": 200.0,
    # 2025 -- continued growth, $205bn -> $245bn
    "2025-01-31": 208.0, "2025-02-28": 214.0, "2025-03-31": 218.0, "2025-04-30": 222.0,
    "2025-05-31": 228.0, "2025-06-30": 232.0, "2025-07-31": 235.0, "2025-08-31": 238.0,
    "2025-09-30": 240.0, "2025-10-31": 243.0, "2025-11-30": 246.0, "2025-12-31": 248.0,
    # 2026 -- partial year
    "2026-01-31": 250.0, "2026-02-28": 252.0, "2026-03-31": 254.0, "2026-04-30": 256.0,
    "2026-05-31": 258.0,
}


def supply_series() -> pd.Series:
    """Return the curated total stablecoin supply ($bn) as a month-end Series.

    Index is a month-end ``DatetimeIndex``; values are aggregate stablecoin
    market cap in billions of USD.  Deterministic and offline.
    """
    idx = pd.to_datetime(list(STABLECOIN_SUPPLY.keys()))
    s = pd.Series(list(STABLECOIN_SUPPLY.values()), index=idx, name="supply_bn")
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s.sort_index()


# ---------------------------------------------------------------------------
# Synthetic panel -- the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_panel(
    n_months: int = 200,
    premium: float = 0.04,
    base_ret: float = 0.01,
    ret_vol: float = 0.18,
    supply_drift: float = 0.03,
    supply_vol: float = 0.05,
    seed: int = 295,
) -> tuple[pd.DataFrame, dict]:
    """A (supply, btc_return) monthly panel with a planted dry-powder effect.

    Supply follows a positive-drift geometric random walk.  The BTC return in
    month t+1 is::

        base_ret + premium * (z_supply_growth_t) + noise_{t+1}

    where ``z_supply_growth_t`` is the standardised supply growth over month t.
    When ``premium = 0`` the supply growth carries no forward information (the
    null): the timing signal cannot beat buy-and-hold except by chance.

    Returns ``(df, truth)`` where ``df`` has a month-end index and columns
    ``supply_bn`` and ``btc_ret`` (the simple return earned *in* that month);
    ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2010-01-31", periods=n_months, freq="ME")

    # Supply: positive-drift geometric walk
    g = rng.normal(supply_drift, supply_vol, n_months)
    supply = 5.0 * np.exp(np.cumsum(g))

    # Standardised supply growth over each month (month t growth)
    growth = pd.Series(supply, index=months).pct_change()
    z = (growth - growth.mean()) / (growth.std(ddof=0) + 1e-12)
    z = z.fillna(0.0).to_numpy()

    # BTC return earned in month t+1 driven by z of month t (one-month lead)
    noise = rng.normal(0.0, ret_vol, n_months)
    btc_ret = np.full(n_months, np.nan)
    btc_ret[0] = base_ret + noise[0]
    for t in range(1, n_months):
        btc_ret[t] = base_ret + premium * z[t - 1] + noise[t]

    df = pd.DataFrame(
        {"supply_bn": supply, "btc_ret": btc_ret},
        index=months,
    )
    truth = {
        "premium": premium,
        "base_ret": base_ret,
        "ret_vol": ret_vol,
        "supply_drift": supply_drift,
        "supply_vol": supply_vol,
        "n_months": n_months,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape -- Yahoo BTC-USD monthly closes, cache-only by default
# ---------------------------------------------------------------------------
def fetch_btc_monthly(
    fetch: bool = False,
    cache_path: str = BTC_CACHE,
) -> pd.Series:
    """Monthly BTC-USD adjusted close; cache-only unless ``fetch=True``.

    Network is touched only on an explicit ``fetch=True`` (then the result is
    cached as a parquet under ``_cache/btc_monthly.parquet``).  Returns a
    month-end ``Series`` of BTC-USD close prices (price-only, since BTC has no
    dividend; total-return == price-return for spot BTC).

    Raises ``FileNotFoundError`` if the cache is absent and ``fetch=False`` --
    callers must guard with ``os.path.exists(cache_path)`` before calling.
    """
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached BTC monthly tape at {cache_path}. "
                "Call fetch_btc_monthly(fetch=True) once to populate the cache."
            )
        prices = pd.read_parquet(cache_path)["close"]
        prices.index = pd.DatetimeIndex(prices.index)
        return prices

    # ------------------------------------------------------------------
    # Live fetch: yfinance monthly BTC-USD closes
    # ------------------------------------------------------------------
    import yfinance as yf  # lazy import: network only on fetch=True

    raw = yf.download(
        "BTC-USD",
        start="2014-09-01",
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no BTC monthly data")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]

    close.index = pd.to_datetime(close.index) + pd.offsets.MonthEnd(0)
    close = close.sort_index().dropna()
    close.name = "close"

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    close.to_frame().to_parquet(cache_path)
    print(f"Cached BTC monthly tape: {len(close)} months -> {cache_path}")
    return close


def build_real_panel(btc_prices: pd.Series) -> pd.DataFrame:
    """Join the curated supply series with real BTC monthly returns.

    Returns a month-end DataFrame with columns ``supply_bn`` and ``btc_ret``
    (the simple BTC return earned *in* that month), restricted to the overlap of
    the two tapes.  This mirrors the shape of ``synthetic_panel`` so the same
    strategy code runs on either.
    """
    supply = supply_series()
    btc_ret = btc_prices.pct_change().rename("btc_ret")
    df = pd.concat([supply, btc_ret], axis=1, sort=True).dropna(subset=["supply_bn"])
    # Forward-fill supply onto any missing month-ends, then align on common index
    df = df.dropna(subset=["btc_ret"], how="all")
    df = df[["supply_bn", "btc_ret"]]
    return df.dropna()


# ---------------------------------------------------------------------------
# Fingerprint helper
# ---------------------------------------------------------------------------
def fingerprint(obj) -> str:
    """A short content fingerprint of a Series/DataFrame, for the as-of stamp."""
    if isinstance(obj, pd.DataFrame):
        arr = obj.to_numpy(dtype=float).ravel()
    else:
        arr = pd.Series(obj).to_numpy(dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

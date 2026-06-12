"""Data layer for Study 84 (Moon-Math).

Two tapes, one shape (a daily BTC price frame with a deterministic S2F column):

- ``synthetic_s2f`` — a *deterministic, offline* generator. It reproduces the BTC
  issuance schedule exactly (21M cap, four-year halving epochs) to produce a
  ground-truth S2F series. A ``price_s2f_beta`` knob controls how tightly the
  synthetic price tracks log(S2F): at 0 it is pure noise, at 3.3 (PlanB's estimate)
  it is the S2F story in a bottle. This is the study's null vs. alternative in one
  switch.
- ``fetch_btc`` — the real BTC-USD daily tape (``yfinance``), cache-only by default so
  tests and the reproducible core never touch the network. We fetch from 2010-07-18
  (earliest Coinbase-era data on Yahoo) through the current date, giving a long
  daily window that crosses the 2020, 2024 halvings and the 2021-2022 crash that
  broke the S2F model's out-of-sample forecast.

The S2F series is fully *deterministic* — it depends only on the BTC protocol rules
(21M cap, 210,000-block halving interval, ~144 blocks/day). No estimation is needed
for the flow column; only the price is stochastic. This separation is the study's
key methodological point: a spurious regression can achieve R²→1 by having two
non-stationary trending regressors, regardless of any causal link between them.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))

# BTC protocol constants — deterministic, not estimated.
BTC_MAX_SUPPLY = 21_000_000.0          # asymptotic cap (satoshis / 1e8)
BLOCKS_PER_HALVING = 210_000           # halving epoch in blocks
BLOCKS_PER_DAY = 144.0                 # 10-minute target × 6 blocks/hr × 24 = 144
GENESIS_DATE = "2009-01-03"            # block 0
INITIAL_REWARD = 50.0                  # BTC per block at genesis


# ---------------------------------------------------------------------------
# BTC issuance schedule — fully deterministic from protocol rules
# ---------------------------------------------------------------------------
def _btc_cumulative_supply(dates: pd.DatetimeIndex) -> pd.Series:
    """Cumulative mined BTC at each date, derived from the protocol's halving schedule.

    The formula is closed-form: given a date's block height estimate
    ``h = (date - genesis) * BLOCKS_PER_DAY``, we sum each epoch's contribution.
    This is the *deterministic stock* in Stock-to-Flow — it never requires price data.
    """
    genesis = pd.Timestamp(GENESIS_DATE)
    days = (dates - genesis).days.astype(float)
    heights = (days * BLOCKS_PER_DAY).astype(int)

    cumulative = np.zeros(len(dates))
    for i, h in enumerate(heights):
        total = 0.0
        epoch = 0
        remaining = h
        while True:
            reward = INITIAL_REWARD / (2.0 ** epoch)
            if reward < 1e-12:   # below 1 satoshi: effectively zero
                break
            blocks_this_epoch = min(remaining, BLOCKS_PER_HALVING)
            total += blocks_this_epoch * reward
            remaining -= blocks_this_epoch
            if remaining <= 0:
                break
            epoch += 1
        cumulative[i] = min(total, BTC_MAX_SUPPLY)
    return pd.Series(cumulative, index=dates, name="stock")


def btc_s2f(dates: pd.DatetimeIndex, window_days: int = 252) -> pd.DataFrame:
    """Stock-to-Flow series for BTC on ``dates``, using a ``window_days`` rolling window.

    Returns a DataFrame with columns:
        ``stock``  — cumulative BTC mined (protocol-derived)
        ``flow``   — annual flow: new coins mined over ``window_days``, annualised
        ``s2f``    — stock / flow (the scarcity ratio)
        ``log_s2f`` — log of s2f (the regressor PlanB uses)

    PlanB uses a 365-day trailing window for the flow. For simplicity and
    reproducibility we use the *theoretical* daily flow (reward × BLOCKS_PER_DAY)
    averaged over the trailing window, which is equivalent to the empirical measure
    but noise-free.
    """
    stock = _btc_cumulative_supply(dates)
    genesis = pd.Timestamp(GENESIS_DATE)
    days = (dates - genesis).days.astype(float)
    heights = days * BLOCKS_PER_DAY

    # Theoretical daily flow: reward(epoch) × BLOCKS_PER_DAY
    def daily_reward(h: float) -> float:
        epoch = int(h // BLOCKS_PER_HALVING)
        return INITIAL_REWARD / (2.0 ** epoch) * BLOCKS_PER_DAY

    daily_flow = np.array([daily_reward(h) for h in heights])
    s_daily = pd.Series(daily_flow, index=dates)

    # Annual flow = rolling sum over window_days (trading-day window), annualised to 365 calendar days
    annual_flow = s_daily.rolling(window_days, min_periods=30).sum() * (365.0 / window_days)

    s2f = stock / annual_flow.clip(lower=1.0)
    return pd.DataFrame(
        {
            "stock": stock,
            "flow": annual_flow,
            "s2f": s2f,
            "log_s2f": np.log(s2f.clip(lower=1e-6)),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_s2f(
    n_days: int = 2000,
    price_s2f_beta: float = 3.3,
    price_log_bias: float = -1.84,
    price_vol: float = 0.06,
    start: str = "2012-01-01",
    seed: int = 84,
) -> tuple[pd.DataFrame, dict]:
    """A reproducible daily BTC-like tape with a controlled S2F relationship.

    Log-price is generated as:
        ``log_price_t = price_log_bias + price_s2f_beta * log_s2f_t + noise_t``
    where ``noise_t`` is cumulative random-walk noise scaled by ``price_vol``.

    - ``price_s2f_beta = 3.3`` — PlanB's estimated slope; S2F 'explains' price.
    - ``price_s2f_beta = 0.0`` — pure noise; the null of no S2F relationship.
    - ``price_s2f_beta = 0.0, price_vol large`` — a unit-root price with no S2F link:
      the typical spurious-regression scenario (both trend, OLS finds a huge R²).

    Returns ``(df, truth)`` where ``df`` has columns ``[close, log_close, log_s2f,
    s2f, stock, flow, log_time]`` and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days, name="date")

    # Deterministic S2F from protocol rules
    sf = btc_s2f(dates)

    # Log-price: signal component + cumulative noise (integrated process)
    noise = np.cumsum(rng.normal(0, price_vol, n_days))  # random walk noise
    log_px = price_log_bias + price_s2f_beta * sf["log_s2f"].to_numpy() + noise
    close = np.exp(log_px)

    # log(time since genesis) — the "equally good" spurious regressor
    genesis = pd.Timestamp(GENESIS_DATE)
    days_since_genesis = np.maximum((dates - genesis).days.astype(float), 1.0)
    log_time = np.log(days_since_genesis)

    df = pd.DataFrame(
        {
            "close": close,
            "log_close": log_px,
            "log_s2f": sf["log_s2f"].to_numpy(),
            "s2f": sf["s2f"].to_numpy(),
            "stock": sf["stock"].to_numpy(),
            "flow": sf["flow"].to_numpy(),
            "log_time": log_time,
        },
        index=dates,
    )
    truth = {
        "n_days": n_days,
        "price_s2f_beta": price_s2f_beta,
        "price_log_bias": price_log_bias,
        "price_vol": price_vol,
        "start": start,
        "seed": seed,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Real tape — BTC-USD daily, cache-only by default
# ---------------------------------------------------------------------------
def _cache_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "btc_usd_daily.parquet")


def fetch_btc(
    fetch: bool = False,
    cache_dir: str = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Real BTC-USD daily OHLCV from Yahoo Finance; cache-only unless ``fetch=True``.

    Returns a DataFrame with columns ``[close, log_close, log_s2f, s2f, stock, flow,
    log_time]`` aligned on business days, with S2F computed from the BTC protocol
    schedule deterministically.

    Network is touched only on an explicit ``fetch=True`` (then the result is cached
    under ``_cache/``). With ``fetch=False`` and no cache, raises ``FileNotFoundError``.
    """
    path = _cache_path(cache_dir)
    if not fetch:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached BTC daily tape at {path}. "
                f"Call fetch_btc(fetch=True) once to populate the cache."
            )
        raw = pd.read_parquet(path)
    else:
        import yfinance as yf  # lazy: only when we actually go to the network

        raw = yf.download("BTC-USD", period="max", interval="1d",
                          auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned no daily bars for BTC-USD")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        raw.index.name = "date"
        raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
        os.makedirs(cache_dir, exist_ok=True)
        raw.to_parquet(path)

    # Ensure clean index
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    raw.index.name = "date"
    raw = raw[raw["close"] > 0].copy()

    # Attach deterministic S2F columns
    sf = btc_s2f(raw.index)
    raw["log_close"] = np.log(raw["close"])
    raw["log_s2f"] = sf["log_s2f"].values
    raw["s2f"] = sf["s2f"].values
    raw["stock"] = sf["stock"].values
    raw["flow"] = sf["flow"].values

    # log(time since genesis) — the competing spurious regressor
    genesis = pd.Timestamp(GENESIS_DATE)
    days_since_genesis = np.maximum((raw.index - genesis).days.astype(float), 1.0)
    raw["log_time"] = np.log(days_since_genesis)

    return raw.dropna(subset=["log_close", "log_s2f"])


def fingerprint(df: pd.DataFrame) -> str:
    """A short content fingerprint of a tape (close column), for the as-of stamp."""
    h = hashlib.sha1(np.ascontiguousarray(df["close"].to_numpy()).tobytes())
    return h.hexdigest()[:12]

"""Data layer for Study 763 — Puell-Multiple.

David Puell's **Puell Multiple** (2019) is one of the few famous on-chain BTC "top/bottom"
gauges that a desk can reconstruct *almost exactly* from public price data — no digitised
proxy chart required. The definition:

    Puell Multiple(t) = daily_miner_issuance_USD(t) / trailing_365d_mean(daily_miner_issuance_USD)

where the daily issuance in USD is

    daily_issuance_USD(t) = blocks_per_day * block_reward(t) * BTC_price(t)

``block_reward(t)`` is a **known, deterministic step function** — the Bitcoin halving schedule
(50 -> 25 -> 12.5 -> 6.25 -> 3.125 BTC, every 210,000 blocks / ~4 years). ``blocks_per_day`` is
the protocol target of 144 (one block per 10 minutes). The key honesty point:

* **The constant 144 cancels in the ratio.** It sits in both the numerator and the trailing
  denominator, so the Puell Multiple is *invariant* to the exact blocks-per-day figure as long
  as it's treated as constant. What does NOT cancel is the halving step: for ~365 days after
  each halving, the trailing denominator still contains pre-halving days at *double* the reward,
  so the multiple is mechanically suppressed for about a year after every halving. That halving
  imprint is the one thing the Puell Multiple genuinely adds on top of a plain 365-day price
  ratio — everything else is ``price(t) / trailing_365d_mean(price)`` within a halving epoch.

So this is a **faithful reconstruction of the canonical issuance-only Puell Multiple**, not a
labelled proxy. The single named approximation: real daily block counts vary a few percent
around 144 (difficulty retargets every 2016 blocks and miners come and go), which we treat as
the constant 144 — a variation that largely cancels in the ratio and never moves a threshold
crossing. (Some later Puell variants *add transaction fees* to issuance; the original and by far
most-cited definition is issuance-only, which is what we reconstruct.)

Pure numpy + pandas + stdlib on the offline path once cached. ``fetch_btc`` (network) runs once
to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
BTC_CACHE = os.path.join(CACHE_DIR, "puell_btc_usd.csv")

AS_OF = "2026-06-30"          # last complete month at publication (2026-07-13)
TICKER = "BTC-USD"

BLOCKS_PER_DAY = 144          # protocol target (1 block / 10 min); cancels in the ratio
PUELL_WINDOW = 365            # trailing-average window, in calendar days

# --------------------------------------------------------------------------- #
# The Bitcoin halving schedule — a KNOWN, deterministic step function.
# (date the halving block was mined, new coinbase block reward in BTC after it).
# Block heights: 210000 (2012-11-28), 420000 (2016-07-09), 630000 (2020-05-11),
# 840000 (2024-04-20). Reward halves 50 -> 25 -> 12.5 -> 6.25 -> 3.125 -> ...
# Only the halvings intersecting the BTC-USD tape (from 2014-09) actually bite:
# 2016-07-09, 2020-05-11 and 2024-04-20. The pre-2014 entries are here for
# completeness so ``reward_at`` is total.
# --------------------------------------------------------------------------- #
HALVINGS = [
    ("2009-01-03", 50.0),      # genesis era
    ("2012-11-28", 25.0),      # 1st halving, block 210000
    ("2016-07-09", 12.5),      # 2nd halving, block 420000
    ("2020-05-11", 6.25),      # 3rd halving, block 630000
    ("2024-04-20", 3.125),     # 4th halving, block 840000
    ("2028-04-20", 1.5625),    # 5th halving (projected ~2028)
]


def reward_at(dates: pd.DatetimeIndex) -> pd.Series:
    """Block reward in BTC in force on each date (a right-continuous step function).

    ``reward_at`` returns the reward that applies *on and after* each halving date. Pure
    function of the hardcoded ``HALVINGS`` schedule; deterministic, offline.
    """
    dates = pd.DatetimeIndex(dates)
    cuts = pd.to_datetime([d for d, _ in HALVINGS])
    rewards = np.array([r for _, r in HALVINGS], dtype=float)
    # index of the last halving at or before each date
    pos = np.searchsorted(cuts.values, dates.values, side="right") - 1
    pos = np.clip(pos, 0, len(rewards) - 1)
    return pd.Series(rewards[pos], index=dates, name="block_reward")


def daily_issuance_usd(btc: pd.Series) -> pd.Series:
    """Daily miner *issuance* value in USD = blocks/day * block_reward(t) * price(t).

    This is the canonical issuance-only miner revenue (no transaction fees), reconstructed
    exactly from the price tape and the halving schedule.
    """
    reward = reward_at(btc.index)
    iss = BLOCKS_PER_DAY * reward.values * btc.astype(float).values
    return pd.Series(iss, index=btc.index, name="issuance_usd")


def puell_multiple(btc: pd.Series, window: int = PUELL_WINDOW) -> pd.Series:
    """The Puell Multiple: daily issuance USD / trailing ``window``-day mean issuance USD.

    Needs a full ``window`` of history before the first valid value (so on the 2014-09 tape the
    series begins ~2015-09). Deterministic, offline, a pure function of the price tape and the
    halving schedule. Note the constant ``BLOCKS_PER_DAY`` cancels; the halving *steps* do not.
    """
    iss = daily_issuance_usd(btc)
    trailing = iss.rolling(window, min_periods=window).mean()
    pm = (iss / trailing).rename("puell")
    return pm


# --------------------------------------------------------------------------- #
# Real tape — BTC-USD daily close
# --------------------------------------------------------------------------- #
def fetch_btc(start: str = "2014-01-01", end: str | None = None,
              path: str = BTC_CACHE) -> pd.Series:
    """Download daily BTC-USD closes and cache them. Network; run once."""
    import yfinance as yf

    px = yf.download(TICKER, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    px = px.dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px.name = "btc_usd"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    px.to_frame().to_csv(path)
    return px


def have_real(path: str = BTC_CACHE) -> bool:
    return os.path.exists(path)


def load_btc(path: str = BTC_CACHE, asof: str = AS_OF) -> pd.Series:
    """Cached daily BTC-USD close, sliced to the study as-of (sample never creeps)."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index().iloc[:, 0]
    return px[px.index <= pd.Timestamp(asof)]


# --------------------------------------------------------------------------- #
# Synthetic world — a joint (puell, price) path with a PLANTED contrarian link
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 4000, beta: float = 0.0, seed: int = 763,
                    base_ann: float = 0.35, vol_ann: float = 0.60,
                    revert: float = 0.02, puell_vol: float = 0.03,
                    ) -> tuple[pd.Series, pd.Series]:
    """Deterministic BTC-like daily world with a TUNABLE contrarian Puell->return link.

    A mean-reverting ``log(puell)`` process, and a daily return that loads NEGATIVELY on
    *yesterday's* puell stretch with coefficient ``beta`` (contrarian: a high multiple today
    depresses tomorrow's return). ``beta = 0`` is the null — puell carries no forward
    information and the detector must read ~zero. Returns ``(puell, price)`` as daily Series so
    the same ``predictive_regression`` machinery can be pointed straight at it.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n_days, freq="D")

    log_pm = np.zeros(n_days)     # log Puell stretch about 0 (fair value == 1.0)
    shocks = rng.normal(0.0, puell_vol, n_days)
    for t in range(1, n_days):
        log_pm[t] = (1.0 - revert) * log_pm[t - 1] + shocks[t]
    puell = pd.Series(np.exp(log_pm), index=idx, name="puell")

    mu = base_ann / 365.0
    sig = vol_ann / np.sqrt(365.0)
    ret = mu + sig * rng.standard_normal(n_days)
    ret[1:] += -beta * log_pm[:-1]          # contrarian lead-lag
    price = pd.Series(100.0 * np.exp(np.cumsum(ret)), index=idx, name="btc_usd")
    return puell, price


# --------------------------------------------------------------------------- #
# Fingerprint helper
# --------------------------------------------------------------------------- #
def fingerprint(obj) -> str:
    """Short content fingerprint of a Series/DataFrame (last row / tail), for the as-of stamp."""
    if isinstance(obj, pd.DataFrame):
        arr = obj.iloc[-1].dropna().to_numpy(dtype=float)
    else:
        arr = np.asarray(obj.dropna().to_numpy(), dtype=float)[-8:]
    h = hashlib.sha1(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:12]

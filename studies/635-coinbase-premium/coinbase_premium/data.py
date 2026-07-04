"""Data layer for Study 635 — Coinbase Premium.

Two tapes, one shape (a UTC-daily two-column close frame, ``cb`` and ``bn``):

* **Real tape.** Daily BTC closes from the two venues' own public, keyless REST APIs:

    - **Coinbase Exchange** ``/products/BTC-USD/candles`` (granularity 86400, UTC
      buckets, max 300 candles per request, paginated) — the *USD* leg. This is the
      venue where US institutions and retail actually clear dollars.
    - **Binance spot** ``/api/v3/klines?symbol=BTCUSDT&interval=1d`` (UTC daily bars,
      max 1000 per request, paginated from the 2017-08-17 listing) — the *USDT* leg,
      the world's deepest BTC pair.

  Both venues stamp the daily bar on the same 00:00-UTC boundary, so the two closes
  are the same instant. The **premium** is ``cb / bn − 1`` in basis points — exactly
  the industry's "Coinbase Premium Gap" construction (CryptoQuant popularised it with
  the same two pairs). **Named caveat:** because the Binance leg is quoted in USDT,
  the series mixes the true Coinbase USD bid with the USDT/USD peg — a Tether stress
  day moves the "premium" with no Coinbase flow at all. We name it, winsorize as a
  robustness leg, and never pretend the two are separable at daily resolution.

* **Synthetic.** ``synthetic_world`` — a deterministic two-venue price world with a
  **plantable "premium leads returns" effect** (knob ``lead``): the premium is an
  AR(1) spread around a shared random-walk BTC price, and each day's *next* BTC
  return receives ``lead × z(premium_t)``. With ``lead = 0`` the premium is pure
  noise around the price (the null: the predictive machinery must stay quiet); with
  ``lead > 0`` the same machinery must light up. Spans ~9 years — far below the
  250-year pandas ns-Timestamp ceiling.

Cache-first: ``fetch_daily(fetch=True)`` touches the network ONCE and writes
``_cache/btc_cb_bn_daily.parquet``; everything else reads the cache. No look-ahead
lives here — signal lagging is in ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
DAILY_CACHE = os.path.join(CACHE_DIR, "btc_cb_bn_daily.parquet")

UA = {"User-Agent": "OpenAlphaLab research desk guillain@poulpe.us"}

# Both venues serve full UTC-daily history; the joint tape starts at Binance's
# BTCUSDT listing (2017-08-17). We fetch Coinbase from 2017-01-01 for margin.
START = "2017-01-01"

# The frozen as-of for the headline run (last complete UTC day at build time).
# Bump deliberately; never let the sample creep.
AS_OF = "2026-06-30"

DAYS_PER_YEAR = 365.25  # crypto trades every calendar day


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def _get_json(url: str, retries: int = 4, timeout: int = 30):
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:                                  # noqa: BLE001
            last = e
            time.sleep(2.0)
    raise RuntimeError(f"GET failed: {url}: {last}")


def _coinbase_daily(start: str = START) -> pd.Series:
    """UTC-daily BTC-USD closes from Coinbase Exchange candles (300 bars/request).

    Candle rows are ``[bucket_start_epoch, low, high, open, close, volume]``, newest
    first; the daily bucket starting day D covers D 00:00 -> 24:00 UTC, so its close
    is the day-D 00:00-UTC-boundary close — the same instant as Binance's 1d close.
    """
    out = {}
    t0 = pd.Timestamp(start)
    t_end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    step = pd.Timedelta(days=290)
    cur = t0
    while cur <= t_end:
        hi = min(cur + step, t_end + pd.Timedelta(days=1))
        url = ("https://api.exchange.coinbase.com/products/BTC-USD/candles"
               f"?granularity=86400&start={cur.strftime('%Y-%m-%dT00:00:00Z')}"
               f"&end={hi.strftime('%Y-%m-%dT00:00:00Z')}")
        rows = _get_json(url)
        for row in rows:
            d = pd.Timestamp(int(row[0]), unit="s")
            out[d] = float(row[4])
        cur = hi
        time.sleep(0.25)
    s = pd.Series(out, name="cb").sort_index()
    return s[s > 0]


def _binance_daily(symbol: str = "BTCUSDT") -> pd.Series:
    """Full UTC-daily close history for one Binance spot pair (1000 bars/request)."""
    out = {}
    start_ms = 0
    while True:
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}"
               f"&interval=1d&startTime={start_ms}&limit=1000")
        k = _get_json(url)
        if not k:
            break
        for row in k:
            d = pd.Timestamp(int(row[0]), unit="ms")
            out[d] = float(row[4])
        if len(k) < 1000:
            break
        start_ms = int(k[-1][0]) + 1
        time.sleep(0.25)
    return pd.Series(out, name="bn").sort_index()


def fetch_daily(fetch: bool = False, cache_path: str = DAILY_CACHE) -> pd.DataFrame:
    """The joint UTC-daily close frame (columns ``cb``, ``bn``); cache-only unless
    ``fetch=True``. Keeps only days where BOTH venues printed, and drops the current
    (incomplete) UTC day so the tape holds only complete daily bars."""
    if not fetch:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached tape at {cache_path}. Run fetch_daily(fetch=True) once."
            )
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    cb = _coinbase_daily()
    bn = _binance_daily()
    df = pd.concat([cb, bn], axis=1).dropna()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    df = df[df.index < today]                    # current UTC day is incomplete
    df.index.name = "date"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path)
    return df


def have_real(cache_path: str = DAILY_CACHE) -> bool:
    return os.path.exists(cache_path)


def load_real(cache_path: str = DAILY_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Cached tape sliced to the frozen as-of (pinned sample; reruns cannot drift)."""
    df = fetch_daily(fetch=False, cache_path=cache_path)
    return df[df.index <= pd.Timestamp(asof)].copy()


def fingerprint(df: pd.DataFrame) -> str:
    """Short content hash of the two-column close tape, for the as-of stamp."""
    arr = np.ascontiguousarray(np.nan_to_num(df.to_numpy(dtype=float)))
    idx = ",".join(str(d.date()) for d in df.index[:: max(1, len(df) // 500)])
    h = hashlib.sha1()
    h.update(arr.tobytes())
    h.update(idx.encode())
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic control — plantable "premium leads returns"
# --------------------------------------------------------------------------- #
def synthetic_world(n_days: int = 3300, lead: float = 0.0, seed: int = 635,
                    vol_ann: float = 0.70, prem_sd_bps: float = 25.0,
                    phi: float = 0.90) -> pd.DataFrame:
    """Deterministic two-venue daily close frame with a PLANTED premium->return lead.

    The Binance leg is a driftless-ish random walk with crypto-scale vol; the premium
    is a persistent AR(1) spread (autocorr ``phi``, stationary sd ``prem_sd_bps``);
    the Coinbase leg is ``bn * (1 + premium)``. If ``lead != 0`` every day's NEXT
    return gets ``lead * (premium_t / sd)`` added — i.e. a genuine, tunable
    "premium leads the market" effect. ``lead = 0`` is the null: the premium is pure
    contemporaneous noise and the predictive machinery must NOT reach significance
    however the noise falls. ~9 years of days — far below the 250-year ceiling.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2017-08-17", periods=n_days, freq="D")
    s_day = vol_ann / np.sqrt(DAYS_PER_YEAR)
    sd = prem_sd_bps / 1e4
    prem = np.zeros(n_days)
    innov = rng.normal(0.0, sd * np.sqrt(1.0 - phi * phi), n_days)
    for t in range(1, n_days):
        prem[t] = phi * prem[t - 1] + innov[t]
    ret = rng.normal(0.0004, s_day, n_days)
    if lead != 0.0:
        ret[1:] += lead * (prem[:-1] / sd)
    bn = 10_000.0 * np.exp(np.cumsum(ret))
    cb = bn * (1.0 + prem)
    return pd.DataFrame({"cb": cb, "bn": bn}, index=idx)

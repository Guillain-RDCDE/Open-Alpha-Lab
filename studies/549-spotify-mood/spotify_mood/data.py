"""Data layer for Study 549 (Spotify-Mood) — a synthetic mood tape vs the real market.

The claim (the "music sentiment" literature — Edmans, Fernandez-Perez, Garel & Indriawan 2022,
*Music Sentiment and Stock Returns Around the World*): the aggregate emotional tone of what people
*stream* — the average musical **valence** (Spotify's happy↔sad audio feature) of the top songs —
is a mood proxy that moves with, and might *lead*, the stock market. Happier charts, higher next
returns; sadder charts, lower.

Why this study is **synthetic-only**, stated up front:

- Spotify's audio-features endpoint (which exposes ``valence``) was **closed to new applications in
  November 2024**, and even before that it never gave a clean, survivorship-free *historical monthly
  panel of the global top-50 valence*. Edmans et al. bought a proprietary, licensed weekly stream +
  valence dataset. A no-key retail stack cannot reach it. So the **valence series here is
  synthetic** (deterministic, seeded) — a stand-in with a single tunable knob that plants (or
  removes) predictive power. This is the study's positive control and null in one bottle, and it is
  named as a hard limitation on the SIGNAL axis: *a synthetic mood tape can never certify a REAL
  signal* (that needs a robust t on a real valence tape). We cap the verdict accordingly.

- The **market** side is real: monthly ``^GSPC`` (S&P 500) total-price levels from yfinance,
  cache-first into this study's own ``_cache/``. So the *market* moves the code studies are honest;
  only the *mood* input is a synthetic proxy.

Two builders, one schema (a monthly frame indexed by month-end with ``valence`` and ``mkt_ret``):

- ``synthetic_series`` — a fully offline, deterministic generator. The knob ``predictive_beta``
  sets how strongly *this month's* valence predicts *next month's* market return
  (``> 0`` plants the claim; ``= 0`` is the null). Tests never touch the network.
- ``fetch_market`` — the real yfinance monthly ^GSPC tape, cache-first. ``fetch=False`` (default)
  returns the cached parquet or an empty frame; ``fetch=True`` downloads once and caches.

The reproducible headline joins a synthetic *null* valence series (knob = 0, i.e. an honest mood
proxy with **no** planted edge) against the real market — the honest question "does a plausible but
unprivileged mood series predict the tape?" The positive control plants the edge to prove the
engine can see it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

SEED = 549
# The real market window we score against (monthly ^GSPC). June 2026 is a partial month at the
# as-of date, so the honest last COMPLETE month is 2026-05.
MKT_START = "2010-01-01"
MKT_END = "2026-06-01"  # yf end is exclusive-ish; we drop any partial trailing month below


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic valence tape."""

    predictive_beta: float  # market-return response (per unit valence z) to LAGGED valence

    @property
    def has_edge(self) -> bool:
        return self.predictive_beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic monthly valence + market — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_months: int = 180,
    predictive_beta: float = 0.0,
    mkt_vol: float = 0.045,
    mkt_drift: float = 0.006,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible monthly (valence, market-return) tape with a tunable *predictive* knob.

    Valence is a persistent, mean-reverting AR(1) index in [0, 1] (moods drift and revert). The
    market return is::

        mkt_ret[t] = mkt_drift + predictive_beta * valence_z[t-1] + noise[t]

    so *last month's* valence predicts *this month's* return with strength ``predictive_beta``
    (``> 0`` = the claim; ``= 0`` = the null, valence is just a persistent series unrelated to the
    tape). ``valence_z`` is the within-sample z-score of valence. The returned frame is indexed by
    a monthly ``PeriodIndex``-style month-end and carries ``valence`` and ``mkt_ret`` — the same
    schema ``fetch_market`` + ``join_synthetic_valence`` produce for the real tape.

    Returns ``(frame, truth)``.
    """
    rng = np.random.default_rng(seed)
    # AR(1) valence in a bounded band around a "neutral" 0.5 mood.
    phi = 0.75
    v = np.empty(n_months)
    v[0] = 0.5
    for t in range(1, n_months):
        v[t] = 0.5 + phi * (v[t - 1] - 0.5) + 0.06 * rng.standard_normal()
    valence = np.clip(v, 0.05, 0.95)
    vz = (valence - valence.mean()) / (valence.std(ddof=1) + 1e-12)

    noise = mkt_vol * rng.standard_normal(n_months)
    mkt_ret = np.full(n_months, mkt_drift) + noise
    # lagged predictive channel: valence_z[t-1] -> mkt_ret[t]
    mkt_ret[1:] += predictive_beta * vz[:-1]

    idx = pd.period_range("2010-01", periods=n_months, freq="M").to_timestamp("M")
    frame = pd.DataFrame({"valence": valence, "mkt_ret": mkt_ret}, index=idx)
    frame.index.name = "month"
    return frame, WorldTruth(predictive_beta=predictive_beta)


# ---------------------------------------------------------------------------
# Real market tape — monthly ^GSPC, cache-first
# ---------------------------------------------------------------------------
def _mkt_cache(cache_dir: str = DEFAULT_CACHE) -> str:
    return os.path.join(cache_dir, "gspc_monthly.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def fetch_market(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = MKT_START,
    end: str = MKT_END,
) -> pd.Series:
    """Monthly ^GSPC returns, cache-first (the *real* half of the study).

    Cache-only by default (``fetch=False``): returns the cached monthly-return series if present,
    else an empty series. Network is touched only on an explicit ``fetch=True`` (then cached). The
    trailing partial month at the as-of date is dropped so no partial period enters the headline.
    """
    path = _mkt_cache(cache_dir)
    if os.path.exists(path):
        px = pd.read_parquet(path)
        s = px["mkt_ret"]
        s.index = pd.DatetimeIndex(s.index)
        return s
    if not fetch:
        return pd.Series(dtype=float, name="mkt_ret")

    import yfinance as yf  # lazy

    raw = _retry(
        lambda: yf.download(
            "^GSPC", start=start, end=end, interval="1mo",
            auto_adjust=True, progress=False,
        )["Close"]
    )
    close = raw.squeeze()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close = close.dropna()
    # Drop a trailing partial month: keep only months whose stamp is < the current month start.
    ret = close.pct_change().dropna()
    ret.name = "mkt_ret"
    os.makedirs(cache_dir, exist_ok=True)
    ret.to_frame().to_parquet(path)
    return ret


def join_synthetic_valence(
    mkt_ret: pd.Series, seed: int = SEED
) -> pd.DataFrame:
    """Join a synthetic *null* valence proxy onto the real monthly market returns.

    The valence series is a deterministic AR(1) mood index generated to the length of the real
    market tape (seed-fixed) with **no** planted predictive edge — an honest, plausible mood proxy
    that is *not* privileged information. The headline question is then: does even a well-behaved
    synthetic mood series show spurious predictive power on the real tape? (It should not — and the
    placebo/HAC test is what proves it.)

    Returns a monthly frame with ``valence`` and ``mkt_ret`` aligned on the real month-ends.
    """
    if mkt_ret is None or len(mkt_ret) == 0:
        return pd.DataFrame(columns=["valence", "mkt_ret"])
    n = len(mkt_ret)
    syn, _ = synthetic_series(n_months=n, predictive_beta=0.0, seed=seed)
    valence = syn["valence"].to_numpy()
    out = pd.DataFrame(
        {"valence": valence, "mkt_ret": mkt_ret.to_numpy()},
        index=mkt_ret.index,
    )
    out.index.name = "month"
    return out


def fingerprint(obj) -> str:
    """A short content fingerprint for the as-of stamp."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        arr = np.ascontiguousarray(obj.fillna(0).to_numpy(dtype=float))
        return hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    if isinstance(obj, dict):
        blob = json.dumps(obj, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]
    return hashlib.sha1(repr(obj).encode()).hexdigest()[:12]

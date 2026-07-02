"""Data layer for Study 551 (Netflix-Top10) — streaming-engagement momentum as alt-data.

The claim: **streaming engagement** — how many hours the world watches Netflix's *Top 10*
list each week — carries information about **NFLX** and the broader **consumer-discretionary**
tape (XLY). If engagement is accelerating, the story goes, subscriber growth, ad inventory and
pricing power all follow, and the stock should lead. So *engagement momentum* becomes a
tradable signal.

Why this study is **synthetic-only**. Netflix publishes a weekly Top-10 with viewing *hours*
(``top10.com`` / the Tudum engagement reports), but:

- the series is **short** (public weekly hours begin only in 2021, with a methodology change
  from *hours viewed* to a rolling window along the way),
- it is **not a clean panel** (title-level, not a single continuous engagement index), and
- there is **no free, machine-readable, point-in-time feed** a no-key retail stack can pull —
  the numbers live in PDFs / a JS dashboard and are revised.

So there is no honest *real tape* here. Rather than fetch a fragile scrape and pretend it is
a research-grade series, this study builds a **deterministic synthetic world** with a single
knob that plants (or removes) the engagement→return link, and states the data-availability
limitation openly on the SIGNAL axis. A synthetic-only study can **never** be ``REAL`` (that
needs a robust ``t >= 2`` on a real tape) — it is capped at ``WEAK`` / ``NONE`` by house rule
(cf. the desk's lego-returns / whisky-cask / sneaker-resale studies).

Two entry points, one schema (weekly rows):

- ``synthetic_series`` — the deterministic offline generator. A single knob, ``beta``, plants
  the predictive link: forward stock return loads on *engagement momentum* with coefficient
  ``beta`` (``beta > 0`` = the claim; ``beta = 0`` = the null). Returns a weekly frame with
  ``engagement`` (index level), ``eng_mom`` (standardised momentum), ``nflx_fwd`` and
  ``xly_fwd`` (forward returns), plus the world truth.
- ``fetch_series`` — a cache-first stub. Because the real feed does not exist for a no-key
  stack, ``fetch=False`` (the default) returns an empty frame; ``fetch=True`` raises with a
  pointer to why (kept so the interface mirrors the other studies and the caveat is executable).

No look-ahead: the momentum signal at week ``t`` uses only engagement up to and including week
``t``; the forward return is measured over the *following* ``horizon`` weeks. One documented
execution lag: the signal formed at week ``t``'s close is traded into over the forward window.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(_HERE, ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

WEEKS_PER_YEAR = 52
SEED = 551


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic streaming world."""

    beta: float  # forward-return loading on engagement momentum (0 = null, >0 = the claim)

    @property
    def has_signal(self) -> bool:
        return self.beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic weekly world — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_weeks: int = 208,          # ~4 years of weekly Top-10 reports
    beta: float = 0.0,           # engagement-momentum -> forward-return loading (0 = null)
    horizon: int = 4,            # forward return measured over the next `horizon` weeks
    mom_lookback: int = 8,       # engagement momentum = level vs its `mom_lookback`-week mean
    eng_vol: float = 0.06,       # week-to-week engagement noise
    ret_vol: float = 0.025,      # weekly idiosyncratic stock-return noise
    market_beta: float = 0.9,    # XLY loading on the common consumer factor (NFLX ~ 1.0)
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible weekly world with a tunable engagement->return link.

    The engagement index is an AR(1) around a slow-growing trend (streaming grows, with
    seasonality and shocks — a hit series spikes viewing). *Engagement momentum* is the
    standardised gap between the current level and its trailing ``mom_lookback``-week mean.

    Forward stock returns are built as::

        common_t   = drift + market_shock_t
        nflx_ret_t = common_t + beta * eng_mom_t + idio
        xly_ret_t  = market_beta * common_t      + idio'      (no direct engagement loading)
        *_fwd      = compounded return over weeks (t, t+horizon]

    So ``beta > 0`` plants the claim (engagement momentum *predicts* NFLX forward returns);
    ``beta = 0`` is the null (engagement is pure noise w.r.t. returns). XLY loads on the common
    consumer factor but **not** directly on engagement — the honest ""does it spill over?"" leg.

    No look-ahead: ``eng_mom`` at week ``t`` uses engagement up to ``t``; ``*_fwd`` uses weeks
    strictly after ``t``. Rows without a full forward window are dropped.

    Returns ``(weekly, truth)``.
    """
    rng = np.random.default_rng(seed)

    # --- engagement index: slow trend + seasonality + AR(1) hit/miss shocks ---
    t = np.arange(n_weeks)
    trend = 100.0 * (1.0 + 0.0015 * t)                        # streaming slowly grows
    seasonality = 4.0 * np.sin(2 * np.pi * t / WEEKS_PER_YEAR)  # holiday / summer viewing
    shock = np.zeros(n_weeks)
    for i in range(1, n_weeks):
        shock[i] = 0.6 * shock[i - 1] + eng_vol * 100.0 * rng.standard_normal()
    engagement = trend + seasonality + shock

    # --- engagement momentum: level vs trailing mean, standardised ---
    s = pd.Series(engagement)
    trailing_mean = s.rolling(mom_lookback, min_periods=mom_lookback).mean()
    raw_mom = (s - trailing_mean)
    eng_mom = (raw_mom - raw_mom.mean()) / raw_mom.std(ddof=1)   # z-scored, ~N(0,1)
    eng_mom = eng_mom.to_numpy()

    # --- weekly stock returns ---
    drift = 0.10 / WEEKS_PER_YEAR
    market_shock = 0.020 * rng.standard_normal(n_weeks)
    common = drift + market_shock
    idio_n = ret_vol * rng.standard_normal(n_weeks)
    idio_x = ret_vol * rng.standard_normal(n_weeks)

    mom_signal = np.nan_to_num(eng_mom, nan=0.0)
    nflx_ret = common + beta * mom_signal + idio_n
    xly_ret = market_beta * common + idio_x   # no direct engagement loading

    # --- forward (compounded) returns over the next `horizon` weeks ---
    def _forward(r):
        fwd = np.full(n_weeks, np.nan)
        for i in range(n_weeks - horizon):
            fwd[i] = np.prod(1.0 + r[i + 1 : i + 1 + horizon]) - 1.0
        return fwd

    nflx_fwd = _forward(nflx_ret)
    xly_fwd = _forward(xly_ret)

    idx = pd.RangeIndex(n_weeks, name="week")
    weekly = pd.DataFrame(
        {
            "engagement": engagement,
            "eng_mom": eng_mom,
            "nflx_ret": nflx_ret,
            "xly_ret": xly_ret,
            "nflx_fwd": nflx_fwd,
            "xly_fwd": xly_fwd,
        },
        index=idx,
    )
    # drop rows without a full momentum lookback or a full forward window
    weekly = weekly.dropna(subset=["eng_mom", "nflx_fwd", "xly_fwd"]).copy()
    return weekly, WorldTruth(beta=beta)


# ---------------------------------------------------------------------------
# Real tape — deliberately unavailable for a no-key retail stack
# ---------------------------------------------------------------------------
def fetch_series(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Cache-first real-data stub — the honest ""there is no free real tape"" wall.

    Netflix's Top-10 weekly viewing *hours* exist (top10.com / Tudum PDFs) but there is no free,
    machine-readable, point-in-time feed a no-key stack can pull, and the series is short and
    methodology-revised. So there is no research-grade real tape here.

    ``fetch=False`` (default) returns an empty frame so the offline core and tests never touch
    the network. ``fetch=True`` raises with the reason (kept executable so the caveat cannot rot).
    """
    path = os.path.join(cache_dir, "netflix_top10_real.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    if not fetch:
        return pd.DataFrame()
    raise NotImplementedError(
        "No free, point-in-time Netflix Top-10 engagement feed exists for a no-key retail "
        "stack: the weekly viewing-hours series is short (2021+), methodology-revised, and "
        "published as PDFs / a JS dashboard rather than a machine-readable panel. Study 551 is "
        "synthetic-only by design (capped at WEAK/NONE); see docs/results.md."
    )


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

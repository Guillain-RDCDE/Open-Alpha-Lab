"""Data layer for Study 555 (OpenTable-Reservations) — reservations-as-nowcast.

The claim is an **alt-data nowcast**: a dining-reservation index (OpenTable-style *seated-diners*,
the year-over-year change in people seated via online reservations) is supposed to lead the tape —
when restaurant traffic recovers, restaurant stocks (and, more loosely, consumer-discretionary /
XLY) should follow. The trade is: read the reservations trend this week, position the
restaurant basket for next week's move.

Why this study is **synthetic-only**. There is no free, clean, point-in-time, machine-readable
history of OpenTable seated-diners that a no-key retail stack can reach and cache reproducibly.
OpenTable published a public "State of the Industry" dashboard during 2020-2022 (daily YoY seated
diners) but (a) it was a moving HTML widget, never a stable CSV API; (b) it was discontinued /
repeatedly restated; and (c) it covers only the pandemic recovery window — a single, highly
non-stationary episode, not a tradable multi-cycle panel. So the honest move (like the desk's
lego-returns / whisky-cask / sneaker-resale studies) is to build a **deterministic synthetic
world** with a single knob that plants the nowcast, prove the engine catches it when present and
stays flat at the null, and state the data-availability limitation openly on the SIGNAL axis. A
synthetic-only study can **never** be `REAL` (that needs a robust *t* ≥ 2 on a REAL tape) — it is
capped at `WEAK`/`NONE`.

Two generators, one schema (a weekly series):

- ``synthetic_series`` — a deterministic, offline generator. A single knob, ``nowcast_beta``,
  plants the *predictive* link: next-week restaurant-basket return loads on this week's
  reservations *surprise* (the residual of the reservations index after removing its own trend and
  a common market factor). ``nowcast_beta > 0`` reproduces the claim; ``= 0`` is the null.
- ``fetch_series`` — the real-tape hook. It is **cache-first and returns empty by default**, and
  even with ``fetch=True`` it does not invent a reservations tape: the free seated-diners history
  does not exist, so there is nothing to cache. It exists to keep the schema honest and to make the
  "no real tape" limitation explicit in code, not just prose.

Columns of the weekly panel:

- ``reservations`` — the level of the seated-diners index (an alt-data traffic proxy).
- ``resv_yoy`` — the year-over-year change (what a dashboard actually prints).
- ``resv_surprise`` — the *tradable* signal: reservations YoY orthogonalised to its own recent
  trend and to the contemporaneous market factor (so it is not just "the market went up").
- ``mkt_ret`` — the weekly return of a broad market factor (the thing to control for).
- ``basket_ret`` — the weekly return of the restaurant basket (the thing to predict).

No look-ahead: the signal at week *t* uses only information available at the close of week *t*
(the reservations print and the trailing trend); the position is entered at that close and the
return realised over week *t+1*. That one-week execution lag is the single documented convention.
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

SEED = 555
WEEKS_PER_YEAR = 52


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic weekly world."""

    nowcast_beta: float  # loading of next-week basket return on this week's reservations surprise

    @property
    def has_nowcast(self) -> bool:
        return self.nowcast_beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic weekly series — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_weeks: int = 312,
    nowcast_beta: float = 0.35,
    mkt_vol: float = 0.020,
    idio_vol: float = 0.022,
    seed: int = SEED,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible weekly panel with a tunable reservations *nowcast*.

    Construction (all offline, seeded):

    1. A broad **market factor** ``mkt_ret`` (weekly, Gaussian).
    2. A latent **reservations surprise** ``u`` — the genuinely predictive part of the
       reservations trend, independent of the market. The observed reservations index is built as
       ``trend + loadings*mkt + u`` so that a naive regression on *raw* reservations would be
       confounded by the market; the strategy therefore orthogonalises first.
    3. The **restaurant basket** return next week loads on this week's surprise:

           basket_ret[t+1] = alpha + basket_beta*mkt_ret[t+1] + nowcast_beta*u[t] + idio[t+1]

       With ``nowcast_beta > 0`` a positive reservations surprise this week predicts a higher
       basket return next week (the claim); ``= 0`` is the null.

    Returns ``(panel, truth)`` where ``panel`` has one row per week with columns
    ``reservations, resv_yoy, resv_surprise, mkt_ret, basket_ret`` and a weekly DatetimeIndex.
    The ``resv_surprise`` column is the study's own reconstruction of ``u`` from observables (trend
    + market removed), so tests can check the engine recovers the planted link *without* peeking at
    the latent ``u``.
    """
    rng = np.random.default_rng(seed)

    # 1. market factor
    mkt = mkt_vol * rng.standard_normal(n_weeks)

    # 2. latent reservations surprise (persistent AR(1), the tradable part)
    u = np.zeros(n_weeks)
    innov = rng.standard_normal(n_weeks)
    phi = 0.6
    for t in range(1, n_weeks):
        u[t] = phi * u[t - 1] + np.sqrt(1 - phi**2) * innov[t]
    u = u - u.mean()

    # observed reservations index: a slow trend + market loading + the surprise + noise
    trend = np.cumsum(0.15 * rng.standard_normal(n_weeks))  # a wandering level
    resv_level = 100.0 + trend + 8.0 * mkt.cumsum() + 3.0 * u + 0.8 * rng.standard_normal(n_weeks)

    # 3. restaurant basket: loads on the market and (predictively) on last week's surprise
    basket_beta = 1.1
    alpha = 0.0015
    idio = idio_vol * rng.standard_normal(n_weeks)
    basket_ret = alpha + basket_beta * mkt + idio
    # nowcast: shift the surprise forward one week (u[t-1] predicts basket_ret[t])
    u_lag = np.concatenate([[0.0], u[:-1]])
    basket_ret = basket_ret + nowcast_beta * mkt_vol * u_lag  # scaled to a return magnitude

    idx = pd.date_range("2020-01-05", periods=n_weeks, freq="W-SUN")
    resv = pd.Series(resv_level, index=idx)
    resv_yoy = resv.pct_change(WEEKS_PER_YEAR)

    # resv_surprise: what the STRATEGY sees — YoY orthogonalised to trend + contemporaneous market.
    # (Reconstructed from observables, not the latent u.)
    surprise = _orthogonalize(resv_yoy, mkt, idx)

    panel = pd.DataFrame(
        {
            "reservations": resv.values,
            "resv_yoy": resv_yoy.values,
            "resv_surprise": surprise.values,
            "mkt_ret": mkt,
            "basket_ret": basket_ret,
        },
        index=idx,
    )
    return panel, WorldTruth(nowcast_beta=nowcast_beta)


def _orthogonalize(resv_yoy: pd.Series, mkt: np.ndarray, idx: pd.DatetimeIndex) -> pd.Series:
    """Reservations YoY with its own short trend and the contemporaneous market removed.

    The tradable "surprise" is what's left of the reservations trend after you strip (a) a slow
    moving-average trend (so we react to *changes*, not the level) and (b) the part explained by the
    market that week (so we're not just re-discovering beta). Returns a z-scored surprise series.
    """
    yoy = resv_yoy.copy()
    # YoY is undefined for the first year; keep those rows NaN so the aligned frame drops them
    # (no fake year-1 signal enters the regression).
    detr = yoy - yoy.rolling(13, min_periods=1).mean()  # remove a ~quarter trend
    m = pd.Series(mkt, index=idx)
    # regress detrended yoy on market (over the defined rows), keep residual
    valid = detr.notna() & m.notna()
    x = m[valid].to_numpy()
    y = detr[valid].to_numpy()
    if x.std() > 0 and len(x) > 2:
        b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
        resid = detr - b * m
    else:
        resid = detr
    sd = resid.std(ddof=1)
    z = (resid - resid.mean()) / sd if sd and sd > 0 else resid * 0.0
    # preserve NaN where YoY was undefined
    z = z.where(yoy.notna())
    return z.rename("resv_surprise")


# ---------------------------------------------------------------------------
# Real tape hook — deliberately empty (no free seated-diners history exists)
# ---------------------------------------------------------------------------
def fetch_series(cache_dir: str = DEFAULT_CACHE, fetch: bool = False) -> pd.DataFrame:
    """Real-tape hook — cache-first, and empty by design.

    There is no free, stable, machine-readable OpenTable seated-diners history to fetch and cache,
    so this returns whatever is cached (nothing, by default) and an **empty frame** otherwise. It
    exists to make the "no real tape" limitation explicit in code: ``HAVE_REAL`` is always False for
    this study, which is exactly why the Signal axis is capped below `REAL`.
    """
    path = os.path.join(cache_dir, "opentable_seated_diners.parquet")
    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df
    # No real source: even fetch=True cannot invent a tape. Return empty.
    return pd.DataFrame()


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

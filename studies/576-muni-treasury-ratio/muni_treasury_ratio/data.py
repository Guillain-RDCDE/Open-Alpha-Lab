"""Data layer for Study 576 (Muni-Treasury-Ratio) — the muni/Treasury yield ratio timer.

The **muni-Treasury ratio** (the "M/T ratio") is the single most-quoted rich/cheap gauge in the
municipal-bond market: the tax-exempt muni yield divided by the comparable-maturity Treasury
yield. Because muni coupons escape federal income tax, the ratio normally sits *below* 1.0 (a
muni can pay less and still leave the top-bracket investor even). The folklore: when the ratio is
**high** (munis yield an unusually large fraction of Treasuries) munis are *cheap* and should
outperform going forward; when it is **low**, munis are *rich* and should lag. The tradable claim
is that the ratio **times** muni or duration exposure.

This study works with what a no-key retail stack can actually reach on yfinance:

- **Daily total-return prices** for ``MUB`` (iShares National Muni Bond ETF) and ``IEF`` (iShares
  7-10Y Treasury ETF), the two instruments whose *yield ratio* is the signal and whose *forward
  returns* are the target — fully historical, so both the trailing yield and the forward return
  are honest.
- **Trailing-12-month distribution yields** for each ETF, built from ``Ticker.dividends`` divided
  by price — a standard retail proxy for the muni and Treasury yields. yfinance exposes no free
  muni **yield index**, so the ratio is a *distribution-yield* proxy, not the MMD/AAA-GO curve a
  desk would use — the limitation is named on the SIGNAL axis.

Two tapes, one schema (a daily frame with columns ``mt_ratio``, ``muni_ret``, ``tsy_ret``):

- ``synthetic_series`` — a deterministic, offline generator. A single knob, ``timing_beta``,
  plants the *timing* effect: the higher the (mean-centred) M/T ratio today, the **higher** the
  muni-minus-Treasury forward return (``timing_beta > 0`` reproduces the folklore; ``= 0`` is the
  null). The study's positive control and null in one bottle. Tests never touch the network.
- ``fetch_series`` — the real yfinance tape (daily total-return prices + trailing distribution
  yields), cache-first into this study's own ``_cache/`` so the reproducible core never needs the
  network.

No look-ahead: the ratio at date *t* uses only prices and distributions **up to and including**
*t* (a trailing-12-month distribution sum over the price at *t*); the position is entered at that
close and held over the *forward* window (*t*, *t*+h]. No future data enters the signal. The
same-close fill is a documented convention (one horizon *h*; a one-bar-later entry moves the
headline by a hair and changes no stamp — see ``strategy.py``).

Data availability: this is a **real**-tape study (yfinance reaches both ETFs and their
distributions). The muni-yield *proxy* (distribution yield, not the MMD curve) is the one honest
gap, stated on the SIGNAL axis; the tape itself is real, so a robust ``t >= 2`` here *can* earn
``REAL`` — the data decides.
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
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DEFAULT_CACHE = os.path.join(STUDY_DIR, "_cache")

# The two instruments whose yield ratio is the signal and whose forward returns are the target.
MUNI = "MUB"   # iShares National Muni Bond ETF (tax-exempt)
TSY = "IEF"    # iShares 7-10Y Treasury ETF (taxable, comparable duration)

TRADING_DAYS = 252


@dataclass(frozen=True)
class WorldTruth:
    """The planted effect for the synthetic series."""

    timing_beta: float  # forward (muni - tsy) return per unit of centred M/T ratio (>0 = folklore)

    @property
    def has_signal(self) -> bool:
        return self.timing_beta != 0.0


# ---------------------------------------------------------------------------
# Synthetic series — the deterministic offline core
# ---------------------------------------------------------------------------
def synthetic_series(
    n_days: int = 2600,
    timing_beta: float = 0.020,
    ratio_mean: float = 0.85,
    ratio_vol: float = 0.06,
    ratio_ar: float = 0.985,
    idio_vol: float = 0.010,
    horizon: int = 63,
    seed: int = 576,
) -> tuple[pd.DataFrame, WorldTruth]:
    """A reproducible daily tape with a tunable M/T-ratio *timing* effect.

    The M/T ratio follows a slow, mean-reverting (AR(1)) path around ``ratio_mean`` (munis
    normally yield ~0.85x Treasuries). Each day's *forward* muni-minus-Treasury excess return over
    the next ``horizon`` days is driven by the (mean-centred) ratio:

        fwd_excess = timing_beta * (ratio_t - ratio_mean) + idio

    With ``timing_beta > 0`` a *high* ratio (munis cheap) predicts muni outperformance (the
    folklore); ``timing_beta = 0`` is the null (forward excess is pure noise). We then back out
    daily muni and Treasury returns whose forward-``horizon`` difference reproduces that excess,
    so the frame has the same schema as ``fetch_series``: a daily ``mt_ratio`` plus daily
    ``muni_ret`` and ``tsy_ret`` from which the strategy re-derives forward returns itself (no
    forward leakage baked into the columns).

    Returns ``(frame, truth)`` with a ``DatetimeIndex`` and columns
    ``["mt_ratio", "muni_ret", "tsy_ret"]``.
    """
    rng = np.random.default_rng(seed)

    # Slow mean-reverting ratio path (AR(1)).
    ratio = np.empty(n_days)
    ratio[0] = ratio_mean
    innov = ratio_vol * np.sqrt(1 - ratio_ar**2)
    for i in range(1, n_days):
        ratio[i] = ratio_mean + ratio_ar * (ratio[i - 1] - ratio_mean) + innov * rng.standard_normal()
    ratio = np.clip(ratio, 0.55, 1.25)

    # Target: forward-horizon muni-minus-Treasury excess, driven by the centred ratio + noise.
    fwd_excess = timing_beta * (ratio - ratio_mean) + idio_vol * rng.standard_normal(n_days)

    # Base daily returns for the two legs (a common duration factor + idiosyncratic). The
    # idiosyncratic leg noise is kept modest so the planted forward tilt is detectable at
    # realistic betas without being swamped by day-to-day noise over the horizon.
    dur_factor = 0.0002 + 0.004 * rng.standard_normal(n_days)
    muni_ret = 0.00012 + 0.9 * dur_factor + 0.0011 * rng.standard_normal(n_days)
    tsy_ret = 0.00010 + 1.0 * dur_factor + 0.0011 * rng.standard_normal(n_days)

    # Inject the planted forward-excess as a small daily muni tilt so that the realised
    # forward-horizon (muni - tsy) difference tracks fwd_excess. Spreading the target excess
    # evenly across the horizon keeps the daily series realistic and forward-return-consistent.
    muni_ret = muni_ret + fwd_excess / horizon

    idx = pd.bdate_range("2015-01-02", periods=n_days)
    frame = pd.DataFrame(
        {"mt_ratio": ratio, "muni_ret": muni_ret, "tsy_ret": tsy_ret}, index=idx
    )
    frame.index.name = "date"
    return frame, WorldTruth(timing_beta=timing_beta)


# ---------------------------------------------------------------------------
# Real tape — yfinance daily prices + trailing distribution yields, cache-first
# ---------------------------------------------------------------------------
def _series_cache() -> str:
    return os.path.join(DEFAULT_CACHE, "muni_treasury_series.parquet")


def _retry(fn, tries: int = 3, pause: float = 1.0):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(pause)
    raise last  # type: ignore[misc]


def _trailing_yield(price: pd.Series, divs: pd.Series) -> pd.Series:
    """Trailing-12-month distribution yield: sum of the last 365d of distributions / price.

    A standard retail proxy for an ETF's yield. Uses only past distributions and the current
    price (no look-ahead).
    """
    divs = divs.copy()
    if getattr(divs.index, "tz", None) is not None:
        divs.index = divs.index.tz_localize(None)
    divs = divs[divs > 0]
    ttm = pd.Series(index=price.index, dtype=float)
    if divs.empty:
        return ttm
    # For each price date, sum distributions in the trailing 365 days.
    dval = divs.values
    didx = divs.index.values.astype("datetime64[ns]")
    pidx = price.index.values.astype("datetime64[ns]")
    window = np.timedelta64(365, "D")
    for i, d in enumerate(pidx):
        lo = d - window
        mask = (didx > lo) & (didx <= d)
        ttm.iloc[i] = dval[mask].sum()
    return (ttm / price).replace([np.inf, -np.inf], np.nan)


def fetch_series(
    cache_dir: str = DEFAULT_CACHE,
    fetch: bool = False,
    start: str = "2008-01-01",
    end: str = "2026-06-30",
) -> pd.DataFrame:
    """Daily total-return prices + trailing distribution yields for MUB & IEF, cache-first.

    Cache-only by default (``fetch=False``): returns the cached parquet if present, else an empty
    frame. Network is touched only on an explicit ``fetch=True`` (then cached).

    The returned frame is daily with columns:
      - ``mt_ratio``   — muni distribution yield / Treasury distribution yield (the signal)
      - ``muni_ret``   — MUB daily total return
      - ``tsy_ret``    — IEF daily total return
    plus the raw ``muni_yield`` / ``tsy_yield`` legs for transparency.
    """
    path = _series_cache()
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    if not fetch:
        return pd.DataFrame()

    import yfinance as yf  # lazy

    # Total-return (auto_adjust) prices for the two legs.
    px = _retry(
        lambda: yf.download(
            [MUNI, TSY], start=start, end=end, auto_adjust=True, progress=False, threads=True
        )["Close"]
    )
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px = px.dropna(how="any")

    # Distributions for each leg -> trailing-12m yield.
    yields = {}
    for tk, col in [(MUNI, "muni_yield"), (TSY, "tsy_yield")]:
        t = _retry(lambda tk=tk: yf.Ticker(tk))
        divs = _retry(lambda t=t: t.dividends)
        yields[col] = _trailing_yield(px[tk], divs)

    muni_ret = px[MUNI].pct_change()
    tsy_ret = px[TSY].pct_change()
    df = pd.DataFrame(
        {
            "muni_yield": yields["muni_yield"],
            "tsy_yield": yields["tsy_yield"],
            "muni_ret": muni_ret,
            "tsy_ret": tsy_ret,
        },
        index=px.index,
    )
    df["mt_ratio"] = (df["muni_yield"] / df["tsy_yield"]).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["mt_ratio", "muni_ret", "tsy_ret"])
    os.makedirs(cache_dir, exist_ok=True)
    df.to_parquet(path)
    return df


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

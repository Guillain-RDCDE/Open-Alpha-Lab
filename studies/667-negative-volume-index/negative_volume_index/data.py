"""Data layer for Study 667 — Negative Volume Index (Fosback).

Two ingredients, both offline-friendly once cached:

* **Real tape.** Daily ^GSPC OHLC + Volume (the long-history index tape Fosback's own
  claim is about — 1950-01-03 onward, the first session Yahoo! reports non-zero S&P 500
  volume) and daily SPY total-return OHLC + Volume (the tradable ETF proxy used for the
  costed timer / third axis, 1993-01-29 inception onward), both from yfinance (no key),
  cached as CSV under the study's own ``_cache/``.

  ^GSPC is a **price index — no dividends, no adjustment concept** (it cannot be
  "total-return"); its Volume column is Yahoo's index-level composite tape, a data-vendor
  construct rather than the NYSE's official composite tape Fosback used in 1976 — a named
  proxy limitation, not a silent one. SPY Close is **total-return** (``auto_adjust=True``
  folds dividends in) and its Volume is the ETF's own consolidated tape.

* **Synthetic world.** A deterministic, seeded random-walk price/volume tape with a
  TUNABLE planted "smart money" effect (knob ``edge``): on days volume is manufactured to
  fall, the next day's drift is nudged by ``edge``; days volume rises carry no extra
  drift. ``edge = 0`` is the null world — falling-volume days carry no informational edge,
  and the NVI-regime split must NOT manufacture significance. This is the exact structure
  the NVI folklore claims exists in the real tape (smart money trades quietly).

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
GSPC_CACHE = os.path.join(CACHE_DIR, "nvi_gspc.csv")
SPY_CACHE = os.path.join(CACHE_DIR, "nvi_spy.csv")

GSPC_START = "1950-01-03"   # first session Yahoo! reports non-zero ^GSPC volume
SPY_START = "1993-01-29"    # SPY inception
AS_OF = "2026-06-30"        # last complete month at publication (2026-07-10)
EMA_SPAN = 255              # Fosback's "1-year" EMA (255 US trading sessions)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch() -> None:
    """Download ^GSPC OHLCV and SPY total-return OHLCV; cache them. Network; once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)

    # ^GSPC — an index; no adjustment concept, take the bars as printed (price-only)
    gspc = yf.download("^GSPC", period="max", auto_adjust=False, progress=False)
    if isinstance(gspc.columns, pd.MultiIndex):
        gspc.columns = gspc.columns.get_level_values(0)
    gspc = gspc[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    gspc.to_csv(GSPC_CACHE)

    # SPY — total-return adjusted OHLCV (auto_adjust folds dividends into every field)
    spy = yf.download("SPY", period="max", auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy = spy[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
    spy.to_csv(SPY_CACHE)


def have_real() -> bool:
    return all(os.path.exists(p) for p in (GSPC_CACHE, SPY_CACHE))


def load_real(asof: str = AS_OF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cached (gspc, spy) frames, sliced to [own start, asof]."""
    gspc = pd.read_csv(GSPC_CACHE, index_col=0, parse_dates=True).sort_index()
    gspc = gspc.loc[(gspc.index >= GSPC_START) & (gspc.index <= asof)].copy()
    spy = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).sort_index()
    spy = spy.loc[(spy.index >= SPY_START) & (spy.index <= asof)].copy()
    return gspc, spy


# --------------------------------------------------------------------------- #
# Synthetic world — planted "smart money on quiet days" effect (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(edge: float = 0.0, seed: int = 667,
                    n_days: int = 9500, daily_vol: float = 0.009,
                    start: str = "1990-01-02",
                    p_quiet_neutral: float = 0.50, p_quiet_accum: float = 0.62,
                    flip_prob: float = 1.0 / 130) -> pd.DataFrame:
    """Deterministic daily price/volume tape with a TUNABLE planted NVI effect.

    A hidden two-state Markov regime (sticky, mean duration ~130 sessions) drives the
    literal Fosback mechanism: an "accumulation" state where (a) volume is MORE likely
    to fall day-to-day (``p_quiet_accum`` vs the neutral state's ``p_quiet_neutral`` —
    quiet days cluster exactly when Fosback says they should) and (b) the index drifts
    an extra ``edge`` (daily-vol units) while that state is active. NVI, built only from
    quiet days, over-samples the accumulation state even before the drift is visible in
    the plain index — the exact "smart money trades quietly, price follows" claim.

    ``edge = 0`` is the null: quiet days still cluster by (hidden, unobservable) state,
    but the state carries NO drift information, so the NVI-regime split must not fire.
    ``edge > 0`` plants the claimed effect for the detector to recover.

    Business-day index, span < 40 years — far below the 250-year pandas ns-timestamp
    trap. Returns an OHLCV-shaped frame (Close, Volume only needed downstream).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)

    state = np.zeros(n_days, dtype=int)
    flips = rng.random(n_days) < flip_prob
    flips[0] = False
    state = (np.cumsum(flips) % 2).astype(int)

    p_quiet = np.where(state == 1, p_quiet_accum, p_quiet_neutral)
    quiet = rng.random(n_days) < p_quiet

    volume = np.empty(n_days)
    volume[0] = 1.0e8
    down_mult = rng.uniform(0.75, 0.97, n_days)
    up_mult = rng.uniform(1.03, 1.35, n_days)
    for t in range(1, n_days):
        volume[t] = volume[t - 1] * (down_mult[t] if quiet[t] else up_mult[t])

    drift = edge * daily_vol * state
    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = drift + eps
    close = 100.0 * np.exp(np.cumsum(log_ret))

    return pd.DataFrame({"Close": close, "Volume": volume}, index=idx)

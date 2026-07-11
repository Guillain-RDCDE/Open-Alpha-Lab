"""Data layer for Study 671 — Special K.

Three ingredients, all offline-friendly once cached:

* **Real tape, primary.** SPY daily **total-return** closes (``auto_adjust=True``, dividends
  reinvested) since SPY's 1993-01-29 inception — the house-standard tape used by every
  sibling technical-indicator study on this desk (105/424/425/426/427). Used for the
  headline crossover event study, the long/flat timer race and the parameter-robustness
  sweep.

* **Real tape, secondary.** ``^GSPC`` daily **price-only** closes (no adjustment concept for
  an index — this is *not* total return, and is labeled as such everywhere it appears) since
  1962-01-02, giving ~64 years and every major post-war cyclic turn (1962 flash break, 1966,
  1970, 1973-74, 1980-82, 1987 crash, 2000-02 dot-com, 2007-09 GFC, 2020 COVID, 2022 bear) —
  a cross-check that the crossover event study isn't an artefact of SPY's shorter 33-year
  sample or its dividend stream.

* **Weekly bars.** Derived by resampling the cached SPY daily frame to Friday closes
  (``resample("W-FRI")``) — no separate network call or cache file. Used for the weekly-bar
  robustness check (Special K's periods scaled by /5, rounded, to keep the same real-world
  lookback in calendar time).

* **Synthetic world.** A deterministic, seeded **regime-switching** log-return tape with a
  TUNABLE planted **cycle amplitude** knob: the drift alternates between a bull state and a
  bear state with a geometric (mean ~1.6-year) duration — genuine multi-year structure at the
  timescale Special K's slowest 530-day ROC is built to see, unlike a one-day AR(1) tweak
  (tested and rejected: it decays in ~1-2 sessions and never reaches Special K's smoothing
  horizon). ``amp = 0`` collapses both states to the same drift — a driftless-of-regime walk:
  the crossover timing must show NO reliable regime-return gap and no value-add over
  buy-and-hold. This is the study's faithful-engine / power check, never cited as market
  evidence.

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
SPY_CACHE = os.path.join(CACHE_DIR, "sk_spy.csv")
GSPC_CACHE = os.path.join(CACHE_DIR, "sk_gspc.csv")

SPY_START = "1993-01-29"     # SPY inception
GSPC_START = "1962-01-02"    # deep enough for several pre-1993 bear/bull cycles
AS_OF = "2026-06-30"         # last complete month at publication (2026-07-10)


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(end: str = "2026-07-01") -> None:
    """Download SPY (total-return) and ^GSPC (price-only) daily closes; cache them."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)

    spy = yf.download("SPY", start=SPY_START, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy[["Close"]].dropna().to_csv(SPY_CACHE)

    gspc = yf.download("^GSPC", start=GSPC_START, end=end, auto_adjust=False, progress=False)
    if isinstance(gspc.columns, pd.MultiIndex):
        gspc.columns = gspc.columns.get_level_values(0)
    gspc[["Close"]].dropna().to_csv(GSPC_CACHE)


def have_real() -> bool:
    return os.path.exists(SPY_CACHE) and os.path.exists(GSPC_CACHE)


def load_real(asof: str = AS_OF) -> tuple[pd.Series, pd.Series]:
    """Cached (SPY total-return close, ^GSPC price-only close), sliced to <= asof."""
    spy = pd.read_csv(SPY_CACHE, index_col=0, parse_dates=True).sort_index()["Close"]
    gspc = pd.read_csv(GSPC_CACHE, index_col=0, parse_dates=True).sort_index()["Close"]
    spy = spy[spy.index <= asof].dropna()
    gspc = gspc[gspc.index <= asof].dropna()
    spy.name, gspc.name = "close", "close"
    return spy, gspc


def weekly_from_daily(close: pd.Series) -> pd.Series:
    """Friday-close weekly resample of a daily close series (last obs each week)."""
    wk = close.resample("W-FRI").last().dropna()
    wk.name = "close"
    return wk


# --------------------------------------------------------------------------- #
# Synthetic world — planted MULTI-YEAR regime cycle (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_tape(amp: float = 0.0, seed: int = 671, n_days: int = 6000,
                   daily_vol: float = 0.011, base_drift: float = 0.0002,
                   mean_regime_days: int = 400,
                   start: str = "2000-01-03") -> pd.Series:
    """Deterministic daily close series with a TUNABLE planted bull/bear regime cycle.

    A two-state Markov chain (bull / bear) with geometric sojourn times of mean
    ``mean_regime_days`` (~1.6y at the default) sets each day's drift to
    ``base_drift + state * amp`` (state in {+1, -1}); shocks are i.i.d. normal
    ``N(0, daily_vol)``. ``amp = 0`` collapses the two states to the same drift — a plain
    driftful random walk with no regime structure a crossover rule could possibly exploit.
    ``amp > 0`` plants genuine multi-year cyclic turns at the timescale Special K's slowest
    (530-day) component is built to detect.

    ``n_days`` business days from ``start``; span stays far below the ~250-year pandas
    ns-timestamp ceiling.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    p_switch = 1.0 / mean_regime_days
    state = 1.0
    states = np.empty(n_days)
    for i in range(n_days):
        if rng.random() < p_switch:
            state = -state
        states[i] = state
    eps = rng.normal(0.0, daily_vol, n_days)
    log_ret = base_drift + amp * states + eps
    close = pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=idx, name="close")
    return close

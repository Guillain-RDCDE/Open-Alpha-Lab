"""Data layer for Study 609 — VIX Weekend Arithmetic.

Three ingredients, all offline-friendly once cached:

* **Real tape.** Daily ^VIX closes from yfinance (no key), 1990 onward — the full history of
  the index — cached as CSV under the study's own ``_cache/``. Plus **VIXY** (ProShares VIX
  Short-Term Futures ETF, auto-adjusted closes, 2011+) for the third axis: does the index's
  weekend arithmetic leak into a *tradable* vehicle?

* **The variance-day-count model.** The VIX squares to the expected variance of the S&P 500
  over the next **30 calendar days**, annualized on a calendar-day clock. But variance does
  not accrue uniformly over calendar days — most of it accrues while the market trades. If a
  weekend day carries only a fraction ``f`` of a trading day's variance, the *effective*
  variance-day count of the 30-day window depends on the weekday you quote it:
  the window quoted on **Friday** contains 20 trading + 10 weekend days, the window quoted on
  **Monday** contains 22 trading + 8 weekend days. Pure arithmetic then forces the quoted VIX
  **down into Thursday/Friday and up over the weekend** — with a *predicted* day-of-week
  change of ``0.5 · ln(Neff(d)/Neff(d-1))`` where ``Neff(d) = N_trading(d) + f·N_weekend(d)``.
  ``model_weekday_changes(f)`` returns that prediction; ``implied_weekend_fraction`` (in
  strategy.py) inverts the tape for the market's implied ``f``.

* **Synthetic world.** A deterministic, seeded generator: a mean-reverting log-AR(1) "true
  vol" process, quoted through the day-count arithmetic with a TUNABLE planted weekend
  fraction ``f``. ``f = 1.0`` is the **null** (weekends count fully — no weekday pattern, the
  machinery must not manufacture significance); ``f < 1`` plants the seesaw and the estimator
  must recover both the pattern and ``f`` itself. Index built with ``pd.bdate_range`` over
  ~30 years (well under the 250-year ns-Timestamp ceiling).

Pure numpy + pandas + stdlib on the offline path. ``fetch()`` (network) runs once to build
the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
VIX_CACHE = os.path.join(CACHE_DIR, "vix_close.csv")
VIXY_CACHE = os.path.join(CACHE_DIR, "vixy_close.csv")

START = "1990-01-02"      # first ^VIX print
AS_OF = "2026-06-30"      # last complete month at publication (2026-07-03)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "1990-01-01", end: str = "2026-07-01") -> None:
    """Download ^VIX and VIXY daily closes and cache them. Network; run once."""
    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for tk, path, adj in (("^VIX", VIX_CACHE, False), ("VIXY", VIXY_CACHE, True)):
        raw = yf.download(tk, start=start, end=end, auto_adjust=adj, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw[["Close"]].dropna().to_csv(path)


def have_real() -> bool:
    return os.path.exists(VIX_CACHE) and os.path.exists(VIXY_CACHE)


def _load(path: str, start: str, asof: str) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    s = df["Close"].astype(float)
    return s.loc[(s.index >= start) & (s.index <= asof)].copy()


def load_vix(start: str = START, asof: str = AS_OF) -> pd.Series:
    """Cached ^VIX daily closes, sliced to [start, asof]."""
    return _load(VIX_CACHE, start, asof)


def load_vixy(asof: str = AS_OF) -> pd.Series:
    """Cached VIXY auto-adjusted (total-return) daily closes, sliced to as-of."""
    return _load(VIXY_CACHE, "2011-01-01", asof)


# --------------------------------------------------------------------------- #
# The variance-day-count model
# --------------------------------------------------------------------------- #
def window_day_counts(weekday: int) -> tuple[int, int]:
    """(trading days, weekend days) inside the 30-calendar-day window (d, d+30]
    for a VIX quote printed on ``weekday`` (0=Mon … 4=Fri). Stylized week — no holidays."""
    n_weekend = sum(1 for i in range(1, 31) if (weekday + i) % 7 in (5, 6))
    return 30 - n_weekend, n_weekend


def n_effective(weekday: int, f: float) -> float:
    """Effective variance-day count of the 30-day window quoted on ``weekday``,
    when one weekend day carries a fraction ``f`` of a trading day's variance."""
    n_trade, n_wknd = window_day_counts(weekday)
    return n_trade + f * n_wknd


def model_weekday_changes(f: float) -> dict[int, float]:
    """Model-predicted close-to-close ΔlogVIX (in %) by weekday of the *arriving* close.

    The quoted VIX is ``true_vol · sqrt(Neff(d)/30)`` under the day-count model, so the pure-
    arithmetic close-to-close change landing on weekday ``d`` (previous trading day = d-1,
    with Monday's predecessor Friday) is ``50 · ln(Neff(d)/Neff(prev(d)))`` percent.
    ``f = 1`` (weekends count fully) → all zeros. ``f = 0`` (weekends carry no variance) →
    the full-arithmetic bound: Monday ≈ +4.8 %, Thursday/Friday ≈ −2.4 % each.
    """
    prev = {0: 4, 1: 0, 2: 1, 3: 2, 4: 3}
    return {d: 50.0 * math.log(n_effective(d, f) / n_effective(prev[d], f))
            for d in range(5)}


# --------------------------------------------------------------------------- #
# Synthetic world (positive control)
# --------------------------------------------------------------------------- #
def synthetic_tape(n_years: int = 30, f: float = 1.0, seed: int = 609,
                   phi: float = 0.98, sig_eps: float = 0.05,
                   mean_level: float = 19.0) -> pd.Series:
    """Deterministic synthetic "quoted VIX" with a TUNABLE planted weekend fraction ``f``.

    True (unobservable) vol follows a mean-reverting log-AR(1); the *quoted* index applies
    the day-count arithmetic ``quoted = true · sqrt(Neff(weekday, f)/30)``.

    * ``f = 1.0`` — the **null**: weekends count as full variance days, Neff ≡ 30, the quote
      equals the true process and carries **no** weekday pattern.
    * ``f < 1.0`` — the planted effect: the quote is marked down into Thursday/Friday and pops
      on Monday by exactly the model arithmetic. The estimator must recover the seesaw *and*
      the planted ``f``.

    Business-day index (~``n_years``·261 rows, far below the 250-year ns-Timestamp ceiling).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1990-01-02", periods=int(n_years * 261))
    n = len(idx)
    lv = np.empty(n)
    mu = math.log(mean_level)
    lv[0] = mu
    eps = rng.normal(0.0, sig_eps, size=n)
    for t in range(1, n):
        lv[t] = mu + phi * (lv[t - 1] - mu) + eps[t]
    true_vol = np.exp(lv)
    factor = np.array([math.sqrt(n_effective(d, f) / 30.0) for d in idx.weekday])
    return pd.Series(true_vol * factor, index=idx, name="Close")

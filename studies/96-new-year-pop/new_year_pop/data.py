"""Data layer for Study 96 (New-Year-Pop) — the January small-cap effect.

The unit of analysis here is a **monthly small-minus-large return spread**: in each
calendar month, the return of a small-cap index minus the return of a large-cap index.
The claim under test is that *January's* spread is systematically higher than the rest of
the year (Keim 1983; Reinganum 1983) — and that it decayed after publication.

Two real pairs, two honest labels:

- **Long sample (price-only):** ``^RUT`` (Russell 2000) vs ``^GSPC`` (S&P 500), both
  **price-only** (no dividends — Yahoo has no total-return series for these indices).
  State this loudly: a price-only small-minus-large spread omits the (small) dividend
  difference between the legs. For a *seasonal contrast within* the spread this is benign
  (the dividend gap is roughly constant across calendar months), but it is not a
  total-return number.
- **Tradable pair (total-return):** ``IWM`` (iShares Russell 2000, since 2000) vs ``SPY``
  (SPDR S&P 500), both **total-return adjusted** via :mod:`quantlab.data`. This is the
  thing you could actually hold, dividends folded in, for the Beat-6 tradability read.

A synthetic control plants a known January bump in the small leg (the harness must detect
it) or no seasonal at all (the harness must NOT manufacture one) — the study's spine.

No look-ahead is baked in: a calendar month is known the moment it ends; the seasonal
rule (hold small in January, large otherwise) is a calendar-known rule that needs no
execution lag. See ``strategy.py``.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))


# ---------------------------------------------------------------------------
# Synthetic tape — the deterministic offline core (positive + negative control)
# ---------------------------------------------------------------------------
def synthetic_monthly(
    n_years: int = 36,
    jan_bump: float = 0.0,
    base_vol: float = 0.045,
    start: str = "1990-01-31",
    seed: int = 96,
) -> tuple[pd.DataFrame, dict]:
    """Two monthly return series (``small``, ``large``) with a tunable January bump.

    - ``jan_bump = 0`` → small and large differ only by i.i.d. noise; there is **no**
      January seasonality in the small-minus-large spread. The harness must NOT find a
      significant January effect. This is the study's null.
    - ``jan_bump > 0`` → the small leg earns an *extra* ``jan_bump`` return in every
      January (and only January). The harness must detect that January's small-minus-large
      spread is significantly higher than the rest of the year. This is the positive
      control that proves the pipeline can bank a planted seasonal.

    Returns ``(frame, truth)`` where ``frame`` has monthly ``small`` and ``large`` return
    columns indexed by month-end, and ``truth`` records the planted parameters.
    """
    rng = np.random.default_rng(seed)
    n = n_years * 12
    idx = pd.date_range(start=start, periods=n, freq="ME")

    # Shared market factor both legs load on, plus idiosyncratic noise per leg.
    mkt = rng.normal(0.007, base_vol, n)
    large = mkt + rng.normal(0.0, base_vol * 0.4, n)
    small = mkt + rng.normal(0.0, base_vol * 0.7, n)  # small is noisier (higher beta-ish)

    is_jan = idx.month == 1
    small = small + np.where(is_jan, jan_bump, 0.0)

    frame = pd.DataFrame({"small": small, "large": large}, index=idx)
    frame.index.name = "Date"
    truth = {
        "jan_bump": jan_bump,
        "n_years": n_years,
        "n_months": n,
        "base_vol": base_vol,
        "seed": seed,
    }
    return frame, truth


# ---------------------------------------------------------------------------
# Real tapes — monthly returns from daily closes via quantlab, cache-first
# ---------------------------------------------------------------------------
def _monthly_returns(ticker: str, start: str, end: str | None, mode: str,
                     use_cache: bool) -> pd.Series:
    """Monthly total/price return for ``ticker`` from daily closes (cache-first).

    Returns a month-end-indexed Series of simple monthly returns. The series is
    resampled month-end then pct-changed, so each bar is a complete calendar month;
    callers drop any in-progress final month before stamping stats.
    """
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
    from quantlab import data as qdata

    raw = qdata.fetch(ticker, start=start, end=end, mode=mode, use_cache=use_cache)
    monthly_close = raw["Close"].resample("ME").last()
    return monthly_close.pct_change().rename(ticker)


def load_real_long(
    small: str = "^RUT",
    large: str = "^GSPC",
    start: str = "1990-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Long-sample monthly small/large returns — ``^RUT`` vs ``^GSPC``, **price-only**.

    Both legs are price-only (Yahoo carries no dividend-adjusted series for these spot
    indices). Returns a frame with monthly ``small`` and ``large`` return columns over the
    overlapping window. Price-only is stated honestly everywhere this number appears.
    """
    s = _monthly_returns(small, start, end, mode="split_only", use_cache=use_cache)
    l = _monthly_returns(large, start, end, mode="split_only", use_cache=use_cache)
    out = pd.concat([s.rename("small"), l.rename("large")], axis=1).dropna()
    out.index.name = "Date"
    return out


def load_real_tradable(
    small: str = "IWM",
    large: str = "SPY",
    start: str = "2000-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Tradable monthly small/large returns — ``IWM`` vs ``SPY``, **total-return**.

    Both legs dividend-adjusted (``mode="total_return"``) — the thing you could actually
    hold. IWM lists in 2000, so the tradable window starts there. Returns a frame with
    monthly ``small`` and ``large`` return columns over the overlapping window.
    """
    s = _monthly_returns(small, start, end, mode="total_return", use_cache=use_cache)
    l = _monthly_returns(large, start, end, mode="total_return", use_cache=use_cache)
    out = pd.concat([s.rename("small"), l.rename("large")], axis=1).dropna()
    out.index.name = "Date"
    return out


def fingerprint(frame: pd.DataFrame) -> str:
    """A short content fingerprint of a two-leg monthly frame, for the as-of stamp."""
    arr = np.ascontiguousarray(frame[["small", "large"]].to_numpy())
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]

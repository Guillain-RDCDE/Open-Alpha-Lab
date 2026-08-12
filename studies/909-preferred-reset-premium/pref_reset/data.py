"""Data layer for Study 909 — Preferred Reset Premium.

The claim under test: **fixed-rate preferreds got crushed in 2022 as rates rose, while
fixed-to-FLOATING / variable-rate preferreds reset their coupon and held up** — so a
variable-rate preferred sleeve should deliver a *better rate-adjusted carry* than a plain
fixed-rate preferred sleeve in the high-rate regime. Distinct from study 338 (which tests
the preferred asset class *as a whole* against equities/bonds); here the axis is the
**within-preferred variable-vs-fixed spread**.

Two ingredients, both offline once cached.

* **Real tape — five liquid preferred ETFs + a cash leg (yfinance, total-return closes).**

  Variable / floating-rate sleeve:
    - **VRP**  — Invesco Variable Rate Preferred ETF (2014-05).
    - **PFFV** — Global X Variable Rate Preferred ETF (2020-06).
  Fixed-rate sleeve:
    - **PFF**  — iShares Preferred & Income Securities ETF (2007-03), the flagship.
    - **PGX**  — Invesco Preferred ETF (2008-01).
    - **PGF**  — Invesco Financial Preferred ETF (2006-12).
  Cash leg:
    - **BIL**  — SPDR Bloomberg 1-3 Month T-Bill ETF (2007-05), the excess-of-cash base.

  ``auto_adjust=True`` (dividends folded in — mandatory for income instruments whose
  return is mostly the coupon). VRP bounds the flagship variable-vs-fixed pair at
  **2014-06**; PFFV bounds the multi-name sleeve at **2020-07**.

  **Short history — named on the Signal axis.** The variable-rate preferred ETFs are
  *young*: VRP has ~12 years, PFFV barely six, and the whole thesis leans on a **single**
  high-rate episode (the 2022 hiking cycle). One regime is not a law; the caveat travels
  with every number.

* **Synthetic world — the positive control.** A deterministic seeded monthly generator
  (``synthetic_world``) with a TUNABLE knob ``edge``: a common credit factor drives both
  sleeves, but in the *high-rate* months the fixed sleeve eats a duration hit while the
  variable sleeve resets and earns ``edge`` extra. ``edge = 0`` is the null (both sleeves
  identical up to noise; the detector must find nothing). ``edge > 0`` plants exactly the
  reset premium the estimator targets. The monthly index is a ``PeriodIndex`` (no
  ns-Timestamp overflow).

``fetch()`` (network, yfinance) runs once and writes ``_cache/pref_prices.parquet``;
``load_prices`` and everything downstream are offline.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PRICES_CACHE = os.path.join(CACHE_DIR, "pref_prices.parquet")

VARIABLE = ["VRP", "PFFV"]          # fixed-to-floating / variable-rate preferred sleeve
FIXED = ["PFF", "PGX", "PGF"]       # fixed-rate preferred sleeve
CASH = "BIL"                        # 1-3 month T-bill, the excess-of-cash base
TICKERS = VARIABLE + FIXED + [CASH]

AS_OF = "2026-06-30"                # last complete calendar month at build time

# Inception dates that bound the joint windows (yfinance first close).
VRP_INCEPTION = "2014-05-01"        # flagship variable-vs-fixed pair starts 2014-06 monthly
PFFV_INCEPTION = "2020-06-25"       # full variable sleeve starts 2020-07 monthly

# The regime the thesis lives in: the 2022 hiking cycle onward.
HIGH_RATE_SPLIT = "2022-01-01"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = "2005-01-01", end: str | None = None,
          path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download total-return closes for all tickers and cache them (network, run once)."""
    import yfinance as yf

    raw = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                              progress=False)["Close"]
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data for the preferred ETFs")
    raw = raw.dropna(how="all").sort_index()
    raw = raw[raw.index <= pd.Timestamp(AS_OF)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_parquet(path)
    return raw


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE) -> pd.DataFrame:
    """Wide daily total-return close frame, cache-first (fetches once on a miss)."""
    if not os.path.exists(path):
        return fetch(path=path)
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    return df.sort_index()


def monthly_returns(prices: pd.DataFrame, asof: str = AS_OF) -> pd.DataFrame:
    """Monthly simple total returns per ticker, sliced to the as-of month-end.

    Resamples daily total-return closes to month-end, drops the partial current month, and
    takes ``pct_change`` so every column is a monthly total return in decimal.
    """
    px = prices[prices.index <= pd.Timestamp(asof)]
    m = px.resample("ME").last()
    m = m[m.index <= pd.Timestamp(asof)]
    return m.pct_change()


def sleeve_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight sleeve monthly returns: ``variable`` (VRP,PFFV), ``fixed`` (PFF,PGX,PGF),
    ``cash`` (BIL), plus the flagship single names ``VRP`` and ``PFF``.

    Sleeve columns are the row-mean of whichever member names have a return that month
    (so ``variable`` starts as VRP-only in 2014 and adds PFFV from 2020), matching a
    naive equal-weight hold of the available ETFs.
    """
    out = pd.DataFrame(index=monthly.index)
    out["variable"] = monthly[VARIABLE].mean(axis=1, skipna=True)
    out["fixed"] = monthly[FIXED].mean(axis=1, skipna=True)
    out["cash"] = monthly[CASH]
    out["VRP"] = monthly["VRP"]
    out["PFF"] = monthly["PFF"]
    return out


# --------------------------------------------------------------------------- #
# Synthetic world (positive control + null)
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 150, edge: float = 0.0030, seed: int = 909,
                    credit_vol: float = 0.030, dur_hit: float = 0.010,
                    idio_vol: float = 0.006, cash_annual: float = 0.02,
                    high_frac: float = 0.4) -> pd.DataFrame:
    """Deterministic monthly world with a PLANTED reset premium.

    A common credit factor drives both preferred sleeves. In the ``high_frac`` fraction of
    months flagged as the *high-rate regime* the fixed sleeve eats a duration hit
    ``dur_hit`` while the variable sleeve resets its coupon and earns ``edge`` extra::

        variable = credit + regime * edge      + idio_v + cash/12
        fixed    = credit - regime * dur_hit    + idio_f + cash/12

    so (variable - fixed) has mean ~ (edge + dur_hit) inside the high-rate regime and ~0
    outside it — exactly the regime-contingent spread the estimator targets. ``edge = 0``
    AND ``dur_hit = 0`` is the null: the two sleeves are the same asset up to noise and the
    detector must NOT manufacture a spread. Monthly index kept as a ``PeriodIndex`` (no
    ns-Timestamp overflow on the CI).

    Returns a frame with columns ``variable``, ``fixed``, ``cash``, ``regime`` (1/0) on a
    monthly ``PeriodIndex``.
    """
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2010-01", periods=n_months, freq="M")
    credit = rng.normal(0.004, credit_vol, n_months)
    # Deterministic high-rate regime: a contiguous block in the back third of the sample.
    regime = np.zeros(n_months, dtype=float)
    k = int(round(n_months * high_frac))
    regime[n_months - k:] = 1.0
    cash = np.full(n_months, cash_annual / 12.0)
    variable = credit + regime * edge + rng.normal(0.0, idio_vol, n_months) + cash
    fixed = credit - regime * dur_hit + rng.normal(0.0, idio_vol, n_months) + cash
    return pd.DataFrame(
        {"variable": variable, "fixed": fixed, "cash": cash, "regime": regime},
        index=idx,
    )


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
def fingerprint(frame: pd.DataFrame) -> str:
    """Short, stable content hash of a price/return frame for the as-of stamp."""
    h = hashlib.sha1()
    for c in sorted(map(str, frame.columns)):
        h.update(c.encode())
        h.update(np.ascontiguousarray(
            frame[c].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]

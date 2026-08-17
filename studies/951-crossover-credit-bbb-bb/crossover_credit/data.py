"""Data layer for Study 951 — The Crossover Rung (BBB/BB, the credit-ladder boundary).

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the rungs of the US corporate credit ladder:
  ``AGG`` (broad aggregate), ``LQD`` (investment grade), ``ANGL`` and ``FALN``
  (fallen-angel / crossover funds that sit *on* the BBB-to-BB boundary), ``HYG`` and
  ``USHY`` (broad high yield), plus the two risk factors the race is adjusted against
  (``IEF`` for duration, ``SPY`` for equity beta) and ``BIL`` for the cash leg.
  ``fetch`` touches the network and caches parquet under the **shared** ``studies/_cache``
  (retry up to 4x); ``load_prices`` reads that cache **offline** and never imports
  yfinance. The whole test-suite runs with NO cache present (synthetic-only), so CI is
  green on a fresh checkout.

- ``synthetic_panel`` — a *deterministic, offline* generator. A three-rung credit ladder
  driven by a duration factor and an equity factor, plus a cash accrual leg. The
  ``signal_strength`` knob plants a crossover premium *on top of* the factor exposures:
  at ``signal_strength=0`` every rung is pure factor beta and zero alpha (the null — the
  estimator must stay quiet); at ``signal_strength=1`` the crossover rung carries a real
  ``alpha_crossover`` the two-factor regression must recover. Seed is fixed → tests are
  deterministic.

**Proxies and assumptions, named up front** (all swept or cross-checked in
``strategy.py`` and ``docs/results.md``):

- *Crossover rung.* There is no free daily total-return tape of a literal BBB-/BB+ index.
  We use the **fallen-angel** ETFs (ANGL, FALN) as the tradable proxy: their holdings are
  bonds downgraded out of investment grade, i.e. issuers sitting exactly on the boundary.
  A fallen-angel fund is a *proxy* for the crossover rung, not an identity — it is
  event-tilted (recent downgrades) and longer-duration than broad high yield.
- *Risk adjustment.* Duration is proxied by **IEF** (7-10y Treasuries) and equity beta by
  **SPY**. This is a two-factor adjustment, not a full credit-factor model — no separate
  term-slope, liquidity or default-risk factor.
- *Cash.* **BIL** (1-3M T-bills) total return, so every Sharpe is excess-of-cash against
  the *actual* path of short rates (0% in 2012-2015, ~5% in 2023-2026).
- *Borrow.* Only the optional long/short expression shorts anything; its borrow rate is an
  assumption and is swept 0 / 25 / 50 / 100 bps.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (weights formed
from data through day ``t`` are acted on at day ``t+1``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache, one level up from the study directory.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The ladder, from the safest rung to the riskiest, plus factors and cash.
RUNGS = ("AGG", "LQD", "ANGL", "HYG")          # the headline ladder (ANGL-gated window)
CROSSOVER = ("ANGL", "FALN")                    # boundary proxies
BROAD_HY = ("HYG", "USHY")                      # broad high-yield tapes
FACTORS = ("IEF", "SPY")                        # duration, equity
CASH = "BIL"

TICKERS = ("LQD", "ANGL", "HYG", "USHY", "FALN", "AGG", "BIL", "IEF", "SPY")

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2002-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and dividend-adjusted **total return** — non-negotiable
    for bond funds, whose entire return is coupon.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    tk, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {tk}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        df = raw[["close"]].copy()
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.dropna(subset=["close"])
        df.to_parquet(_cache_path(tk, cache_dir))
        out[tk] = df
    return out


def have_real(tickers=TICKERS, cache_dir: str = DEFAULT_CACHE) -> bool:
    """True iff every ticker's parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read cached daily total-return closes OFFLINE into one aligned close frame.

    Returns a frame indexed by date with one column per ticker (the adjusted close),
    sliced to ``asof`` so the sample never creeps. Raises ``FileNotFoundError`` if any
    ticker is missing — the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call crossover_credit.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a price frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted crossover premium)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 14,
    signal_strength: float = 1.0,      # 0 = null (zero alpha everywhere), 1 = full premium
    alpha_crossover: float = 0.030,    # annualised premium planted on the crossover rung
    dur_vol: float = 0.070,            # annualised vol of the duration factor
    eq_vol: float = 0.170,             # annualised vol of the equity factor
    dur_drift: float = 0.010,          # annualised excess return of the duration factor
    eq_drift: float = 0.060,           # annualised excess return of the equity factor
    idio_vol: float = 0.020,           # annualised idiosyncratic vol per rung
    start: str = "2012-04-12",
    seed: int = 951,
    cash_rate_ann: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """A three-rung credit ladder built from a duration factor and an equity factor.

    Rung loadings are hard-coded to roughly mimic the real tape: investment grade is
    almost pure duration, the crossover rung splits duration and equity, broad high yield
    is the most equity-like and the shortest duration. The ``signal_strength`` knob scales
    the *planted* crossover premium:

    - ``signal_strength = 1`` → the crossover rung earns ``alpha_crossover`` per year on
      top of its factor exposure; the two-factor HAC regression must recover it.
    - ``signal_strength = 0`` → every rung is pure factor beta with **zero** alpha (the
      null — the estimator must not manufacture a premium).

    Returns ``(prices, truth)`` where ``prices`` carries total-return close levels for
    ``ig`` / ``crossover`` / ``hy`` (the rungs), ``dur`` / ``eq`` (the factor tapes) and
    ``cash``, and ``truth`` records the planted parameters. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    sd = np.sqrt(TRADING_DAYS_PER_YEAR)
    r_dur = rng.normal(dur_drift / TRADING_DAYS_PER_YEAR, dur_vol / sd, n_days)
    r_eq = rng.normal(eq_drift / TRADING_DAYS_PER_YEAR, eq_vol / sd, n_days)

    loadings = {
        "ig": (0.95, 0.15),
        "crossover": (0.35, 0.35),
        "hy": (0.25, 0.40),
    }
    alphas = {
        "ig": 0.0,
        "crossover": signal_strength * alpha_crossover,
        "hy": 0.0,
    }

    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0
    cash_idx = np.cumprod(np.full(n_days, 1.0 + cash_daily))

    cols = {}
    for name, (b_dur, b_eq) in loadings.items():
        idio = rng.normal(0.0, idio_vol / sd, n_days)
        excess = alphas[name] / TRADING_DAYS_PER_YEAR + b_dur * r_dur + b_eq * r_eq + idio
        total = excess + cash_daily          # rung total return = excess + cash
        cols[name] = 100.0 * np.cumprod(1.0 + total)
    cols["dur"] = 100.0 * np.cumprod(1.0 + r_dur + cash_daily)
    cols["eq"] = 100.0 * np.cumprod(1.0 + r_eq + cash_daily)
    cols["cash"] = 100.0 * cash_idx

    prices = pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date"))
    truth = {
        "signal_strength": signal_strength,
        "alpha_crossover": alpha_crossover,
        "alpha_planted": alphas["crossover"],
        "loadings": loadings,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "cash_rate_ann": cash_rate_ann,
        "dur_vol": dur_vol,
        "eq_vol": eq_vol,
        "idio_vol": idio_vol,
    }
    return prices, truth


def synthetic_daily(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Alias for :func:`synthetic_panel` (desk-wide generator naming convention)."""
    return synthetic_panel(**kwargs)

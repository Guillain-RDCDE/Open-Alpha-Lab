"""Data layer for Study 925 — Front-End Trend (trend-following the short rate).

Two tapes, one shape (a date-indexed daily frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the duration ladder (SHY 1-3y, IEF 7-10y,
  TLT 20y+) and the cash proxy (BIL, the 1-3 month T-bill ETF), plus the **price-only**
  level of ``^IRX`` — the 13-week Treasury bill *discount yield in percent*, not a
  tradable price series. That distinction is carried through the whole study: ^IRX is a
  **signal input only**; every return comes from an ETF total-return close.
  ``fetch`` touches the network and caches parquet under the shared ``studies/_cache``
  (retry up to 4x); ``load_prices`` reads that cache **offline** and never imports
  yfinance. The whole test-suite runs with NO cache present (synthetic-only), so CI is
  green on a fresh checkout where ``_cache/`` is absent.

- ``synthetic_daily`` — a *deterministic, offline* generator. A short-rate path whose
  daily *increments* follow an AR(1), plus a long-duration and an intermediate-duration
  total-return leg priced off that rate (``r = carry - D * dy``) and a cash leg that
  accrues the prevailing short rate. The ``signal_strength`` knob is the AR(1)
  coefficient on the rate increments: at ``signal_strength=0`` the increments are i.i.d.,
  so a trailing rate change carries **no** information about the next one (the null — a
  trend rule must not win); at ``signal_strength=1`` rate moves are strongly persistent
  and trend-following the front end genuinely picks the right duration. Unconditional
  rate volatility is held fixed across the knob, so the two worlds differ only in
  *predictability*, not in *risk*.

The claim under test: the front of the curve trends — cutting cycles and hiking cycles
run for quarters, not days — so the sign of the recent change in the 3-month bill rate
should tell you whether to own duration (rates falling) or sit in bills (rates rising),
beating a static intermediate-duration holding on a risk-adjusted basis.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (a signal
formed on data through day ``t`` is acted on at day ``t+1``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# SHY / IEF / TLT = the 1-3y, 7-10y and 20y+ Treasury ETFs (the duration ladder);
# BIL = the 1-3 month T-bill ETF (the tradable cash leg); ^IRX = the 13-week bill
# discount yield in percent (the signal input, price-only, NOT tradable).
TICKERS = ("SHY", "IEF", "TLT", "BIL", "^IRX")

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "1999-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` makes the
    ETF ``close`` columns split- and distribution-adjusted **total return** — essential
    here, because a bond ETF's coupon *is* most of its return and a price-only TLT
    series would understate every arm of the race. ``^IRX`` has no distributions: its
    ``close`` is a yield level in percent and is used only to form the signal.
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
    """Read the cached daily closes OFFLINE into one aligned frame.

    Columns are named by ticker exactly as passed (so ``^IRX`` stays visibly different
    from the tradable ETF columns). Sliced to ``asof`` so the sample never creeps.
    Raises ``FileNotFoundError`` if any ticker is missing — the offline core and the
    test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached data for {tk} at {path}. "
                f"Call front_end_trend.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (planted rate-trend persistence)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 24,
    signal_strength: float = 1.0,   # AR(1) coefficient on daily rate increments
    rate_start: float = 3.0,        # starting short rate, percent
    rate_vol_bp: float = 4.0,       # unconditional sd of the daily rate change, bp
    dur_long: float = 17.0,         # duration of the long leg (TLT-like)
    dur_mid: float = 7.5,           # duration of the intermediate leg (IEF-like)
    carry_long_ann: float = 0.035,  # long leg's running yield (annual)
    carry_mid_ann: float = 0.030,   # intermediate leg's running yield (annual)
    idio_vol_ann: float = 0.035,    # duration-leg noise not explained by the front rate
    ar_max: float = 0.92,
    start: str = "2002-07-30",
    seed: int = 925,
) -> tuple[pd.DataFrame, dict]:
    """A daily (short rate, long-duration, mid-duration, cash) tape with a trend knob.

    The rate path is a random walk whose *increments* follow an AR(1) with coefficient
    ``phi = ar_max * signal_strength``. The innovation scale is rescaled by
    ``sqrt(1 - phi^2)`` so the unconditional daily rate volatility is the **same** at
    every ``signal_strength``: the two worlds differ only in how *predictable* the next
    rate move is from the last few months of rate moves.

    - ``signal_strength = 1`` → rate moves are strongly persistent; a trailing 3-month
      rate change genuinely forecasts the next one, so switching into duration when the
      front end is falling should beat a static duration holding.
    - ``signal_strength = 0`` → i.i.d. increments; the trailing change is pure noise and
      the switch rule must **not** beat static duration (the null).

    Duration legs are priced off the same rate path with textbook bond arithmetic
    (``r = carry/252 - D * dy`` plus idiosyncratic noise), and cash accrues the
    prevailing short rate — so the excess-of-cash race is meaningful on the synthetic
    tape too. Returns ``(frame, truth)``; deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    phi = float(np.clip(ar_max * signal_strength, 0.0, ar_max))
    sd_uncond = rate_vol_bp / 100.0            # bp -> percentage points
    sd_innov = sd_uncond * np.sqrt(max(1.0 - phi * phi, 1e-12))

    innov = rng.normal(0.0, sd_innov, n_days)
    dy = np.zeros(n_days)                      # daily rate change, percentage points
    prev = 0.0
    for i in range(n_days):
        prev = phi * prev + innov[i]
        dy[i] = prev

    rate = rate_start + np.cumsum(dy)
    # Keep the short rate in a plausible box; reflect rather than clip so the path keeps
    # moving (a pinned rate would silently kill the signal at the boundary).
    rate = np.abs(rate - 0.15) + 0.15
    rate = 8.0 - np.abs(8.0 - rate)
    dy_eff = np.diff(np.concatenate([[rate_start], rate]))

    idio_d = idio_vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    r_long = (carry_long_ann / TRADING_DAYS_PER_YEAR
              - dur_long * dy_eff / 100.0
              + rng.normal(0.0, idio_d, n_days))
    r_mid = (carry_mid_ann / TRADING_DAYS_PER_YEAR
             - dur_mid * dy_eff / 100.0
             + rng.normal(0.0, idio_d * 0.6, n_days))
    r_cash = np.maximum(rate, 0.0) / 100.0 / TRADING_DAYS_PER_YEAR

    frame = pd.DataFrame(
        {
            "rate": rate,
            "long": 100.0 * np.cumprod(1.0 + r_long),
            "mid": 100.0 * np.cumprod(1.0 + r_mid),
            "cash": 100.0 * np.cumprod(1.0 + r_cash),
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "phi": phi,
        "n_days": n_days,
        "n_years": n_years,
        "seed": seed,
        "rate_vol_bp": rate_vol_bp,
        "dur_long": dur_long,
        "dur_mid": dur_mid,
        "sd_innov": float(sd_innov),
        "rate_mean": float(rate.mean()),
        "dy_autocorr1": float(np.corrcoef(dy_eff[1:], dy_eff[:-1])[0, 1]),
    }
    return frame, truth


def synthetic_panel(
    n_seeds: int = 8,
    signal_strength: float = 0.0,
    seed0: int = 925,
    **kwargs,
) -> list[tuple[pd.DataFrame, dict]]:
    """A deterministic *panel* of synthetic worlds — the null / power sweep.

    Used to check that the detector fires on planted rate-trend persistence across
    seeds and stays quiet when there is none. Never supports a real-tape stamp.
    """
    return [
        synthetic_daily(signal_strength=signal_strength, seed=seed0 + i, **kwargs)
        for i in range(n_seeds)
    ]

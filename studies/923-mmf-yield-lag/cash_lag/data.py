"""Data layer for Study 923 — The Cash Lag (how fast cash vehicles reprice).

Two tapes, one shape (a date-indexed daily frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for four liquid cash vehicles — **BIL**
  (1-3 month T-bills), **SGOV** (0-3 month T-bills), **USFR** (floating-rate
  Treasury notes) and **SHV** (0-1 year Treasuries) — plus **^IRX**, the 13-week
  T-bill *discount yield quote* in percent. ``fetch`` touches the network and
  caches parquet under the shared ``studies/_cache/`` (retry up to 4x);
  ``load_prices`` reads that cache **offline** and never imports yfinance. The
  whole test-suite runs with NO cache present (synthetic-only), so CI is green on
  a fresh checkout.

  **Price-only vs total-return.** The four vehicle series are total-return
  (``auto_adjust=True``) — for a cash ETF the distribution *is* essentially the
  whole return, so a price-only series would be nearly flat and useless here.
  ^IRX is neither: it is a *yield quote*, not an investable index, and is used
  only as the right-hand-side rate variable.

- ``synthetic_panel`` / ``synthetic_daily`` — a *deterministic, offline*
  generator. A short-rate path drives a set of laddered bill portfolios with
  different weighted-average maturities (WAM). A vehicle of WAM ``w`` days earns
  the *average* rate of the last ``w`` days (its book yield is a moving average
  of the rate at which its holdings were bought) and takes a mark-to-market hit
  of ``duration x d(rate)`` on the day rates move. The ``signal_strength`` knob
  blends toward a null in which every vehicle collapses to the *same* (panel
  average) WAM — so there is no lag ordering and no duration *spread* left to
  detect — and the rate path is a pure random walk (no exploitable direction).
  Seeds are fixed, so tests are deterministic.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the
rate-direction signal formed on data through day ``t`` is acted on at ``t+1``).
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

# The four cash vehicles, ordered by published weighted-average maturity (short
# to long). USFR is a floating-rate note fund: its coupon resets weekly off the
# 13-week bill auction, so its *duration* is ~0 even though its maturities are long.
VEHICLES = ("SGOV", "BIL", "USFR", "SHV")

# The rate variable: ^IRX = 13-week T-bill discount yield, in percent. A quote,
# not a tradable total return.
RATE = "^IRX"

TICKERS = VEHICLES + (RATE,)

# Published WAM (weighted average maturity) in days, from the issuers' own fund
# pages. PROXY/ASSUMPTION: these are current, static figures used only to order
# the vehicles a priori and to seed the synthetic panel; the study never uses
# them in a return calculation. The tape measures the realised duration itself.
NOMINAL_WAM_DAYS = {"SGOV": 25, "BIL": 45, "USFR": 5, "SHV": 110}

# The headline window: the common BIL / USFR / SHV / ^IRX sample. USFR's 2014
# inception gates it. SGOV (2020) joins as a fourth-vehicle cross-check.
HEADLINE_START = "2014-02-04"
HEADLINE_VEHICLES = ("BIL", "USFR", "SHV")

# The benchmark cash leg. Every excess return in this study is excess-of-BIL,
# the desk's standard cash proxy — so "excess-of-cash" here is literal.
BENCH = "BIL"

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the
# partial current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2007-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so
    the vehicle ``close`` columns are dividend-adjusted total return — the whole
    point for a cash ETF, whose entire return is the distribution. ^IRX comes
    back as a yield level in percent and is passed through unchanged.
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
    """Read cached daily closes OFFLINE into one aligned frame.

    Returns a frame indexed by date with one column per ticker (vehicles = adjusted
    total-return closes; ``^IRX`` = the yield quote in percent), sliced to ``asof``
    so the sample never creeps. Raises ``FileNotFoundError`` if any ticker is
    missing — the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call cash_lag.data.fetch() once to populate the cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    return df[df.index <= pd.Timestamp(asof)]


def headline_frame(
    prices: pd.DataFrame,
    vehicles=HEADLINE_VEHICLES,
    start: str = HEADLINE_START,
) -> pd.DataFrame:
    """Slice the loaded frame to the common vehicle + rate window used for the headline."""
    cols = list(vehicles) + [RATE]
    df = prices[cols].dropna()
    return df[df.index >= pd.Timestamp(start)]


def fingerprint(prices: pd.DataFrame) -> str:
    """Short content fingerprint of a frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(prices.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted lag + duration)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 12,
    wam_days=(5, 25, 45, 110),
    names=("FAST", "SHORT", "MID", "LONG"),
    signal_strength: float = 1.0,
    rate_start: float = 2.0,          # starting short rate, percent
    rate_vol_bp: float = 3.0,         # daily s.d. of the rate innovation, bp
    rate_mr: float = 0.002,           # daily pull back toward ``rate_start``
    trend_phi: float = 0.92,          # AR(1) on rate *changes* at full strength
    idio_bp: float = 0.15,            # daily s.d. of fund-specific NAV noise, bp
    start: str = "2010-01-04",
    seed: int = 923,
) -> tuple[pd.DataFrame, dict]:
    """A daily panel of laddered-bill NAVs driven by one short-rate path.

    Mechanics, deliberately transparent:

    - the short rate ``r(t)`` (percent) is a floored, mean-reverting walk whose
      *changes* follow an AR(1) with persistence ``trend_phi * signal_strength``
      — so at full strength rate moves trend (a switch rule can profit) and at
      zero strength they are unpredictable white noise;
    - a vehicle of WAM ``w`` days holds a ladder bought over the last ``w`` days,
      so its **book yield** is the ``w``-day moving average of ``r`` — that *is*
      the repricing lag, and it is longer for longer ``w``;
    - its **duration** is ``w / 2 / 252`` years, so a rate move marks its NAV by
      ``-duration * dr`` on the day — bigger for longer ``w``.

    The ``signal_strength`` knob blends toward the null: at ``0`` every WAM
    collapses to the panel *average* (so the vehicles share one duration and one
    lag — no ordering left to detect) *and* the rate path loses its trend, so
    neither the lag ordering nor the switch rule has anything to find. Each
    vehicle also carries a small fund-specific NAV noise (``idio_bp``, standing in
    for expense drag, discount-margin wobble and trading noise), always on, so the
    null is a genuine noise world rather than a degenerate one in which every
    series is literally identical.

    Returns ``(prices, truth)``. ``prices`` carries one total-return NAV column
    per vehicle plus ``^IRX`` (the rate quote, in percent, matching the real
    frame's column name). Deterministic given ``seed``.
    """
    ss = float(np.clip(signal_strength, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a few thousand business days stays far inside pandas' ns horizon.
    dates = pd.bdate_range(start=start, periods=n_days)

    phi = trend_phi * ss
    eps = rng.normal(0.0, rate_vol_bp / 100.0, n_days)  # in percentage points
    dr = np.zeros(n_days)
    r = np.zeros(n_days)
    r[0] = rate_start
    for i in range(1, n_days):
        dr[i] = phi * dr[i - 1] + eps[i] + rate_mr * (rate_start - r[i - 1])
        r[i] = max(0.0, r[i - 1] + dr[i])
    dr = np.diff(r, prepend=r[0])

    w_bar = float(np.mean(wam_days))
    eff_wam = [max(1, int(round(w_bar + ss * (w - w_bar)))) for w in wam_days]
    rate_s = pd.Series(r, index=dates)

    cols = {}
    truth_wam = {}
    for j, (name, w) in enumerate(zip(names, eff_wam)):
        book_yield = rate_s.rolling(w, min_periods=1).mean().to_numpy()
        duration = w / 2.0 / TRADING_DAYS_PER_YEAR          # years
        carry = book_yield / 100.0 / TRADING_DAYS_PER_YEAR  # daily accrual
        mark = -duration * dr / 100.0                       # daily MTM
        idio = np.random.default_rng(seed * 1000 + j).normal(0.0, idio_bp * 1e-4, n_days)
        ret = carry + mark + idio
        ret[0] = 0.0
        cols[name] = 100.0 * np.cumprod(1.0 + ret)
        truth_wam[name] = {"wam_days": w, "duration_years": duration}

    cols[RATE] = r
    prices = pd.DataFrame(cols, index=pd.DatetimeIndex(dates, name="date"))
    truth = {
        "signal_strength": ss,
        "names": list(names),
        "nominal_wam_days": list(wam_days),
        "effective_wam_days": eff_wam,
        "per_vehicle": truth_wam,
        "trend_phi_effective": phi,
        "idio_bp": idio_bp,
        "n_days": n_days,
        "n_years": n_years,
        "seed": seed,
        "rate_mean": float(rate_s.mean()),
        # Ties are broken on the NOMINAL WAM so the same two vehicles are raced at
        # every signal strength (at ss=0 every effective WAM is the panel average,
        # so an argmin on the effective values would collapse to a single name).
        "fastest": names[int(np.argmin(wam_days))],
        "slowest": names[int(np.argmax(wam_days))],
    }
    return prices, truth


def synthetic_daily(
    wam_days: int = 45,
    signal_strength: float = 1.0,
    seed: int = 923,
    **kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Single-vehicle convenience wrapper around :func:`synthetic_panel`.

    Returns a two-column frame (``VEHICLE`` NAV and ``^IRX``) plus the same truth
    dict — handy for the one-vehicle lag tests that do not need a whole panel.
    """
    prices, truth = synthetic_panel(
        wam_days=(wam_days,), names=("VEHICLE",),
        signal_strength=signal_strength, seed=seed, **kwargs,
    )
    return prices, truth

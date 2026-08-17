"""Data layer for Study 921 — Bill Ladder vs ETF.

Two tapes, one shape (a date-indexed daily frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the three cash ETFs (BIL, SGOV, SHV) plus
  ``^IRX``, the CBOE 13-week Treasury-bill rate index. ``^IRX`` is *not* a price: its
  ``close`` is a **rate in percent**, and it is carried in the same frame purely because
  the cache convention is one parquet per symbol. ``fetch`` touches the network and
  caches parquet under the **shared** ``studies/_cache`` (retry up to 4x); ``load_prices``
  reads that cache **offline** and never imports yfinance. The whole test-suite runs with
  NO cache present (synthetic-only), so CI is green on a fresh checkout.

- ``synthetic_daily`` — a *deterministic, offline* generator. A mean-reverting
  (Ornstein-Uhlenbeck) short-rate path, the matching 13-week discount quote, and a
  synthetic cash ETF that accrues that short rate **minus a fee** and carries a little
  mark-to-market bounce. The ``signal_strength`` knob scales the planted fee: at
  ``signal_strength=0`` the ETF is free (the null — a ladder must NOT beat it), at
  ``signal_strength=1`` the ETF charges the full ``fee_bps`` (the planted effect a ladder
  must recover). Seed is fixed → tests are deterministic.

The question this study asks: a rolling ladder of 3-month bills, bought at issue and held
to maturity, is the thing a cash ETF *is*. Does running it yourself beat paying someone
0.09–0.15% a year to run it for you — and by how much, once realistic per-auction friction
and reinvestment idle days are charged?

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the ^IRX quote
observed at the close of day ``t`` prices a bill bought on day ``t+1``.
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a per-study one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

DAYS_PER_YEAR = 365.0
TRADING_DAYS_PER_YEAR = 252

# BIL = SPDR 1-3 Month T-Bill; SGOV = iShares 0-3 Month Treasury; SHV = iShares Short
# Treasury (<=1yr, i.e. NOT a pure bill fund — kept as the duration control).
# ^IRX = CBOE 13-week T-bill rate index (a rate in percent, not a price).
TICKERS = ("BIL", "SGOV", "SHV", "^IRX")
ETFS = ("BIL", "SGOV", "SHV")
RATE_SYMBOL = "^IRX"

# PROXY / ASSUMPTION — sponsor-published net expense ratios, in bps per year, as of
# 2026. These are NOT tape: they are hardcoded reference numbers used only to ask
# "is the measured gap the size of the fee?". They are never used to compute a return.
# SGOV in particular has been repriced since its 2020 launch, so its effective
# whole-sample fee is lower than the current sticker; treat it as a band, not a point.
EXPENSE_RATIO_BPS = {"BIL": 13.54, "SGOV": 9.0, "SHV": 15.0}

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
    start: str = "1990-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the ETF
    ``close`` columns are **total return** (dividends reinvested) — essential here, since a
    bill ETF's entire return arrives as monthly distributions and its price alone is flat.
    ``^IRX`` is unaffected by adjustment: its close is a quoted rate in percent.
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
    """True iff every symbol's parquet is present in the cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read the cached daily closes OFFLINE into one aligned frame.

    Columns are the raw symbols: ``BIL``/``SGOV``/``SHV`` are total-return close levels,
    ``^IRX`` is the 13-week bill rate **in percent**. Sliced to ``asof`` so the sample never
    creeps. Raises ``FileNotFoundError`` if any symbol is missing — the offline core and the
    test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached data for {tk} at {path}. "
                f"Call bill_ladder.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "date"
    df = df[df.index <= pd.Timestamp(asof)]
    return df


def fingerprint(frame: pd.DataFrame) -> str:
    """Short content fingerprint of a data frame, for the as-of data stamp."""
    arr = np.ascontiguousarray(frame.to_numpy(dtype=float))
    arr = np.nan_to_num(arr, nan=0.0)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted ETF fee)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 20,
    theta: float = 0.025,            # long-run short rate (2.5%)
    kappa: float = 1.2,              # annual mean-reversion speed
    sigma: float = 0.010,            # annual vol of the short rate
    r0: float = 0.02,
    fee_bps: float = 13.5,           # the planted ETF expense ratio, bps/yr
    signal_strength: float = 1.0,    # 0 = free ETF (null), 1 = full planted fee
    nav_noise_bps: float = 1.5,      # daily mark-to-market bounce on the ETF close
    start: str = "2006-01-02",
    seed: int = 921,
) -> tuple[pd.DataFrame, dict]:
    """A synthetic short-rate world with a cash ETF that charges a *known* fee.

    Columns of the returned frame:

    - ``irx`` — the simulated 13-week bill **discount** quote in percent, produced by
      inverting the bond-equivalent yield through the same 91/360 → 91/365 arithmetic
      ``strategy.discount_to_bey`` uses, so the round-trip is exact.
    - ``etf`` — a cash-ETF total-return index that accrues the *current* short rate,
      pays ``fee_bps * signal_strength`` per year, and carries a small mean-zero NAV
      bounce (the bid-offer / rounding noise a real bill ETF's close shows).

    The ``signal_strength`` knob is the planted effect:

    - ``signal_strength = 1`` → the ETF charges the full fee; a ladder run off ``irx``
      must recover a gap of roughly ``fee_bps`` bps/yr.
    - ``signal_strength = 0`` → the ETF is free; the ladder must show **no** gap (the null).

    Deterministic given ``seed``. Returns ``(frame, truth)``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with a few thousand bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)
    dt_days = np.diff(dates.to_numpy().astype("datetime64[D]").astype(int), prepend=0)
    dt_days[0] = 1.0
    dt_days = dt_days.astype(float)

    # Ornstein-Uhlenbeck short rate, floored just above zero (bills do not go negative
    # for long, and a floor keeps the discount arithmetic well-behaved).
    r = np.empty(n_days)
    r[0] = r0
    for i in range(1, n_days):
        h = dt_days[i] / DAYS_PER_YEAR
        r[i] = r[i - 1] + kappa * (theta - r[i - 1]) * h + sigma * np.sqrt(h) * rng.normal()
        if r[i] < 0.0005:
            r[i] = 0.0005

    # Bond-equivalent yield -> price -> discount quote (the exact inverse of
    # strategy.discount_to_bey with tenor 91).
    tenor = 91.0
    price = 1.0 / (1.0 + r * tenor / DAYS_PER_YEAR)
    disc = (1.0 - price) * 360.0 / tenor

    fee_ann = fee_bps * 1e-4 * float(signal_strength)
    accrual = (r - fee_ann) / DAYS_PER_YEAR * dt_days
    noise = rng.normal(0.0, nav_noise_bps * 1e-4, n_days)
    noise[0] = 0.0
    etf_ret = accrual + noise - np.concatenate([[0.0], noise[:-1]])  # bounce, not drift
    etf = np.cumprod(1.0 + etf_ret) * 100.0

    frame = pd.DataFrame(
        {"irx": disc * 100.0, "etf": etf},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": float(signal_strength),
        "fee_bps": float(fee_bps),
        "fee_bps_effective": float(fee_bps * signal_strength),
        "theta": theta, "kappa": kappa, "sigma": sigma, "r0": r0,
        "nav_noise_bps": nav_noise_bps,
        "n_years": n_years, "n_days": n_days, "seed": seed,
        "mean_rate": float(r.mean()),
    }
    return frame, truth

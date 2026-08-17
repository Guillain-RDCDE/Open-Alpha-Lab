"""Data layer for Study 958 — Spot ETF Basis (the futures wrapper's carry after Jan-2024).

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the futures wrapper (``BITO``), the two
  largest spot wrappers (``IBIT``, ``FBTC``), the spot coin itself (``BTC-USD``) and a
  cash proxy (``BIL``, the 1-3 month T-bill ETF). ``fetch`` touches the network and
  caches parquet under the **shared** ``studies/_cache`` (retry up to 4x);
  ``load_prices`` reads that cache **offline** and never imports yfinance. The whole
  test-suite runs with NO cache present (synthetic-only), so CI is green on a fresh
  checkout where ``_cache/`` is git-ignored and absent.

  Total return matters more here than almost anywhere else on the desk: BITO pays very
  large monthly distributions, so a price-only tape would report a wrapper "drag" that
  is mostly just cash handed back to the holder.

- ``synthetic_panel`` / ``synthetic_daily`` — a *deterministic, offline* generator. A
  24/7 spot path, a spot-ETF wrapper that erodes at its stated fee, and a futures
  wrapper that erodes at ``fee + basis - collateral yield``; the daily closes of the
  three wrappers are stamped at a *different hour* from the spot close, which plants
  the exact timestamp artefact the real tape suffers from. The ``signal_strength`` knob
  scales a **planted compression** of the basis at the event date: at
  ``signal_strength=0`` the basis is flat through the event (the null — the era test
  must stay quiet); at ``signal_strength=1`` the compression is large and the estimator
  must recover it.

Non-tape inputs used downstream are **assumptions**, not measurements, and are labelled
as such wherever they appear: the published expense ratios in ``FEES`` (BITO 0.95%,
IBIT 0.25%, FBTC 0.25% headline, both spot funds with launch fee waivers), the
short-borrow rate on the harvest leg, and the event date itself.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the pair's
rebalance weights are formed at the close of day ``t`` and applied from ``t+1``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache (studies/_cache), not a study-local one.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# BITO = futures wrapper (CME front-month, fully collateralised); IBIT / FBTC = the two
# largest spot wrappers; BTC-USD = the coin; BIL = 1-3M T-bill (cash / collateral proxy).
TICKERS = ("BITO", "IBIT", "FBTC", "BTC-USD", "BIL")

# Study-wide as-of: the last COMPLETE calendar month at build time (drop the partial
# current month so the sample never creeps between reruns).
AS_OF = "2026-06-30"

# The event: the first trading day of the US spot bitcoin ETFs (SEC approval effective
# 2024-01-10, first session 2024-01-11). An ASSUMPTION only in the trivial sense that it
# is a calendar fact rather than an estimate.
LAUNCH = "2024-01-11"

# PROXY / ASSUMPTION — published headline expense ratios (annual, decimal). These are
# prospectus numbers, not measured from the tape; every result that uses them is swept
# over a fee grid in ``strategy.fee_sweep``. Note IBIT waived to 0.12% on the first $5bn
# for 12 months and FBTC waived to 0.00% until 2024-07-31, so the *realised* spot-wrapper
# drag should come in at or just under the headline — which is exactly the calibration
# check this study runs.
FEES = {"BITO": 0.0095, "IBIT": 0.0025, "FBTC": 0.0025}


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2014-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and distribution-adjusted total return — essential for
    BITO, whose monthly distributions have at times run above 50% annualised.
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

    Returns a frame indexed by date with one column per ticker, sliced to ``asof`` so
    the sample never creeps. Columns start at wildly different dates (BITO 2021-10-20,
    IBIT/FBTC 2024-01-11) — callers intersect the pair they need. Raises
    ``FileNotFoundError`` if any ticker is missing: the offline core and the test-suite
    never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call etf_basis.data.fetch() once to populate the shared cache."
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


def cash_yield(prices: pd.DataFrame, lo=None, hi=None, col: str = "BIL") -> float:
    """Annualised (geometric) total return of the cash leg over ``[lo, hi]``.

    Used as the collateral-yield PROXY in the implied-basis decomposition: BITO holds
    T-bills and cash against its futures, so its total return picks up roughly the bill
    yield on top of the futures P&L. BIL's own 0.1356% fee makes this a *slightly*
    conservative stand-in for the fund's gross collateral yield.
    """
    s = prices[col].dropna()
    if lo is not None:
        s = s[s.index >= pd.Timestamp(lo)]
    if hi is not None:
        s = s[s.index <= pd.Timestamp(hi)]
    if len(s) < 2:
        return float("nan")
    years = (s.index[-1] - s.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0)


# --------------------------------------------------------------------------- #
# Synthetic tape — the deterministic offline core (a planted basis + compression)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_years: int = 5,
    basis_pre: float = 0.09,          # annualised front-month basis before the event
    compression: float = 0.07,        # size of the planted post-event compression
    signal_strength: float = 1.0,     # 0 = no compression (the null), 1 = full
    fee_futures: float = 0.0095,      # futures-wrapper expense ratio
    fee_spot: float = 0.0025,         # spot-wrapper expense ratio
    cash_rate_ann: float = 0.045,     # collateral yield credited to the futures wrapper
    vol_spot: float = 0.55,           # annualised spot vol (bitcoin-like)
    drift_spot: float = 0.30,         # annualised spot drift
    offset_vol: float = 0.030,        # sd of the close-timestamp offset (16h of spot move)
    event_frac: float = 0.5,          # where in the sample the event lands
    start: str = "2021-10-20",
    seed: int = 958,
) -> tuple[pd.DataFrame, dict]:
    """A daily panel: spot, a spot-ETF wrapper, a futures wrapper, and a cash index.

    The generator plants exactly the three things this study measures:

    1. a **fee drag** on the spot wrapper (``fee_spot``), which is what calibrates the
       ruler on the real tape;
    2. a **carry drag** on the futures wrapper — ``fee_futures + basis - cash_rate_ann``
       — the toll the futures roll pays away every day the curve is in contango;
    3. a **timestamp offset**: the wrapper closes are stamped at a different hour from
       the spot close, so ``log(wrapper) - log(spot)`` carries an i.i.d. offset term of
       sd ``offset_vol`` on *top* of the drag. This is the real tape's problem — a 24/7
       coin quoted at 00:00 UTC against ETFs marked at 16:00 New York — and it is why a
       plain mean-of-daily-differences estimator is far noisier than a trend slope.

    ``signal_strength`` scales the planted compression at the event date:
    ``basis_post = basis_pre - signal_strength * compression``. At ``signal_strength=0``
    the basis is unchanged through the event and the era test must stay quiet; at 1 the
    compression is large and the estimator must recover it.

    Returns ``(prices, truth)`` where ``prices`` has columns ``spot``, ``spot_etf``,
    ``futures_etf`` and ``cash``, and ``truth`` records the planted drags in %/yr.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with a few thousand daily bars stays inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)
    i_event = int(round(event_frac * n_days))

    basis_post = basis_pre - signal_strength * compression
    basis = np.where(np.arange(n_days) < i_event, basis_pre, basis_post)

    d = 1.0 / TRADING_DAYS_PER_YEAR
    shock = rng.normal(drift_spot * d - 0.5 * vol_spot ** 2 * d,
                       vol_spot * np.sqrt(d), n_days)
    log_spot = np.cumsum(shock)

    # Wrapper log paths: the spot path minus the wrapper's own drag. Carry and fees
    # accrue in CALENDAR time (a basis is a calendar-time carry, an expense ratio is
    # charged over weekends too), which is also the clock the trend estimator regresses
    # on — so the planted rates and the recovered slopes are directly comparable.
    dt = np.diff(np.asarray((dates - dates[0]).days, dtype=float) / 365.25, prepend=0.0)
    drag_spot_etf = fee_spot
    drag_fut_etf = fee_futures + basis - cash_rate_ann
    log_spot_etf = log_spot - np.cumsum(drag_spot_etf * dt)
    log_fut_etf = log_spot - np.cumsum(drag_fut_etf * dt)

    # The timestamp artefact: every wrapper close is the spot path plus the same
    # intraday offset (they all mark at 16:00 New York); the spot column marks elsewhere.
    offset = rng.normal(0.0, offset_vol, n_days)
    spot = 100.0 * np.exp(log_spot)
    spot_etf = 50.0 * np.exp(log_spot_etf + offset)
    fut_etf = 20.0 * np.exp(log_fut_etf + offset)
    cash = np.exp(np.cumsum(np.log1p(cash_rate_ann) * dt))

    prices = pd.DataFrame(
        {"spot": spot, "spot_etf": spot_etf, "futures_etf": fut_etf, "cash": cash},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "basis_pre": basis_pre,
        "basis_post": float(basis_post),
        "compression": float(basis_pre - basis_post),
        "fee_futures": fee_futures,
        "fee_spot": fee_spot,
        "cash_rate_ann": cash_rate_ann,
        "event_date": str(dates[i_event].date()),
        "n_days": n_days,
        "seed": seed,
        "offset_vol": offset_vol,
        # The quantities the estimators must recover, in %/yr (negative = a drag).
        "drag_spot_etf_pct": -fee_spot * 100.0,
        "drag_fut_pre_pct": -(fee_futures + basis_pre - cash_rate_ann) * 100.0,
        "drag_fut_post_pct": -(fee_futures + basis_post - cash_rate_ann) * 100.0,
        "drag_change_pct": float(signal_strength * compression * 100.0),
    }
    return prices, truth


def synthetic_daily(**kwargs) -> tuple[pd.DataFrame, dict]:
    """Convenience two-column view of :func:`synthetic_panel` — the headline pair.

    Returns the same ``truth`` dict with a frame carrying only ``futures_etf`` and
    ``spot`` (the pair the era test runs on), for the tests and demos that do not need
    the spot wrapper or the cash leg.
    """
    prices, truth = synthetic_panel(**kwargs)
    return prices[["futures_etf", "spot"]].copy(), truth

"""Data layer for Study 943 — Reset Frequency (monthly- vs daily-reset leverage).

Two tapes, one shape (a date-indexed daily frame):

- ``fetch`` / ``load_prices`` — daily closes from Yahoo! Finance (``yfinance``,
  ``auto_adjust=True``) for the index (SPY), the two daily-reset leveraged funds
  (SSO = 2x S&P 500, UPRO = 3x S&P 500), a cash proxy (BIL, the 1-3 month T-bill
  ETF) and the financing benchmark (``^IRX``, the 13-week T-bill discount yield).
  ``fetch`` touches the network and caches parquet in the **shared** desk cache
  ``studies/_cache`` (retry up to 4x); ``load_prices`` reads that cache **offline**
  and never imports yfinance. The whole test-suite runs with NO cache present
  (synthetic-only), so CI is green on a fresh checkout.

  **Label discipline.** SPY / SSO / UPRO / BIL closes are **total return** (dividends
  and distributions reinvested, ``auto_adjust=True``). ``^IRX`` is **not** a price at
  all: it is a *quoted annualised discount yield in percent* (e.g. ``5.25`` = 5.25%).
  It is carried in the same frame for convenience and converted to a daily financing
  rate in :func:`reset_freq.strategy.financing_rate` — never treated as a close.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators. A
  daily log-return tape with an AR(1) serial-dependence term whose **sign** is the
  choppiness knob: ``phi < 0`` = mean-reverting (choppy, reversal-rich) and ``phi > 0``
  = trending (momentum-rich). ``signal_strength`` scales ``phi`` toward zero, so
  ``signal_strength = 0`` is a plain iid random walk — the null, on which the
  monthly-minus-daily reset gap must be centred at zero.

The question this study asks: retail folklore blames the *daily reset* ("volatility
decay") for leveraged-ETF underperformance and proposes a **monthly** reset as the fix.
Would a monthly-reset replication — SPY on margin at ``^IRX`` + a financing spread,
levered back to 2x/3x once a month — actually have beaten SSO and UPRO on the real,
costed, excess-of-cash tape? And which reset frequency wins in trending versus choppy
markets?

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the exposure
in force on day ``t`` is decided from information through day ``t-1``).
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

TRADING_DAYS_PER_YEAR = 252

# SPY = the underlying index tape; SSO / UPRO = the 2x / 3x DAILY-reset funds;
# BIL = the 1-3M T-bill ETF (the cash leg of every excess-of-cash race);
# ^IRX = the 13-week T-bill discount yield (the financing benchmark, a RATE not a price).
TICKERS = ("SPY", "SSO", "UPRO", "BIL", "^IRX")

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily closes, cache-only by default
# --------------------------------------------------------------------------- #
def safe_name(ticker: str) -> str:
    """Cache-safe column / filename for a ticker (``^IRX`` -> ``IRX``)."""
    return ticker.replace("=", "").replace("^", "").replace("/", "")


def _cache_path(ticker: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"prices_{safe_name(ticker)}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2004-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily closes for ``tickers`` and cache each as parquet in the shared cache.

    Network-only; run once to populate the cache. ``auto_adjust=True`` so the ETF
    ``close`` columns are split- and distribution-adjusted **total return** — essential
    because BIL's whole contribution is its yield and SSO/UPRO pay distributions. The
    ``^IRX`` series is a *yield in percent*, unaffected by adjustment.
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
    """True iff every ticker's parquet is present in the shared cache (offline-testable)."""
    return all(os.path.exists(_cache_path(tk, cache_dir)) for tk in tickers)


def load_prices(
    tickers=TICKERS,
    cache_dir: str = DEFAULT_CACHE,
    asof: str = AS_OF,
) -> pd.DataFrame:
    """Read the cached daily series OFFLINE into one aligned frame.

    Columns are the cache-safe ticker names (``^IRX`` becomes ``IRX``); the ETF columns
    are total-return closes, ``IRX`` is a percent yield. Sliced to ``asof`` so the sample
    never creeps. Raises ``FileNotFoundError`` if any ticker is missing — the offline core
    and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached series for {tk} at {path}. "
                f"Call reset_freq.data.fetch() once to populate the shared cache."
            )
        s = pd.read_parquet(path)["close"]
        s.index = pd.to_datetime(s.index)
        cols[safe_name(tk)] = s
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
# Synthetic tape — the deterministic offline core (a tunable choppiness knob)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 20,
    drift_ann: float = 0.08,        # annualised drift of the underlying
    vol_ann: float = 0.18,          # annualised vol of the underlying
    phi: float = -0.15,             # AR(1) on daily log returns: <0 choppy, >0 trending
    signal_strength: float = 1.0,   # scales phi toward 0; 0 = iid random walk (the null)
    cash_rate_ann: float = 0.03,    # the cash / financing benchmark rate
    start: str = "2006-01-03",
    seed: int = 943,
) -> tuple[pd.DataFrame, dict]:
    """A daily underlying tape with tunable *choppiness*, plus a cash accrual leg.

    Serial dependence is planted as a stationary AR(1) on daily log returns with
    coefficient ``phi_eff = signal_strength * phi``:

    - ``phi_eff < 0`` — **choppy**: today's move partly reverses tomorrow. Rebalancing
      leverage *daily* then systematically buys high and sells low, so a monthly reset
      should win.
    - ``phi_eff > 0`` — **trending**: moves persist. Daily resetting compounds the trend
      (positive convexity), so the daily reset should win.
    - ``signal_strength = 0`` — **the null**: iid log returns, no reversal and no
      persistence, so the monthly-minus-daily gap must be centred at zero.

    The innovation scale is rescaled by ``sqrt(1 - phi_eff**2)`` so total return variance
    is held fixed as the knob turns — the comparison isolates *path shape*, not vol.

    Returns ``(prices, truth)`` where ``prices`` has columns ``asset`` (the underlying
    total-return close) and ``cash`` (a cash accrual index), plus ``truth`` recording the
    planted parameters and the realised first-order autocorrelation. Deterministic.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a few thousand business days stay far inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    phi_eff = float(np.clip(signal_strength * phi, -0.9, 0.9))
    mu = drift_ann / TRADING_DAYS_PER_YEAR
    sigma = vol_ann / np.sqrt(TRADING_DAYS_PER_YEAR)
    eps_scale = sigma * np.sqrt(1.0 - phi_eff ** 2)

    eps = rng.normal(0.0, eps_scale, n_days)
    dev = np.empty(n_days)
    dev[0] = eps[0] / np.sqrt(max(1.0 - phi_eff ** 2, 1e-12))
    for i in range(1, n_days):
        dev[i] = phi_eff * dev[i - 1] + eps[i]
    log_ret = mu + dev

    asset = 100.0 * np.exp(np.cumsum(log_ret))
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash_idx = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(
        {"asset": asset, "cash": cash_idx},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    ac1 = float(np.corrcoef(log_ret[1:], log_ret[:-1])[0, 1])
    truth = {
        "phi": phi,
        "phi_eff": phi_eff,
        "signal_strength": signal_strength,
        "drift_ann": drift_ann,
        "vol_ann": vol_ann,
        "cash_rate_ann": cash_rate_ann,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "realised_ac1": ac1,
        "realised_vol_ann": float(log_ret.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)),
    }
    return prices, truth


def synthetic_panel(
    phis=(-0.15, 0.0, 0.15),
    n_seeds: int = 4,
    base_seed: int = 943,
    **kwargs,
) -> dict[tuple[float, int], tuple[pd.DataFrame, dict]]:
    """A small deterministic panel of synthetic worlds keyed by ``(phi, seed)``.

    Used by the tests and the notebooks to sweep the choppiness knob across seeds — the
    planted effect must appear with the right *sign* at every ``phi`` and vanish on the
    iid null. Purely offline.
    """
    out: dict[tuple[float, int], tuple[pd.DataFrame, dict]] = {}
    for p in phis:
        for s in range(n_seeds):
            out[(float(p), base_seed + s)] = synthetic_daily(
                phi=float(p), seed=base_seed + s, **kwargs
            )
    return out

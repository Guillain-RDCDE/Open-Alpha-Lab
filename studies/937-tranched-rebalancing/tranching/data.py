"""Data layer for Study 937 — Tranches (overlapping-portfolio rebalancing).

Two tapes, one shape (a date-indexed daily frame of total-return closes):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the risky sleeve (SPY), the defensive
  sleeve (IEF, 7-10y Treasuries), a tradable cash ETF (BIL) and the 13-week T-bill
  discount yield (``^IRX``) used as the long-history cash **proxy**. ``fetch``
  touches the network and caches parquet under the shared ``studies/_cache``
  (retry up to 4x); ``load_prices`` reads that cache **offline** and never imports
  yfinance. The whole test-suite runs with NO cache present (synthetic only), so
  CI is green on a fresh checkout.

- ``synthetic_daily`` — a *deterministic, offline* generator producing a risky leg,
  a defensive leg and a cash accrual leg. A two-state Markov bull/bear regime drives
  the risky leg; the ``signal_strength`` knob blends the world toward a flat null.
  At ``signal_strength=0`` a monthly trend sleeve has nothing to read (the null: the
  sleeve must earn nothing, yet the *timing-luck cone* must still be there, because
  the cone is an artefact of the rebalance date, not of edge). At
  ``signal_strength=1`` the regime is real and the sleeve genuinely helps.

The study's question is not whether the sleeve makes money. It is what happens to
the *dispersion of outcomes across the arbitrary choice of rebalance date* when the
same rule is run as N overlapping tranches instead of one — and what that fix costs
in turnover.

No look-ahead is baked in here — that discipline lives in ``strategy.py``: the state formed
from closes through the rebalance day ``t`` is the position that earns the return of ``t+1``
(one ``shift``, the house convention; execution at the ``t`` close). ``verify.py`` also
re-runs the whole study with a second day of delay and gets the same answer.

PROXY / ASSUMPTION note. ``^IRX`` is a *quoted discount yield*, not a tradable fund;
we accrue it daily (``yield/100/252``) to build a cash index back to 2002 so the
excess-of-cash race can use the full SPY-IEF window. The tradable BIL total return is
carried as a cross-check on its own (2007+) window; the two agree closely.
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

# Column name -> Yahoo! symbol. SPY = risky sleeve, IEF = defensive sleeve,
# BIL = tradable cash ETF (cross-check), IRX = 13-week bill yield (cash proxy).
TICKERS = ("SPY", "IEF", "BIL", "IRX")
YAHOO_SYMBOL = {"SPY": "SPY", "IEF": "IEF", "BIL": "BIL", "IRX": "^IRX"}

# Study-wide as-of: the last COMPLETE calendar month at build time.
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
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

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and dividend-adjusted **total return** for the funds
    (SPY, IEF, BIL). ``^IRX`` is an index of *yields*, so its close is a percentage
    figure, not a price — it is turned into a cash accrual index in ``strategy``.
    """
    import yfinance as yf  # lazy: only when we actually go to the network

    out: dict[str, pd.DataFrame] = {}
    os.makedirs(cache_dir, exist_ok=True)
    for tk in tickers:
        sym = YAHOO_SYMBOL.get(tk, tk)
        raw = None
        for _ in range(retries):
            try:
                raw = yf.download(
                    sym, start=start, end=end, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is not None and len(raw) > 0:
                    break
            except Exception:
                time.sleep(2.0)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"yfinance returned no data for {sym}")
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

    Returns a frame indexed by date with one column per ticker, sliced to ``asof`` so
    the sample never creeps. Raises ``FileNotFoundError`` if any ticker is missing —
    the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call tranching.data.fetch() once to populate the shared cache."
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
# Synthetic tape — the deterministic offline core
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 24,
    drift_bull: float = 0.16,        # annualised risky drift in the bull regime
    drift_bear: float = -0.30,       # annualised risky drift in the bear regime
    vol_bull: float = 0.12,
    vol_bear: float = 0.30,
    p_bull_to_bear: float = 0.025,   # monthly transition probabilities
    p_bear_to_bull: float = 0.05,   # ~20-month bear — long enough for a slow filter
    safe_drift: float = 0.035,       # annualised defensive-sleeve drift
    safe_vol: float = 0.06,
    signal_strength: float = 1.0,    # 0 = flat null, 1 = full regime separation
    start: str = "2002-01-02",
    seed: int = 937,
    cash_rate_ann: float = 0.02,
) -> tuple[pd.DataFrame, dict]:
    """A daily (risky, safe, cash) tape with a two-state Markov regime on the risky leg.

    The ``signal_strength`` knob blends the two-regime world toward a flat null:

    - ``signal_strength = 1`` → full separation; a monthly 200-day trend sleeve can see
      the bear regime and rotate into the defensive leg (the planted effect).
    - ``signal_strength = 0`` → drift and vol collapse to a single state; the sleeve
      carries no information (the null — it must not earn anything). The *timing-luck
      cone* must nevertheless still be present, since it is an artefact of the arbitrary
      rebalance date rather than of edge.

    Returns ``(prices, truth)`` where ``prices`` has columns ``risky``, ``safe`` and
    ``cash`` (all accumulation indices). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * TRADING_DAYS_PER_YEAR
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    stationary_bear = p_bull_to_bear / (p_bull_to_bear + p_bear_to_bull)
    avg_vol = (1 - stationary_bear) * vol_bull + stationary_bear * vol_bear
    avg_drift = (1 - stationary_bear) * drift_bull + stationary_bear * drift_bear
    vol_bull_eff = (1 - signal_strength) * avg_vol + signal_strength * vol_bull
    vol_bear_eff = (1 - signal_strength) * avg_vol + signal_strength * vol_bear
    drift_bull_eff = (1 - signal_strength) * avg_drift + signal_strength * drift_bull
    drift_bear_eff = (1 - signal_strength) * avg_drift + signal_strength * drift_bear

    d_bull = drift_bull_eff / TRADING_DAYS_PER_YEAR
    d_bear = drift_bear_eff / TRADING_DAYS_PER_YEAR
    s_bull = vol_bull_eff / np.sqrt(TRADING_DAYS_PER_YEAR)
    s_bear = vol_bear_eff / np.sqrt(TRADING_DAYS_PER_YEAR)

    MONTH = 21
    regime = np.zeros(n_days, dtype=int)   # 0 = bull, 1 = bear
    state = 0
    for i in range(n_days):
        if i % MONTH == 0 and i > 0:
            if state == 0:
                state = 1 if rng.random() < p_bull_to_bear else 0
            else:
                state = 0 if rng.random() < p_bear_to_bull else 1
        regime[i] = state

    log_risky = np.where(
        regime == 0,
        rng.normal(d_bull, s_bull, n_days),
        rng.normal(d_bear, s_bear, n_days),
    )
    log_safe = rng.normal(
        safe_drift / TRADING_DAYS_PER_YEAR,
        safe_vol / np.sqrt(TRADING_DAYS_PER_YEAR),
        n_days,
    )
    risky = 100.0 * np.exp(np.cumsum(log_risky))
    safe = 100.0 * np.exp(np.cumsum(log_safe))
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash_idx = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(
        {"risky": risky, "safe": safe, "cash": cash_idx},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "drift_bull": drift_bull, "drift_bear": drift_bear,
        "vol_bull": vol_bull, "vol_bear": vol_bear,
        "vol_bull_eff": float(vol_bull_eff), "vol_bear_eff": float(vol_bear_eff),
        "drift_bull_eff": float(drift_bull_eff), "drift_bear_eff": float(drift_bear_eff),
        "n_years": n_years, "n_days": n_days, "seed": seed,
        "cash_rate_ann": cash_rate_ann,
        "bear_frac": float(regime.mean()),
    }
    return prices, truth


def synthetic_panel(
    n_seeds: int = 5,
    signal_strength: float = 1.0,
    base_seed: int = 937,
    **kwargs,
) -> list[tuple[pd.DataFrame, dict]]:
    """A deterministic list of ``n_seeds`` synthetic worlds (for the seed sweep)."""
    return [
        synthetic_daily(signal_strength=signal_strength, seed=base_seed + s, **kwargs)
        for s in range(n_seeds)
    ]

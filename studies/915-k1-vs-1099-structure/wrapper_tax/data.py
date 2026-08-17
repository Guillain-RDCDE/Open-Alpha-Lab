"""Data layer for Study 915 — K-1 vs 1099 (two wrappers around the same commodity beta).

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the two wrappers under test — **DBC**
  (Invesco DB Commodity Index Tracking Fund, a commodity pool that issues a
  **Schedule K-1**) and **PDBC** (Invesco Optimum Yield Diversified Commodity
  Strategy No K-1 ETF, a 1940-Act fund that issues a **Form 1099**) — plus three
  context vehicles (**USCI**, **USO**, **BNO**, all K-1 partnerships) and **BIL**
  (1-3 month T-bills) as the cash leg and as the collateral-interest proxy.
  ``fetch`` touches the network and caches parquet under the shared ``studies/_cache``
  (retry up to 4x); ``load_prices`` reads that cache **offline** and never imports
  yfinance. The whole test-suite runs with NO cache present (synthetic only), so CI
  is green on a fresh checkout.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators. Two
  wrapper tapes are drawn on a **shared commodity factor** plus independent tracking
  noise; the 1099 leg additionally carries a **planted annual drag** scaled by the
  ``signal_strength`` knob. At ``signal_strength=0`` the two wrappers are economically
  identical (the null: the race must find nothing); at ``signal_strength=1`` the drag is
  a real, detectable wrapper cost the race must recover. Seed is fixed → deterministic.

Price-only vs total-return: every real series here is **total return** (``auto_adjust=True``
folds distributions back in). This matters more than usual — PDBC makes large ordinary
income distributions in high-short-rate years, and a price-only comparison would flatter
the K-1 leg for no economic reason.

Nothing in this module models tax. The tax layer lives in ``strategy.py`` and is built
entirely from **stated assumptions** (marginal rates, the distribution-payout share),
labelled as such and swept.
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

# The two wrappers under test, three context vehicles, and the cash leg.
#   DBC  — commodity pool, Schedule K-1, DBIQ Optimum Yield Diversified Commodity index
#   PDBC — 1940-Act ETF via a Cayman subsidiary, Form 1099, same index family but
#          ACTIVELY managed (it does not mechanically replicate DBC's index), marketed
#          explicitly as the "No K-1" twin. The active mandate is a labelled feature of
#          the pair, not something this study measures or dates from the tape.
#   USCI, USO, BNO — K-1 partnerships, the index/vehicle-dispersion context
#   BIL  — 1-3 month T-bill ETF: the cash leg AND the collateral-interest proxy
TICKERS = ("DBC", "PDBC", "USCI", "USO", "BNO", "BIL")

WRAPPER_K1 = "DBC"
WRAPPER_1099 = "PDBC"
CASH = "BIL"

# Study-wide as-of: the last COMPLETE calendar month at build time. The annual
# after-tax model additionally uses only COMPLETE calendar years (2015-2025).
AS_OF = "2026-06-30"

# --------------------------------------------------------------------------- #
# Non-tape inputs — PROXIES / ASSUMPTIONS, never measured here.
# Stated so the reader can see exactly what is not coming off the tape. Both
# total-return series are already net of these fees; they are quoted only to say
# what the *fee* gap predicts against what the tape actually delivered.
# --------------------------------------------------------------------------- #
# NOTE the hindsight: these are the CURRENT prospectus ratios applied to the whole
# 2014-2026 window. Fee schedules move; nothing in this study re-reads them per year.
# They are never subtracted from a return series — the tape is already net of the real,
# time-varying fees — so the hindsight touches only the prediction-vs-tape sentence.
EXPENSE_RATIO_PROXY = {"DBC": 0.0087, "PDBC": 0.0059}  # prospectus gross, ASSUMPTION


# --------------------------------------------------------------------------- #
# Real tape — Yahoo! Finance daily total-return, cache-only by default
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str, cache_dir: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("/", "")
    return os.path.join(cache_dir, f"prices_{safe}_1d.parquet")


def fetch(
    tickers=TICKERS,
    start: str = "2006-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` makes the
    ``close`` column split- and distribution-adjusted total return.
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

    Returns a date-indexed frame with one column per ticker, sliced to ``asof`` so the
    sample never creeps. Raises ``FileNotFoundError`` when a ticker is missing — the
    offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call wrapper_tax.data.fetch() once to populate the shared cache."
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
# Synthetic tape — the deterministic offline core (a planted wrapper drag)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 12,
    factor_vol: float = 0.18,        # annualised vol of the shared commodity factor
    factor_drift: float = 0.03,      # annualised drift of the shared factor
    track_vol: float = 0.03,         # annualised idiosyncratic tracking noise per leg
    drag_ann: float = 0.0400,        # planted annual drag on the 1099 leg at strength 1
    signal_strength: float = 1.0,    # 0 = the null (identical wrappers), 1 = full drag
    cash_rate_ann: float = 0.025,    # the collateral / cash accrual rate
    start: str = "2014-11-07",
    seed: int = 915,
) -> tuple[pd.DataFrame, dict]:
    """Two wrapper tapes on a shared commodity factor, one carrying a planted drag.

    Columns: ``k1`` (the Schedule-K-1 commodity pool), ``ric`` (the Form-1099 fund) and
    ``cash`` (a cash accrual index). Both wrappers share the same factor path, so their
    *difference* is exactly (planted drag) + (independent tracking noise) — the same
    structure the real DBC/PDBC difference has.

    ``signal_strength`` scales the planted drag only:

    - ``1`` → the 1099 leg loses ``drag_ann`` per year; the race must recover it;
    - ``0`` → the two wrappers are economically identical (the null: the race must find
      nothing, and must not manufacture a tax or fee story out of tracking noise).

    Returns ``(prices, truth)``; deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: a few thousand business days stays far inside pandas' ns Timestamp range.
    dates = pd.bdate_range(start=start, periods=n_days)

    mu = factor_drift / TRADING_DAYS_PER_YEAR
    sd = factor_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    tsd = track_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    drag_daily = signal_strength * drag_ann / TRADING_DAYS_PER_YEAR

    factor = rng.normal(mu, sd, n_days)
    e_k1 = rng.normal(0.0, tsd, n_days)
    e_ric = rng.normal(0.0, tsd, n_days)

    r_k1 = factor + e_k1
    r_ric = factor + e_ric - drag_daily

    k1 = 100.0 * np.cumprod(1.0 + r_k1)
    ric = 100.0 * np.cumprod(1.0 + r_ric)
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash = np.cumprod(np.full(n_days, cash_daily))

    prices = pd.DataFrame(
        {"k1": k1, "ric": ric, "cash": cash},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "drag_ann": drag_ann,
        "planted_drag_ann": signal_strength * drag_ann,
        "factor_vol": factor_vol,
        "factor_drift": factor_drift,
        "track_vol": track_vol,
        "expected_te_ann": float(np.sqrt(2.0) * track_vol),
        "cash_rate_ann": cash_rate_ann,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
    }
    return prices, truth


def synthetic_panel(
    n_pairs: int = 8,
    signal_strength: float = 0.0,
    seed: int = 915,
    **kwargs,
) -> list[tuple[pd.DataFrame, dict]]:
    """A panel of ``n_pairs`` independent wrapper pairs (seeds ``seed``, ``seed+1``, ...).

    Used for the null sweep: at ``signal_strength=0`` the estimated wrapper drag across
    the panel must be centred on zero and must not clear |t| = 2 more often than chance.
    """
    return [
        synthetic_daily(signal_strength=signal_strength, seed=seed + i, **kwargs)
        for i in range(n_pairs)
    ]

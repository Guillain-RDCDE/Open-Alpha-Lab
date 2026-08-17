"""Data layer for Study 940 — The Turnover Budget (rebalance frequency vs cost).

Two tapes, one shape (a date-indexed daily total-return close panel):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the eleven Select Sector SPDRs (the GICS
  sector sleeve a rotation model actually trades), plus **BIL** (1-3 month T-bills,
  the cash leg) and **SPY** (the cap-weight benchmark). ``fetch`` touches the network
  and caches parquet under the **shared** ``studies/_cache`` (retry up to 4x);
  ``load_prices`` reads that cache **offline** and never imports yfinance. The whole
  test-suite runs with NO cache present (synthetic only), so CI is green on a fresh
  checkout where ``_cache/`` is git-ignored and absent.

- ``synthetic_panel`` — a *deterministic, offline* generator. A cross-sectional panel
  of ``n_assets`` correlated daily tapes whose expected returns carry a slowly-decaying
  persistent component, so a 12-1 momentum rank has something real to read. The
  ``signal_strength`` knob scales that persistent component: at ``signal_strength=0``
  the panel is pure noise around a common market factor (the null — no frequency can
  win); at ``signal_strength=1`` the planted momentum is strong enough that the
  *daily* rebalance should capture the most gross return and the *quarterly* the least.
  ``synthetic_daily`` is the single-name convenience wrapper. Seeds are fixed.

Sector coverage is **staggered by construction**: XLRE was carved out in 2015 and XLC
in 2018, so the tradable breadth grows from 9 to 11 names. The panel builder therefore
returns an availability mask rather than pretending eleven sectors existed in 1999 —
see ``load_prices`` and ``turnover_budget.strategy.momentum_ranks``.

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (weights
formed from data through day ``t`` are earned from day ``t+1``).
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The SHARED desk cache — most of these tapes are already sitting there.
DEFAULT_CACHE = os.path.abspath(os.path.join(HERE, "..", "..", "_cache"))

TRADING_DAYS_PER_YEAR = 252

# The eleven Select Sector SPDRs = the GICS sectors, the standard rotation sleeve.
SECTORS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")
CASH = "BIL"
BENCH = "SPY"
TICKERS = SECTORS + (CASH, BENCH)

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
    start: str = "1998-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. Uses ``auto_adjust=True`` so
    the ``close`` column is split- and dividend-adjusted **total return** — mandatory
    here, because the sector SPDRs yield anywhere from 0.6% (XLK) to 3% (XLU) and a
    price-only panel would silently tilt the cross-section toward the low-yield names.
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
    """Read cached daily total-return closes OFFLINE into one aligned close panel.

    Returns a frame indexed by date with one column per ticker, sliced to ``asof`` so
    the sample never creeps. Columns that start late (XLRE 2015, XLC 2018) are simply
    NaN before their inception — the strategy layer treats NaN as "not investable yet"
    rather than back-filling a sector that did not exist. Raises ``FileNotFoundError``
    if any ticker is missing: the offline core and the test-suite never touch the
    network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call turnover_budget.data.fetch() once to populate the shared cache."
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
# Synthetic tape — the deterministic offline core (a planted momentum panel)
# --------------------------------------------------------------------------- #
def synthetic_panel(
    n_assets: int = 11,
    n_years: int = 20,
    market_vol: float = 0.15,          # annualised common-factor vol
    idio_vol: float = 0.14,            # annualised idiosyncratic vol
    alpha_vol: float = 0.06,           # annualised sd of the persistent expected-return leg
    alpha_halflife_days: float = 63.0, # persistence of that leg (~a quarter)
    signal_strength: float = 1.0,      # 0 = pure-noise null, 1 = fully planted momentum
    start: str = "2004-01-02",
    seed: int = 940,
    cash_rate_ann: float = 0.02,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """A daily cross-sectional panel with a *planted, slowly-decaying* momentum leg.

    Each asset's daily return is ``beta * market + a_i(t) + idio``, where ``a_i(t)`` is
    an AR(1) expected-return component with half-life ``alpha_halflife_days``. Because
    ``a_i`` persists for roughly a quarter, a trailing 12-1 momentum rank is a noisy but
    genuine estimate of it — and a *faster* rebalance tracks it better, which is exactly
    the trade-off this study prices.

    - ``signal_strength = 1`` → full planted momentum; gross (pre-cost) return should
      *fall* monotonically as the rebalance gets slower.
    - ``signal_strength = 0`` → ``a_i`` is switched off; the panel is a market factor
      plus pure idiosyncratic noise, and no frequency should earn anything (the null).

    Returns ``(prices, cash, truth)``: ``prices`` a date-by-asset total-return close
    panel (columns ``A0..A{n-1}``), ``cash`` a cash accrual index on the same index,
    and ``truth`` the planted parameters. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    mkt_d = market_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    idio_d = idio_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    alpha_d = alpha_vol / np.sqrt(TRADING_DAYS_PER_YEAR) * float(signal_strength)

    phi = float(np.exp(-np.log(2.0) / alpha_halflife_days))
    shock_sd = alpha_d * np.sqrt(1.0 - phi ** 2)

    market = rng.normal(0.06 / TRADING_DAYS_PER_YEAR, mkt_d, n_days)
    betas = rng.uniform(0.8, 1.2, n_assets)

    alpha = np.zeros((n_days, n_assets))
    a = rng.normal(0.0, alpha_d, n_assets) if alpha_d > 0 else np.zeros(n_assets)
    shocks = rng.normal(0.0, 1.0, (n_days, n_assets))
    for t in range(n_days):
        a = phi * a + shock_sd * shocks[t]
        alpha[t] = a

    idio = rng.normal(0.0, idio_d, (n_days, n_assets))
    log_ret = market[:, None] * betas[None, :] + alpha + idio
    px = 100.0 * np.exp(np.cumsum(log_ret, axis=0))

    prices = pd.DataFrame(
        px,
        index=pd.DatetimeIndex(dates, name="date"),
        columns=[f"A{i}" for i in range(n_assets)],
    )
    cash_daily = (1.0 + cash_rate_ann) ** (1.0 / TRADING_DAYS_PER_YEAR)
    cash = pd.Series(
        np.cumprod(np.full(n_days, cash_daily)),
        index=prices.index, name="cash",
    )
    truth = {
        "n_assets": n_assets,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "signal_strength": float(signal_strength),
        "alpha_vol_eff": float(alpha_d * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "alpha_halflife_days": alpha_halflife_days,
        "market_vol": market_vol,
        "idio_vol": idio_vol,
        "cash_rate_ann": cash_rate_ann,
        "phi": phi,
    }
    return prices, cash, truth


def synthetic_daily(
    n_years: int = 20,
    signal_strength: float = 1.0,
    seed: int = 940,
) -> tuple[pd.DataFrame, dict]:
    """Single-name convenience wrapper: the first asset of the panel plus the cash leg.

    Returns ``(frame, truth)`` where ``frame`` has columns ``asset`` and ``cash`` — the
    shape the desk's single-asset helpers expect. Deterministic given ``seed``.
    """
    prices, cash, truth = synthetic_panel(
        n_years=n_years, signal_strength=signal_strength, seed=seed
    )
    frame = pd.DataFrame({"asset": prices.iloc[:, 0], "cash": cash})
    return frame, truth

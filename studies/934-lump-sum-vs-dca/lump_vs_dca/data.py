"""Data layer for Study 934 — Lump Sum vs DCA (all-in on day 0 vs twelve monthly tranches).

Two tapes, one shape (a date-indexed daily total-return close frame):

- ``fetch`` / ``load_prices`` — daily **total-return** closes from Yahoo! Finance
  (``yfinance``, ``auto_adjust=True``) for the risk asset (SPY), the bond-heavy
  variant (IEF, 7-10y Treasuries) and the cash leg (BIL, the 1-3 month T-bill ETF —
  what the *uninvested* DCA balance actually earns while it waits). ``fetch`` touches
  the network and writes parquet into the **shared** ``studies/_cache`` (retry up to
  4x); ``load_prices`` reads that cache **offline** and never imports yfinance. The
  whole test-suite runs with NO cache present (synthetic-only), so CI is green on a
  fresh checkout.

- ``synthetic_daily`` / ``synthetic_panel`` — *deterministic, offline* generators. A
  geometric random walk plus a cash accrual leg, with a ``signal_strength`` knob that
  scales the asset's drift **in excess of cash**. At ``signal_strength=1`` there is a
  fat planted risk premium, so the lump sum must win on average and the mean
  terminal-wealth gap must be positive; at ``signal_strength=0`` the asset earns
  exactly the cash rate in expectation, so the gap must sit on zero and the win rate
  on one half (the null). A **negative** ``signal_strength`` flips the world and DCA
  must win — the two-sided control. Seeds are fixed, so tests are deterministic.

The question this study asks: the folk advice says drip a windfall in over a year
because you "buy at a lower average price" and "lower your risk". Over every start
month, who actually finishes richer, by how much, and does the answer flip when you
start from a stretched market or from the bottom of a drawdown?

No look-ahead is baked in here — that discipline lives in ``strategy.py`` (the
decision is taken at a month-end close and executed at the **next** trading day's
close).
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

# SPY = the risk asset the advice is usually given about; IEF = the bond-heavy
# variant (a windfall going into a conservative sleeve); BIL = the cash leg the
# uninvested DCA balance sits in.
TICKERS = ("SPY", "IEF", "BIL")

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
    start: str = "1993-01-01",
    end: str | None = None,
    cache_dir: str = DEFAULT_CACHE,
    retries: int = 4,
) -> dict[str, pd.DataFrame]:
    """Download daily total-return closes for ``tickers`` and cache each as parquet.

    Network-only; run once to populate the shared cache. ``auto_adjust=True`` so the
    ``close`` column is split- and dividend-adjusted **total return** — essential
    here, because the whole contest is between compounding a dividend-paying equity
    tape and accruing the T-bill yield on the cash that has not been invested yet.
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
    sliced to ``asof`` so the sample never creeps. Raises ``FileNotFoundError`` if a
    ticker is missing — the offline core and the test-suite never touch the network.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk, cache_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cached prices for {tk} at {path}. "
                f"Call lump_vs_dca.data.fetch() once to populate the shared cache."
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
# Synthetic tape — the deterministic offline core (a planted risk premium)
# --------------------------------------------------------------------------- #
def synthetic_daily(
    n_years: int = 25,
    excess_drift: float = 0.07,     # annualised asset drift IN EXCESS OF CASH at ss=1
    vol: float = 0.16,              # annualised asset vol
    cash_rate_ann: float = 0.03,    # the cash leg the uninvested tranches earn
    signal_strength: float = 1.0,   # 1 = full planted premium, 0 = null, <0 = DCA world
    start: str = "1999-01-04",
    seed: int = 934,
) -> tuple[pd.DataFrame, dict]:
    """A daily (asset, cash) tape with a planted excess drift of known sign.

    The ``signal_strength`` knob scales the asset's drift *in excess of the cash leg*:

    - ``signal_strength = 1`` → a fat planted risk premium; every dollar waiting in
      cash forgoes it, so the **lump sum must win** (positive mean terminal gap,
      win rate well above 50%).
    - ``signal_strength = 0`` → the asset earns exactly the cash rate in expectation
      (the null); the gap must sit on **zero** and the win rate on **one half**.
    - ``signal_strength < 0`` → a falling market; **DCA must win** — the other side of
      the two-sided control, which is what stops the harness from being a
      lump-sum-shaped rubber stamp.

    Returns ``(prices, truth)`` where ``prices`` has columns ``asset`` and ``cash``
    (both total-return index levels). Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    # OOB-safe: bdate_range with n <= 10000 daily bars stays well inside pandas' ns range.
    dates = pd.bdate_range(start=start, periods=n_days)

    mu_ex = signal_strength * excess_drift
    cash_daily_log = np.log1p(cash_rate_ann) / TRADING_DAYS_PER_YEAR
    s = vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    # Log-drift set so the ARITHMETIC excess drift is mu_ex (Ito correction included).
    d_asset = cash_daily_log + mu_ex / TRADING_DAYS_PER_YEAR - 0.5 * s * s

    log_ret = rng.normal(d_asset, s, n_days)
    asset = 100.0 * np.exp(np.cumsum(log_ret))
    cash = 100.0 * np.exp(np.cumsum(np.full(n_days, cash_daily_log)))

    prices = pd.DataFrame(
        {"asset": asset, "cash": cash},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "excess_drift": excess_drift,
        "excess_drift_eff": float(mu_ex),
        "vol": vol,
        "cash_rate_ann": cash_rate_ann,
        "n_years": n_years,
        "n_days": n_days,
        "seed": seed,
        "realised_excess_ann": float(
            (np.mean(log_ret) - cash_daily_log) * TRADING_DAYS_PER_YEAR
        ),
    }
    return prices, truth


def synthetic_panel(
    n_years: int = 25,
    signal_strength: float = 1.0,
    equity_excess: float = 0.07,
    bond_excess: float = 0.015,
    equity_vol: float = 0.16,
    bond_vol: float = 0.06,
    corr: float = -0.20,
    cash_rate_ann: float = 0.03,
    start: str = "1999-01-04",
    seed: int = 934,
) -> tuple[pd.DataFrame, dict]:
    """A two-asset (equity, bond) + cash tape for the bond-heavy variant.

    Same ``signal_strength`` semantics as :func:`synthetic_daily`, applied to both
    excess drifts. The bond leg carries a *smaller* planted premium and a lower vol,
    which is the whole point of the bond-heavy cut: a thinner premium means a thinner
    cash-drag penalty, so the lump sum's edge should shrink — but not flip sign.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start, periods=n_days)

    cash_daily_log = np.log1p(cash_rate_ann) / TRADING_DAYS_PER_YEAR
    se = equity_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    sb = bond_vol / np.sqrt(TRADING_DAYS_PER_YEAR)
    cov = np.array([[se * se, corr * se * sb], [corr * se * sb, sb * sb]])
    chol = np.linalg.cholesky(cov)

    shocks = rng.normal(0.0, 1.0, size=(n_days, 2)) @ chol.T
    d_eq = cash_daily_log + signal_strength * equity_excess / TRADING_DAYS_PER_YEAR - 0.5 * se * se
    d_bd = cash_daily_log + signal_strength * bond_excess / TRADING_DAYS_PER_YEAR - 0.5 * sb * sb
    eq = 100.0 * np.exp(np.cumsum(d_eq + shocks[:, 0]))
    bd = 100.0 * np.exp(np.cumsum(d_bd + shocks[:, 1]))
    cash = 100.0 * np.exp(np.cumsum(np.full(n_days, cash_daily_log)))

    prices = pd.DataFrame(
        {"equity": eq, "bond": bd, "cash": cash},
        index=pd.DatetimeIndex(dates, name="date"),
    )
    truth = {
        "signal_strength": signal_strength,
        "equity_excess": equity_excess,
        "bond_excess": bond_excess,
        "equity_vol": equity_vol,
        "bond_vol": bond_vol,
        "corr": corr,
        "cash_rate_ann": cash_rate_ann,
        "n_days": n_days,
        "seed": seed,
    }
    return prices, truth

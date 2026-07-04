"""Data layer for Study 591 — Vol-Managed Portfolio (Moreira & Muir, JF 2017).

Two sources, both offline-friendly once cached:

* **Real tape.** Daily **total-return** (auto-adjusted) closes for SPY (headline) plus
  QQQ / EFA / IWM (robustness legs), and the 13-week T-bill discount yield ``^IRX`` as the
  risk-free rate — all from yfinance (public, no key). Prices cached wide
  (``vmp_prices.csv``); the rate cached as a single column (``vmp_rf.csv``). Every race is
  **excess-vs-excess**: both the managed and the unmanaged leg subtract the same
  T-bill rate, so the rf convention (we use ``IRX/100/252`` per trading day, a
  bank-discount-to-daily shortcut worth < 2 bps/yr of error) cancels in the comparison.

* **Synthetic control.** A deterministic, seeded daily-return world with **persistent
  volatility clustering** (log-vol AR(1), GARCH-like) and a **tunable vol-return
  disconnect** knob ``disconnect``:

  - ``disconnect = 0`` (the NULL): the conditional mean scales one-for-one with the
    conditional variance (risk is compensated proportionally, ``mu_t = lam * sig_t^2``).
    In this world 1/RV scaling reshuffles risk but earns **no alpha** — the overlay must
    NOT win, however the noise falls.
  - ``disconnect = 1`` (flat-mean world): the conditional mean is **flat** while
    variance swings (vol spikes carry no extra compensation).
  - ``disconnect = 2`` (the PLANTED leverage-effect world): the conditional mean
    actually FALLS as variance rises — the empirical vol-return disconnect
    Moreira-Muir document. Here 1/RV scaling MUST produce significant alpha; a
    harness that can't bank it proves nothing.

  Both worlds share the same unconditional mean and the same vol path per seed. No
  timestamps are used (plain month-id arrays), so no pandas ns-Timestamp span traps.

Pure numpy + pandas + stdlib for the offline path. ``fetch_all`` (network) is used only
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "_cache")
PRICES_CACHE = os.path.join(CACHE_DIR, "vmp_prices.csv")
RF_CACHE = os.path.join(CACHE_DIR, "vmp_rf.csv")

TICKERS = ["SPY", "QQQ", "EFA", "IWM"]
RF_TICKER = "^IRX"          # 13-week T-bill discount yield, in percent
TRADING_DAYS = 252

# Study-wide as-of: the last COMPLETE calendar month at build time (2026-07-03).
AS_OF = "2026-06-30"


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_all(start: str = "1993-01-01", end: str | None = None,
              prices_path: str = PRICES_CACHE, rf_path: str = RF_CACHE,
              retries: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """Download ETF total-return closes + ^IRX and cache them. Network-only; run once."""
    import yfinance as yf

    px = None
    for _ in range(retries):
        try:
            px = yf.download(TICKERS, start=start, end=end, auto_adjust=True,
                             progress=False)["Close"]
            if px is not None and len(px) > 0:
                break
        except Exception:
            time.sleep(2.0)
    px = px.dropna(how="all")
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)

    irx = None
    for _ in range(retries):
        try:
            irx = yf.download(RF_TICKER, start=start, end=end, auto_adjust=False,
                              progress=False)["Close"]
            if irx is not None and len(irx) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if isinstance(irx, pd.DataFrame):
        irx = irx.iloc[:, 0]
    irx = irx.dropna()
    irx.index = pd.DatetimeIndex(irx.index).tz_localize(None)
    irx.name = "irx_pct"

    os.makedirs(os.path.dirname(prices_path), exist_ok=True)
    px.to_csv(prices_path)
    irx.to_frame().to_csv(rf_path)
    return px, irx


def have_real(prices_path: str = PRICES_CACHE, rf_path: str = RF_CACHE) -> bool:
    return os.path.exists(prices_path) and os.path.exists(rf_path)


def load_prices(path: str = PRICES_CACHE, asof: str = AS_OF) -> pd.DataFrame:
    """Wide total-return close frame (index = date, columns = tickers), sliced to as-of."""
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return px[px.index <= pd.Timestamp(asof)]


def load_rf(path: str = RF_CACHE, asof: str = AS_OF) -> pd.Series:
    """^IRX percent yield series, sliced to as-of."""
    rf = pd.read_csv(path, index_col=0, parse_dates=True).sort_index().iloc[:, 0]
    return rf[rf.index <= pd.Timestamp(asof)]


def daily_frames(ticker: str = "SPY", asof: str = AS_OF) -> pd.DataFrame:
    """Daily frame for one ticker: raw return, rf/day, excess return, month period.

    ``rf_d = IRX/100/252`` aligned forward-fill (the T-bill yield is known a day ahead —
    no look-ahead). Returns a frame indexed by date with columns
    ``ret`` (total daily return), ``rf`` (daily risk-free), ``exc`` (ret - rf),
    ``month`` (calendar Period[M]).
    """
    px = load_prices(asof=asof)[ticker].dropna()
    ret = px.pct_change().dropna()
    irx = load_rf(asof=asof)
    rf_d = (irx / 100.0 / TRADING_DAYS).reindex(ret.index).ffill().bfill()
    out = pd.DataFrame({"ret": ret, "rf": rf_d, "exc": ret - rf_d})
    out["month"] = out.index.to_period("M")
    return out


def complete_months(df: pd.DataFrame, min_days: int = 15) -> pd.DataFrame:
    """Drop partial calendar months (first listing month, or a tape ending mid-month).

    A month is kept if it has >= ``min_days`` observations AND its last observed day is
    within 5 calendar days of that month's true end (i.e. the month really finished).
    """
    keep = []
    for m, g in df.groupby("month"):
        if len(g) < min_days:
            continue
        if (m.end_time.normalize() - g.index.max()).days > 5:
            continue
        keep.append(m)
    return df[df["month"].isin(keep)]


# --------------------------------------------------------------------------- #
# Synthetic control — vol-clustered world with a tunable vol-return disconnect
# --------------------------------------------------------------------------- #
def synthetic_world(n_months: int = 360, days_per_month: int = 21,
                    disconnect: float = 0.0, seed: int = 591,
                    sig_bar_daily: float = 0.010, phi: float = 0.98,
                    vol_of_logvol: float = 0.12,
                    mean_ann: float = 0.07) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic daily-return world with persistent vol clustering.

    Volatility: log-vol AR(1), ``h_t = phi h_{t-1} + vol_of_logvol * u_t``,
    ``sig_t = sig_bar_daily * exp(h_t)`` — realistic multi-month vol regimes.

    Mean: ``mu_t = (1 - disconnect) * lam * sig_t^2 + disconnect * mu_flat`` where
    ``lam`` and ``mu_flat`` are set (per path) so the unconditional daily mean equals
    ``mean_ann / 252`` in EVERY world. ``disconnect=0`` = fully priced risk (the null:
    1/RV scaling must NOT earn alpha); ``disconnect=1`` = flat mean while vol swings;
    ``disconnect=2`` = the planted LEVERAGE-EFFECT world (mean falls as variance
    rises: scaling MUST earn alpha).

    Returns ``(daily_returns, month_id, sig_daily)`` as plain numpy arrays — no
    timestamps (immune to the pandas ns-Timestamp 250-year span trap).
    """
    rng = np.random.default_rng(seed)
    n = n_months * days_per_month
    h = np.empty(n)
    u = rng.standard_normal(n)
    h[0] = vol_of_logvol / np.sqrt(1 - phi**2) * u[0]
    for t in range(1, n):
        h[t] = phi * h[t - 1] + vol_of_logvol * u[t]
    sig = sig_bar_daily * np.exp(h)

    mu_daily = mean_ann / TRADING_DAYS
    lam = mu_daily / float(np.mean(sig**2))          # priced world: mu_t = lam sig_t^2
    mu_t = (1.0 - disconnect) * lam * sig**2 + disconnect * mu_daily

    z = rng.standard_normal(n)
    r = mu_t + sig * z
    month_id = np.repeat(np.arange(n_months), days_per_month)
    return r, month_id, sig

"""Data layer for Study 894 — Trend Overlay on 60/40.

Two tapes, one shape (a date-indexed daily total-return **Close** frame with one column
per ticker):

* **Real tape — four liquid ETFs.** Daily total-return closes for ``SPY`` (US equity),
  ``IEF`` (7-10y Treasuries), ``AGG`` (aggregate bond, a robustness alternative to IEF)
  and ``BIL`` (1-3m T-bills, the cash leg), pulled with yfinance (``auto_adjust=True`` so
  the closes are split- and dividend-adjusted **total-return** prices). Cached as parquet
  under this study's OWN ``_cache/``. ``fetch()`` (network) runs once to build the cache;
  ``load_prices()`` reads it OFFLINE. The common window is set by the youngest ticker —
  **BIL launched 2007-05-25** — so the balanced book here spans ~2007-2026 (it does
  capture the 2008 crash once the 200-day window warms up in early 2008). A short-history
  caveat travels with every number on the **Signal** axis.

* **Synthetic world — the positive control.** A deterministic, seeded three-column price
  panel (``synthetic_prices``) with a TUNABLE knob ``edge``: the equity leg carries a
  two-state bull/bear Markov regime whose bear severity scales with ``edge``. At
  ``edge = 0`` the regimes collapse to a single drift/vol (the null: the 200-day filter
  has nothing to duck, so the overlay only *pays* costs and must not beat the static
  book); at ``edge > 0`` the bear regimes are deep enough for the filter to step out and
  cut drawdown — the mechanical reason a trend overlay *should* help. This only proves the
  machinery is unbiased; it never supports the real-tape stamp.

The offline path is pure numpy + pandas + stdlib. The notebooks never import yfinance.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))

TICKERS = ["SPY", "IEF", "AGG", "BIL"]
START = "2007-01-01"          # BIL (the cash leg) launches 2007-05-25 — it sets the window
AS_OF = "2026-06-30"          # last complete calendar month at publication
TRADING_DAYS = 252

__all__ = [
    "TICKERS", "START", "AS_OF", "CACHE_DIR", "TRADING_DAYS",
    "fetch", "have_real", "load_prices", "synthetic_prices",
]


# --------------------------------------------------------------------------- #
# Real tape — yfinance daily total-return closes, cache-only offline
# --------------------------------------------------------------------------- #
def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"px_{ticker}_1d.parquet")


def fetch(tickers=TICKERS, start: str = START, retries: int = 4) -> None:
    """Download each ticker's daily total-return close; cache one parquet per ticker.

    ``auto_adjust=True`` gives split/dividend-adjusted **total-return** closes — essential
    for a multi-year balanced-book study (unadjusted prices would understate the bond and
    cash legs' coupon return). Retries up to ``retries`` times per ticker. Network only.
    """
    import time

    import yfinance as yf

    os.makedirs(CACHE_DIR, exist_ok=True)
    for tk in tickers:
        path = _cache_path(tk)
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                raw = yf.download(
                    tk, start=start, interval="1d",
                    auto_adjust=True, progress=False,
                )
                if raw is None or raw.empty:
                    raise RuntimeError(f"yfinance returned no data for {tk}")
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                df = raw[["Close"]].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "date"
                df = df.dropna()
                df.to_parquet(path)
                break
            except Exception as exc:  # noqa: BLE001 — retry transient network errors
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"failed to fetch {tk} after {retries} tries: {last_err}")


def have_real(tickers=TICKERS) -> bool:
    """True iff every ticker's cache parquet is present (offline-safe check)."""
    return all(os.path.exists(_cache_path(tk)) for tk in tickers)


def load_prices(tickers=TICKERS, start: str = START, asof: str = AS_OF) -> pd.DataFrame:
    """Cached daily total-return **Close** frame (columns = tickers), OFFLINE.

    Reads the per-ticker parquet directly (no yfinance import), aligns on the common
    dates where **all** requested tickers trade, and slices to ``[start, asof]`` (the
    partial current month is dropped by ``asof``). Rows with any missing leg are removed
    so every downstream return is defined on the same calendar.
    """
    cols = {}
    for tk in tickers:
        path = _cache_path(tk)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No cache for {tk} at {path}. Run trend6040.data.fetch() once."
            )
        s = pd.read_parquet(path)["Close"]
        s.index = pd.to_datetime(s.index)
        cols[tk] = s
    px = pd.DataFrame(cols).sort_index()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px.index.name = "date"
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    px = px[(px.index >= lo) & (px.index <= hi)]
    return px.dropna(how="any")


# --------------------------------------------------------------------------- #
# Synthetic world — planted regime the 200-day filter can duck (positive control)
# --------------------------------------------------------------------------- #
def synthetic_prices(
    edge: float = 1.0,
    seed: int = 894,
    n_days: int = 5000,
    start: str = "2005-01-03",
    drift_bull: float = 0.14,
    vol_bull: float = 0.12,
    drift_bear: float = -0.50,
    vol_bear: float = 0.40,
    p_bull_to_bear: float = 0.030,
    p_bear_to_bull: float = 0.09,
    bond_drift: float = 0.035,
    bond_vol: float = 0.06,
    cash_rate: float = 0.03,
    equity_bond_rho: float = -0.15,
) -> pd.DataFrame:
    """Deterministic seeded three-column price panel with a TUNABLE planted trend edge.

    Columns are ``SPY`` (equity), ``IEF`` (bond) and ``BIL`` (cash accrual index). The
    equity leg follows a two-state bull/bear Markov regime; ``edge`` scales how deep and
    how volatile the **bear** regime is relative to a neutral blend:

        drift_bear_eff = blend + edge * (drift_bear - blend)      (blend = stationary mean)
        vol_bear_eff   = avg_vol + edge * (vol_bear - avg_vol)

    * ``edge = 0`` → both regimes share the stationary drift/vol: the 200-day filter has
      nothing to duck, so a trend overlay can only *pay* switching costs and must **not**
      beat the static 60/40 (the null).
    * ``edge = 1`` → full separation: bear markets are deep and choppy, the filter steps
      to cash and cuts drawdown — the mechanical reason a trend overlay should help.

    Business-day index kept far below the pandas ns-timestamp horizon. No network.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days, name="date")

    # --- blend the bear regime toward the null as edge -> 0 ------------------
    stat_bear = p_bull_to_bear / (p_bull_to_bear + p_bear_to_bull)
    blend_drift = (1.0 - stat_bear) * drift_bull + stat_bear * drift_bear
    avg_vol = (1.0 - stat_bear) * vol_bull + stat_bear * vol_bear
    drift_bear_eff = blend_drift + edge * (drift_bear - blend_drift)
    vol_bear_eff = avg_vol + edge * (vol_bear - avg_vol)
    drift_bull_eff = blend_drift + edge * (drift_bull - blend_drift)
    vol_bull_eff = avg_vol + edge * (vol_bull - avg_vol)

    d_bull, d_bear = drift_bull_eff / TRADING_DAYS, drift_bear_eff / TRADING_DAYS
    s_bull, s_bear = vol_bull_eff / np.sqrt(TRADING_DAYS), vol_bear_eff / np.sqrt(TRADING_DAYS)

    # Monthly regime transitions (~21 trading days).
    MONTH = 21
    regime = np.zeros(n_days, dtype=int)
    state = 0
    for i in range(n_days):
        if i % MONTH == 0 and i > 0:
            if state == 0:
                state = 1 if rng.random() < p_bull_to_bear else 0
            else:
                state = 0 if rng.random() < p_bear_to_bull else 1
        regime[i] = state

    z_eq = rng.standard_normal(n_days)
    z_bd_idio = rng.standard_normal(n_days)
    # a mild equity/bond correlation (bonds a partial hedge)
    z_bd = equity_bond_rho * z_eq + np.sqrt(max(1.0 - equity_bond_rho**2, 0.0)) * z_bd_idio

    r_eq = np.where(regime == 0, d_bull + s_bull * z_eq, d_bear + s_bear * z_eq)
    r_bd = bond_drift / TRADING_DAYS + (bond_vol / np.sqrt(TRADING_DAYS)) * z_bd

    spy = 100.0 * np.cumprod(1.0 + r_eq)
    ief = 100.0 * np.cumprod(1.0 + r_bd)
    bil_daily = (1.0 + cash_rate) ** (1.0 / TRADING_DAYS)
    bil = 100.0 * np.cumprod(np.full(n_days, bil_daily))

    return pd.DataFrame({"SPY": spy, "IEF": ief, "BIL": bil}, index=idx)

"""Data layer for Study 896 — Risk-Parity + Trend.

The claim under test: an inverse-vol **risk-parity** book across four cross-asset
sleeves (equities / long Treasuries / gold / commodities) is a genuinely diversified
portfolio, but it still rides every sleeve straight through its own bear markets (long
bonds −40% in 2022, commodities −60% into 2009). Bolt a **200-day trend gate** onto each
sleeve — hold the sleeve only while it is above its 200-day moving average, otherwise
that sleeve's risk budget sits in **cash** (T-bills) — and you should de-risk in
sustained downtrends without touching the diversification the rest of the time. The
question: does adding trend to risk-parity actually improve the **excess-of-cash Sharpe
and the drawdown**, net of costs, versus plain risk-parity?

Two sources, both offline-friendly once cached.

* **Real tape — six liquid ETFs.** Daily total-return closes (yfinance,
  ``auto_adjust=True`` — dividends/coupons reinvested) for the four risk-parity sleeves
  **SPY / TLT / GLD / DBC**, an intermediate-Treasury alternate **IEF**, and the cash
  leg **BIL** (1-3 month T-bill ETF — the risk-free the trend gate parks in). The common
  sample starts at **BIL's 2007 inception**, so every headline number is conditioned on
  a tape that already contains the 2008 crash, 2020 COVID and the 2022 bond bear. Cached
  as a single wide parquet of adjusted closes under this study's OWN ``_cache/``.

* **Synthetic world — the positive control.** A deterministic, seeded multi-asset world
  (``synthetic_world``) whose assets follow a persistent two-state **bull/bear regime**
  (Markov). A tunable knob ``edge`` sets how punishing the bear regime is:

  - ``edge = 0`` (the NULL): bear and bull regimes are identical — returns carry no
    persistent down-trends, so a 200-day trend gate can only add turnover and noise. The
    overlay must **not** improve risk-adjusted return.
  - ``edge > 0`` (the PLANTED trend-premium world): the bear regime drags drift negative
    and lifts vol, so sleeves spend months grinding down. A 200-day gate that steps out
    of those regimes **must** raise the Sharpe and cut the drawdown; a harness that can't
    bank that proves nothing.

  Business-day index, span far below the pandas ns-Timestamp horizon (OOB-safe).

Pure numpy + pandas + stdlib for the offline path. ``fetch()`` (network) runs once to
build the cache and is never imported by the notebooks' offline cells; ``load_prices()``
reads the cached parquet directly (no yfinance import).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.abspath(os.path.join(HERE, "..", "_cache"))
PRICES_CACHE = os.path.join(CACHE_DIR, "rp_trend_prices.parquet")

# The four risk-parity sleeves, plus an alternate bond and the cash leg.
SLEEVES = ["SPY", "TLT", "GLD", "DBC"]        # stocks, long Treasuries, gold, commodities
CASH = "BIL"                                  # 1-3m T-bill ETF — the risk-free / gate cash
TICKERS = ["SPY", "TLT", "GLD", "DBC", "IEF", "BIL"]

START = "2007-01-01"          # BIL lists 2007-05; the common sample begins there
AS_OF = "2026-06-30"          # last complete calendar month at build time
TRADING_DAYS = 252

__all__ = [
    "SLEEVES", "CASH", "TICKERS", "START", "AS_OF", "CACHE_DIR", "PRICES_CACHE",
    "TRADING_DAYS", "fetch", "have_real", "load_prices", "synthetic_world",
]


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch(start: str = START, path: str = PRICES_CACHE, retries: int = 4) -> pd.DataFrame:
    """Download daily total-return closes for all tickers and cache them as one wide
    parquet. Network-only; run once to build the cache."""
    import yfinance as yf

    px = None
    for _ in range(retries):
        try:
            raw = yf.download(TICKERS, start=start, auto_adjust=True, progress=False)
            px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            if px is not None and len(px) > 0:
                break
        except Exception:
            time.sleep(2.0)
    if px is None or len(px) == 0:
        raise RuntimeError("yfinance returned no data for the RP+Trend panel")
    px = px[[t for t in TICKERS if t in px.columns]].copy()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px.index.name = "date"
    os.makedirs(CACHE_DIR, exist_ok=True)
    px.to_parquet(path)
    return px


def have_real(path: str = PRICES_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = PRICES_CACHE, asof: str = AS_OF,
                start: str = START) -> pd.DataFrame:
    """Cached wide frame of daily total-return closes (columns = tickers), sliced to
    ``[start, asof]``. Reads the parquet directly — OFFLINE, no yfinance import.

    Rows are kept only where **all** of the four sleeves + cash are present, so the
    risk-parity book and its cash leg share one common sample (BIL's 2007 inception)."""
    px = pd.read_parquet(path).sort_index()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    lo, hi = pd.Timestamp(start), pd.Timestamp(asof)
    px = px[(px.index >= lo) & (px.index <= hi)]
    need = SLEEVES + [CASH]
    px = px[[c for c in px.columns if c in TICKERS]]
    return px.dropna(subset=[c for c in need if c in px.columns])


# --------------------------------------------------------------------------- #
# Synthetic world — bull/bear regime assets + a flat cash leg (the positive control)
# --------------------------------------------------------------------------- #
def synthetic_world(
    edge: float = 0.0,
    seed: int = 896,
    n_days: int = 5000,
    n_assets: int = 4,
    start: str = "2007-01-02",
    base_vol: float = 0.010,
    vol_spread: float = 0.004,
    bull_drift_ann: float = 0.10,
    p_stay_bull: float = 0.990,
    p_stay_bear: float = 0.985,
    bear_drift_ann: float = 0.45,
    bear_vol_bump: float = 1.8,
    cash_ann: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Deterministic seeded multi-asset world with persistent bull/bear regimes.

    Each asset carries its OWN two-state Markov regime (bull sticky at
    ``p_stay_bull``, bear sticky at ``p_stay_bear``) and a distinct volatility
    (``base_vol + k*vol_spread``) so inverse-vol weights are non-trivial. In the bull
    state the asset drifts up at ``bull_drift_ann``; in the bear state the drift is
    dragged down by ``edge * bear_drift_ann`` and the vol is lifted by
    ``edge * (bear_vol_bump - 1)``:

        r[k,t] = mu_bull            + sigma_k * z            (bull)
        r[k,t] = mu_bull - edge*mu_bear + sigma_k*(1+edge*(bump-1)) * z   (bear)

    ``edge = 0`` collapses both states onto the same distribution — there is NO
    persistent downtrend for a moving-average filter to catch, so the trend gate must
    add nothing (the null). ``edge > 0`` plants sustained bear grinds that a 200-day
    gate should step out of.

    Returns ``(prices, returns, cash_returns)`` — ``prices`` for the SMA, ``returns``
    the daily simple returns, ``cash_returns`` a flat positive T-bill leg. Business-day
    index kept well below the pandas ns-Timestamp horizon.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days, name="date")
    cols = [f"A{k}" for k in range(n_assets)]
    mu_bull = bull_drift_ann / TRADING_DAYS
    mu_bear = bear_drift_ann / TRADING_DAYS

    ret = np.empty((n_days, n_assets))
    for k in range(n_assets):
        sigma = base_vol + vol_spread * k
        # Markov regime path: state 0 = bull, 1 = bear.
        state = np.empty(n_days, dtype=int)
        state[0] = 0
        u = rng.random(n_days)
        for t in range(1, n_days):
            if state[t - 1] == 0:
                state[t] = 0 if u[t] < p_stay_bull else 1
            else:
                state[t] = 1 if u[t] < p_stay_bear else 0
        z = rng.standard_normal(n_days)
        drift = np.where(state == 0, mu_bull, mu_bull - edge * mu_bear)
        vol = np.where(state == 0, sigma, sigma * (1.0 + edge * (bear_vol_bump - 1.0)))
        ret[:, k] = drift + vol * z

    returns = pd.DataFrame(ret, index=idx, columns=cols)
    prices = 100.0 * (1.0 + returns).cumprod()
    cash = pd.Series(cash_ann / TRADING_DAYS, index=idx, name="cash")
    return prices, returns, cash

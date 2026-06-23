"""Data layer for Study 378 — ETF NAV premium/discount (real tape + synthetic control).

Two sources, both offline-friendly:

* **Real tape.** Daily adjusted closes (yfinance, no key) for three target ETFs — **HYG**
  (US high-yield), **EMB** (EM sovereign $), **MUB** (US munis) — plus the instruments we
  build each one's **NAV proxy** from (a sister credit ETF + a rate leg), cached under
  ``_cache/etf_prices.csv`` (wide CSV, one column per ticker). True intraday iNAV / official
  end-of-day NAV is *not* on the free yfinance endpoint, so we construct an explicit,
  transparent **NAV proxy**: a hedge-basket fair value, and define the transient
  **premium/discount basis** as the ETF's cumulative log-return *residual* to that fair
  value, detrended by a rolling mean (so it is stationary, like a real prem/disc series).

* **Synthetic.** A deterministic, fixed-seed generator that builds a basis with a tunable
  mean-reversion speed and a **planted harvestable edge knob** (``edge_bps``): the residual
  earned back after a discount. With ``edge_bps = 0`` the control must NOT manufacture
  significance; with a large planted edge it must light up. Positive control for the engine.

Pure numpy + pandas + stdlib for the offline path. ``fetch_etfs`` (network) is only used
once to build the cache and is never imported by the notebooks' offline cells.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(HERE, "..", "_cache", "etf_prices.csv")

# Target ETFs and the hedge legs each NAV proxy is built from. The legs are liquid,
# long-listed instruments that co-move with the target (a sister credit ETF + a rate leg);
# the residual of the target to a return-hedge on the legs is our transient prem/disc basis.
# This is a PROXY for the official NAV, named as such everywhere (true iNAV is not free).
TARGETS: dict[str, list[str]] = {
    "HYG": ["JNK", "IEF"],            # US high yield, hedged by a sister HY ETF + 7-10y rates
    "EMB": ["EMLC", "LQD", "IEF"],    # EM sovereign $, hedged by EM-local + IG credit + rates
    "MUB": ["TFI", "IEF"],           # US munis, hedged by a sister muni ETF + rates
}
# Every ticker we need to cache (targets + all hedge legs), de-duplicated.
ALL_TICKERS = sorted({t for t in TARGETS} | {l for legs in TARGETS.values() for l in legs})

ROLL_DETREND = 60       # rolling-mean window that detrends the integrated residual -> basis
Z_WINDOW = 252          # rolling window for the discount z-score


# --------------------------------------------------------------------------- #
# Real tape
# --------------------------------------------------------------------------- #
def fetch_etfs(start: str = "2012-01-01", end: str | None = None,
               path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Download the target ETFs + hedge legs via yfinance and cache a wide close CSV.

    Network-only; used once to build ``_cache/etf_prices.csv``. Never imported by the
    offline notebook cells.
    """
    import yfinance as yf

    raw = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all").ffill(limit=2)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw.to_csv(path)
    return raw


def have_real(path: str = DEFAULT_CACHE) -> bool:
    return os.path.exists(path)


def load_prices(path: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load the cached wide close frame (index = date, columns = tickers)."""
    return pd.read_csv(path, index_col=0, parse_dates=True).sort_index().ffill(limit=2)


def nav_basis(prices: pd.DataFrame, target: str, legs: list[str] | None = None,
              detrend: int = ROLL_DETREND) -> pd.DataFrame:
    """Build the transient premium/discount **basis** for one ETF against its NAV proxy.

    Procedure (all on adjusted closes, fully transparent):
      1. Fit a full-sample OLS of the target's daily log-return on the legs' log-returns
         (an intercept + slopes): the **hedge / NAV-proxy return**.
      2. The daily **residual** = target return − hedged return. Its cumulative sum is the
         ETF's drift *relative to fair value*; integrated, that drifts (it is I(1)-ish), so
      3. we **detrend** it by a rolling mean (``detrend`` days). The result is a stationary,
         mean-reverting **basis**: positive = ETF rich vs proxy NAV (**premium**), negative =
         ETF cheap (**discount**).

    Returns a frame indexed by date with columns:
      * ``price`` — target ETF adjusted close
      * ``resid`` — daily hedged residual return (the harvestable building block)
      * ``basis`` — the detrended cumulative residual = transient premium(+)/discount(-)
    """
    if legs is None:
        legs = TARGETS[target]
    cols = [target] + legs
    P = prices[cols].dropna()
    lr = np.log(P).diff().dropna()
    X = lr[legs].values
    y = lr[target].values
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = pd.Series(y - Xc @ beta, index=lr.index)
    cum = resid.cumsum()
    basis = cum - cum.rolling(detrend, min_periods=20).mean()
    out = pd.DataFrame({
        "price": P[target].reindex(basis.index).ffill(),
        "resid": resid.reindex(basis.index),
        "basis": basis,
    }).dropna()
    return out


def load_real(path: str = DEFAULT_CACHE) -> dict[str, pd.DataFrame]:
    """Cached prices -> {target: basis frame} for every target ETF, in one call."""
    prices = load_prices(path)
    return {t: nav_basis(prices, t) for t in TARGETS}


# --------------------------------------------------------------------------- #
# Synthetic positive control
# --------------------------------------------------------------------------- #
def synthetic_basis(n_days: int = 3_500, edge_bps: float = 0.0, phi: float = 0.90,
                    sig_basis: float = 0.0025, sig_resid: float = 0.0020,
                    seed: int = 378) -> pd.DataFrame:
    """Deterministic prem/disc series with a tunable **planted harvestable edge**.

    The ``basis`` is an AR(1) (persistence ``phi``) so it mean-reverts like a real
    premium/discount. The daily hedged ``resid`` is noise PLUS, when the basis is in a
    **discount** (below its rolling z = -1), a planted drift of ``edge_bps`` per day pulling
    the residual back up — the "earn-back" a believer hopes for. With ``edge_bps = 0`` there
    is no harvest and the inference must NOT find one; with a large ``edge_bps`` the detector
    + inference must recover it. ``price`` is a cosmetic random-walk level for plotting.

    Date index is a decorative ``period_range`` (monthly→daily not needed; we use a plain
    business-day-like ``period_range`` to avoid any out-of-bounds timestamp issues on long
    series).
    """
    rng = np.random.default_rng(seed)
    basis = np.empty(n_days)
    basis[0] = 0.0
    for t in range(1, n_days):
        basis[t] = phi * basis[t - 1] + rng.normal(0, sig_basis)
    # rolling z of the basis (same recipe as the real detector) to mark discounts
    bs = pd.Series(basis)
    z = (bs - bs.rolling(Z_WINDOW, min_periods=60).mean()) / bs.rolling(
        Z_WINDOW, min_periods=60).std()
    discount = (z < -1.0).fillna(False).values
    resid = rng.normal(0, sig_resid, size=n_days)
    resid[discount] += edge_bps / 1e4        # planted earn-back on discount days
    price = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.006, size=n_days)))
    idx = pd.period_range("2005-01", periods=n_days, freq="D").to_timestamp()
    return pd.DataFrame({"price": price, "resid": resid, "basis": basis}, index=idx)

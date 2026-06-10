"""The book — a vol-targeted contrarian dip-buyer with a trend/macro gate.

The steelman strategy these leverage sellers pitch: *lever a smart timing model and you beat
buy-and-hold.* We build a faithful one:

  1. **Contrarian dip-buying** — lean INTO short-term weakness (a low RSI(2) raises target
     exposure), the documented edge of dip-buyers; take profit into strength.
  2. **Vol-targeting** — target a constant risk: ``exposure ≈ target_vol / realised_vol``, capped.
     This *cuts* leverage exactly when volatility spikes (crashes), instead of riding it down.
  3. **Trend gate** — flatten below the long moving average (don't buy a falling knife in a real
     bear), except a capitulation re-entry.

Exposure is in units of capital (0 = cash, 1 = fully invested, 2 = 2× levered). The point of the
study is what financing that exposure *really* costs — that lives in :mod:`house_edge.costs`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _rsi(price: pd.Series, n: int) -> pd.Series:
    d = price.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn)


def exposure(price: pd.Series, target_vol: float = 0.15, lev_cap: float = 2.0,
             ma_window: int = 200, vol_window: int = 20) -> pd.Series:
    """Target exposure (0..lev_cap) for the vol-targeted contrarian dip-buyer.

    ``exposure = clip(target_vol / realised_vol, 0, lev_cap)`` then: ×1.3 when RSI(2) < 10
    (lean into a dip), ×0.6 when RSI(2) > 90 (take profit), flat below the MA (trend gate) —
    except a capitulation re-entry (below MA *and* RSI(2) < 5) capped at 1×.
    """
    ret = price.pct_change()
    vol = ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    ma = price.rolling(ma_window).mean()
    rsi2 = _rsi(price, 2)
    e = (target_vol / vol).clip(0, lev_cap)
    e = e.where(rsi2 >= 10, e * 1.3)          # contrarian add into weakness
    e = e.where(rsi2 <= 90, e * 0.6)          # take profit into strength
    below = price < ma
    e = e.mask(below, 0.0)                     # trend gate
    capit = below & (rsi2 < 5)
    e = e.mask(capit, (target_vol / vol).clip(0, 1.0))
    return e.clip(0, lev_cap).rename("exposure")


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, Calmar for a daily return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "calmar", "n_days")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "cagr": float(cagr), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "max_drawdown": float(dd), "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
            "n_days": int(len(r))}

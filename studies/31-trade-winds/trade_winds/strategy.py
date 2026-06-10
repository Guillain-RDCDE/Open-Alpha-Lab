"""The book — time-series momentum, sized to equal risk, held as a diversified portfolio.

For each market independently:
  1. **Signal** — the sign of a blended trailing return (1/3/12-month), lagged. Long if it's been
     rising, short if falling. No cross-market ranking — each market is judged on its own past.
  2. **Per-market vol scaling** — divide each position by its own realised vol, so a quiet bond
     future and a wild natural-gas future contribute the *same* risk (the managed-futures staple).
  3. **Portfolio target vol** — scale the whole book to a constant annual volatility.

The point of diversification: ~18 markets across four asset classes, each a thin trend signal, add
up to a portfolio Sharpe well above any single market — and one that tends to pay when equities fall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def tsmom_signal(returns: pd.DataFrame, lookbacks=(21, 63, 252)) -> pd.DataFrame:
    """Blended time-series-momentum signal in {-1,0,+1}: the average sign of trailing cumulative
    returns over each lookback (in days), lagged one day so it's tradable. Long a rising market,
    short a falling one."""
    prices = (1.0 + returns.fillna(0.0)).cumprod()
    votes = [np.sign(prices / prices.shift(lb) - 1.0) for lb in lookbacks]
    sig = sum(votes) / len(votes)
    return np.sign(sig).shift(1).fillna(0.0)


def positions(returns: pd.DataFrame, lookbacks=(21, 63, 252), vol_window: int = 63,
              target_vol: float = 0.10, market_cap: float = 3.0) -> pd.DataFrame:
    """Equal-risk TSMOM positions: ``sign × (per_market_vol_target / realised_vol)``, then the whole
    book is scaled to ``target_vol`` annualised. ``market_cap`` bounds any single market's leverage."""
    sig = tsmom_signal(returns, lookbacks)
    realised = returns.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    per_mkt_target = target_vol / np.sqrt(max(1, returns.shape[1]))   # equal risk budget per market
    raw = sig * (per_mkt_target / realised).clip(upper=market_cap)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # scale the assembled book to the portfolio target vol (ex-ante, on trailing portfolio vol)
    book_ret = (raw.shift(1) * returns).sum(axis=1)
    book_vol = book_ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / book_vol).clip(upper=3.0).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return raw.mul(scale, axis=0)


def book_returns(returns: pd.DataFrame, cost_bps: float = 2.0, **kw) -> pd.Series:
    """Net daily return of the TSMOM book: positions applied to next-day returns, minus turnover cost
    (``cost_bps`` per unit traded, round-trip-ish — futures are cheap, which is the point)."""
    pos = positions(returns, **kw)
    gross = (pos.shift(1) * returns).sum(axis=1)
    cost = (cost_bps * 1e-4) * pos.diff().abs().sum(axis=1)
    return (gross - cost).rename("tsmom")


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, Calmar, skew for a daily return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "calmar", "skew", "n_days")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "cagr": float(cagr), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "max_drawdown": float(dd), "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
            "skew": float(r.skew()), "n_days": int(len(r))}

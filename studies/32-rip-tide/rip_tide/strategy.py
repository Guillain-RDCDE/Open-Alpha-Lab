"""The book — short-horizon contrarian: fade each market's own recent move, sized to equal risk.

The deliberate mirror of Study 31's trend book. For each market independently:
  1. **Signal** — the *negative* sign of a blended trailing short return (1/3/5-day), known at the close
     of day t. SHORT a market that just rose, LONG one that just fell. This is the only line that
     differs from TSMOM.
  2. **Per-market vol scaling** — divide each position by its own realised vol, so every market
     contributes the same risk (the managed-futures staple).
  3. **Portfolio target vol** — scale the whole book to a constant annual volatility.
  4. **One execution lag** — the position decided at the close of t earns the return of t+1, applied
     once, in :func:`book_returns`. (Never twice: a double lag executes a 1-5-day reversal at t+2 and
     measures the implementation, not the market.)

Because the signal looks back only a few days, it **flips constantly** — turnover is an order of
magnitude higher than the trend book. That is the whole tension of the study: the reversal can be a
real statistical effect and still be eaten alive by transaction costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def reversal_signal(returns: pd.DataFrame, lookbacks=(1, 3, 5)) -> pd.DataFrame:
    """Blended short-horizon contrarian signal in {-1,0,+1}: the *negative* average sign of trailing
    cumulative returns over each (short) lookback in days. Short a market that has just risen, long one
    that has just fallen.

    The signal at row ``t`` uses returns **through the close of day t** — it is known at that close and
    NOT lagged here. The single execution lag (position set at the close of ``t`` earns the return of
    ``t+1``) is applied once, in :func:`book_returns`, via ``positions.shift(1)``. Lagging in both places
    would execute a 1–5-day reversal at ``t+2`` and skip the very bounce it claims to fade."""
    prices = (1.0 + returns.fillna(0.0)).cumprod()
    votes = [np.sign(prices / prices.shift(lb) - 1.0) for lb in lookbacks]
    sig = sum(votes) / len(votes)
    return (-np.sign(sig)).fillna(0.0)


def positions(returns: pd.DataFrame, lookbacks=(1, 3, 5), vol_window: int = 63,
              target_vol: float = 0.10, market_cap: float = 3.0) -> pd.DataFrame:
    """Equal-risk contrarian positions: ``signal × (per_market_vol_target / realised_vol)``, then the
    whole book is scaled to ``target_vol`` annualised. ``market_cap`` bounds any single market's
    leverage. Identical sizing to Study 31's trend book — only the signal's sign differs.

    Row ``t`` is the position **decided at the close of day t** (signal and vol estimates use data
    through ``t``); it is not yet lagged — :func:`book_returns` applies the one execution lag."""
    sig = reversal_signal(returns, lookbacks)
    realised = returns.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    per_mkt_target = target_vol / np.sqrt(max(1, returns.shape[1]))   # equal risk budget per market
    raw = sig * (per_mkt_target / realised).clip(upper=market_cap)
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    book_ret = (raw.shift(1) * returns).sum(axis=1)
    book_vol = book_ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    scale = (target_vol / book_vol).clip(upper=3.0).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return raw.mul(scale, axis=0)


def book_returns(returns: pd.DataFrame, cost_bps: float = 2.0, **kw) -> pd.Series:
    """Net daily return of the contrarian book: the position decided at the close of ``t`` earns the
    return of ``t+1`` (the **single** execution lag, applied here via ``pos.shift(1)``), minus turnover
    cost (``cost_bps`` per unit traded). The book flips fast, so the cost term is where the dream dies."""
    pos = positions(returns, **kw)
    gross = (pos.shift(1) * returns).sum(axis=1)
    cost = (cost_bps * 1e-4) * pos.diff().abs().sum(axis=1)
    return (gross - cost).rename("reversion")


def turnover(returns: pd.DataFrame, **kw) -> float:
    """Average daily one-way turnover (sum of |Δposition|) — the contrarian book's Achilles heel.
    Compare with the trend book: short-horizon reversal trades many times more, so a per-unit cost that
    a slow trend book shrugs off is fatal here."""
    pos = positions(returns, **kw)
    return float(pos.diff().abs().sum(axis=1).mean())


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

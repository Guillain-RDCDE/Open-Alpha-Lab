"""The teardown that earns the stamps — prove the two-sided edge is look-ahead, and price the honest one.

Two legs, and a smoking gun:

  1. :func:`lookahead_bias` — the Sharpe gap between the two-sided (peeking) and one-sided (causal)
     books, with a Newey-West t on each stream. On a random walk the two-sided t is wildly significant
     (a fabricated edge) while the one-sided t is ~0; that contrast is the `BUSTED` verdict.
  2. :func:`future_leakage` — the smoking gun. Correlate today's *cycle* with **future** returns. The
     two-sided cycle is built from those future returns, so it correlates with them by construction; the
     one-sided cycle, which can't see them, does not. A non-zero forward correlation is direct evidence
     the trend "knows" the future.
  3. :func:`honest_edge` — the one-sided book's own HAC t, so we can say plainly whether anything real
     survives once the peeking stops (on a mean-reverting tape: a little; on a random walk: nothing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .hp import cycle
from .strategy import reversion_returns, summary

TRADING_DAYS_PER_YEAR = 252


def _mean_tstat_hac(r: np.ndarray) -> float:
    r = np.asarray(r, float); r = r[~np.isnan(r)]
    n = r.size; mu = r.mean(); e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e / n)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0); lrv += 2.0 * w * float(e[k:] @ e[:-k] / n)
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else np.nan


def lookahead_bias(close: pd.Series, cost_bps: float = 1.0, lam: float = 1e6, window: int = 252,
                   periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Two-sided vs one-sided cycle-reversion: Sharpe gap and the HAC t of each stream.

    A large two-sided Sharpe with a large *t*, against a near-zero one-sided Sharpe, is the look-ahead
    bias laid bare — the spectacular backtest exists only because the filter peeked.
    """
    two = reversion_returns(close, cost_bps=cost_bps, lam=lam, causal=False)
    one = reversion_returns(close, cost_bps=cost_bps, lam=lam, causal=True, window=window)
    idx = one.index.intersection(two.index)
    two, one = two.reindex(idx), one.reindex(idx)
    s_two, s_one = summary(two, periods_per_year), summary(one, periods_per_year)
    return {
        "two_sided_sharpe": s_two["sharpe"], "two_sided_t": _mean_tstat_hac(two.to_numpy()),
        "one_sided_sharpe": s_one["sharpe"], "one_sided_t": _mean_tstat_hac(one.to_numpy()),
        "lookahead_sharpe_gap": float(s_two["sharpe"] - s_one["sharpe"]),
        "n_days": int(len(idx)),
    }


def future_leakage(close: pd.Series, lam: float = 1e6, window: int = 252, horizons=(1, 5, 10)) -> dict:
    """Correlation of today's cycle with **future** returns — the direct fingerprint of look-ahead.

    For each horizon h, correlate ``cycle_t`` with the forward h-day return. The two-sided cycle was
    computed using those very returns, so it correlates with them; the one-sided cycle cannot, so its
    forward correlation is ~0. Returns both, per horizon.
    """
    r = close.pct_change()
    c2 = cycle(close, lam=lam, causal=False)
    c1 = cycle(close, lam=lam, causal=True, window=window)
    out = {}
    for h in horizons:
        fwd = (1.0 + r).rolling(h).apply(np.prod, raw=True).shift(-h) - 1.0
        d2 = pd.DataFrame({"c": c2, "f": fwd}).dropna()
        d1 = pd.DataFrame({"c": c1, "f": fwd}).dropna()
        out[h] = {
            "two_sided_corr": float(np.corrcoef(d2["c"], d2["f"])[0, 1]) if len(d2) > 2 else np.nan,
            "one_sided_corr": float(np.corrcoef(d1["c"], d1["f"])[0, 1]) if len(d1) > 2 else np.nan,
        }
    return out


def honest_edge(close: pd.Series, cost_bps: float = 1.0, lam: float = 1e6, window: int = 252,
                periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """The one-sided (tradable) book on its own: Sharpe and HAC t — what really survives the peek-stop."""
    one = reversion_returns(close, cost_bps=cost_bps, lam=lam, causal=True, window=window)
    s = summary(one, periods_per_year)
    return {"sharpe": s["sharpe"], "ann_return": s["ann_return"], "t_stat": _mean_tstat_hac(one.to_numpy()),
            "max_drawdown": s["max_drawdown"], "n_days": s["n_days"]}

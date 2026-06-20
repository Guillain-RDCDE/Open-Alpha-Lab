"""The carbon-credit engine and its honest controls — Study 304 (Carbon-Credits).

The folk/green claim: *"carbon allowances only go one way — the cap shrinks every year and
polluters must buy a dwindling supply, so just buy and hold a carbon ETF (KRBN) and ride
the energy transition."* We build the buy-and-hold leg and pin it against the two things
that decide whether the claim survives:

1. **Cash.** Buy-and-hold is only a "reward" if it beats sitting in T-bills — so the
   headline is the **excess-of-cash** mean and its HAC *t* (the inference-bar number), not
   the raw level.
2. **Timing.** If carbon is just a volatile commodity rather than a one-way escalator, a
   simple **time-series (absolute) momentum** overlay — hold KRBN only while its trailing
   return is positive, else sit in cash — should beat naive buy-and-hold on a
   risk-adjusted basis. We race the two **excess-of-cash to excess-of-cash** so the
   comparison is fair, put the Sharpe *difference* through a Newey-West (HAC) *t* and a
   **circular block bootstrap** CI.

Conventions: returns are simple daily total returns from the price frame. The momentum
signal is known at the close of *t* and earns the return of *t+1* — **one** ``shift``,
applied once (documented effective lag = 1 bar). Turnover is charged **one-way × NAV** at
``cost_bps`` each time the overlay flips in/out of the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Return helpers
# ---------------------------------------------------------------------------
def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a total-return price frame (first row dropped)."""
    return prices.pct_change().dropna()


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


# ---------------------------------------------------------------------------
# Buy-and-hold
# ---------------------------------------------------------------------------
def buy_and_hold(returns: pd.DataFrame, asset: str = "KRBN") -> pd.Series:
    """The daily return series of a buy-and-hold position in ``asset``."""
    return returns[asset].rename("buy_and_hold")


# ---------------------------------------------------------------------------
# Time-series (absolute) momentum overlay
# ---------------------------------------------------------------------------
def momentum_signal(
    prices: pd.DataFrame,
    asset: str = "KRBN",
    lookback: int = 126,
    skip: int = 0,
) -> pd.Series:
    """Boolean 'in the market' signal: trailing ``lookback``-day return > 0.

    The signal is stamped on the **close of bar *t*** using only data up to *t* (the
    trailing total return over ``[t-lookback-skip, t-skip]``). ``skip`` lets you drop the
    most recent ``skip`` bars (the classic 12-1 momentum skips the last month). The default
    ``lookback=126`` ≈ 6 months, the standard time-series-momentum window, halved here
    because the real KRBN tape is short.

    The returned series is True when the trailing return is positive (hold the asset),
    False otherwise (go to cash). It is **not** yet lagged — :func:`overlay_returns`
    applies the single execution shift.
    """
    px = prices[asset]
    trailing = px.shift(skip) / px.shift(lookback + skip) - 1.0
    return (trailing > 0.0).fillna(False)


def overlay_returns(
    returns: pd.DataFrame,
    signal: pd.Series,
    asset: str = "KRBN",
    cash: str = "CASH",
    cost_bps: float = 0.0,
) -> pd.Series:
    """Daily net return of the momentum overlay: asset when in, cash when out.

    Execution lag: the signal known at the close of *t* governs the position **held over
    *t+1*** — exactly one ``shift(1)`` (documented effective lag = 1 bar). On every day the
    held position flips (0↔1) one-way turnover of 100% NAV is charged at ``cost_bps``.

    Returns a daily net-return Series aligned to ``returns.index``.
    """
    pos = signal.reindex(returns.index).fillna(False).astype(float)
    held = pos.shift(1).fillna(0.0)                      # the single execution lag
    asset_r = returns[asset]
    cash_r = returns[cash] if cash in returns.columns else pd.Series(0.0, index=returns.index)
    gross = held * asset_r + (1.0 - held) * cash_r
    flips = held.diff().abs().fillna(held.abs())        # one-way NAV turnover per day
    net = gross - flips * (cost_bps * 1e-4)
    return net.rename("momentum_overlay")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def stats(net: pd.Series, rf: pd.Series | None = None) -> dict:
    """Headline stats for a daily return series.

    If ``rf`` (a daily cash return series) is given, the Sharpe is computed on the
    **excess-of-cash** return so two arms with different cash exposure are compared
    like-for-like; CAGR/vol/drawdown stay on the raw total-return series.
    """
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if rf is not None:
        ex = (net - rf.reindex(net.index).fillna(0.0)).astype(float)
    else:
        ex = net
    sharpe = (float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS))
              if ex.std(ddof=1) > 0 else float("nan"))
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "final": float(equity.iloc[-1]),
    }


# ---------------------------------------------------------------------------
# Inference: HAC t-stat and circular block bootstrap
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dependency)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def excess_tstat(net: pd.Series, rf: pd.Series, lags: int | None = None) -> float:
    """HAC t-stat on the **excess-of-cash** daily mean — the inference-bar number.

    A buy-and-hold "reward" is real only if its mean return *above cash* is distinguishable
    from zero; this returns that statistic.
    """
    ex = (net - rf.reindex(net.index).fillna(0.0)).astype(float).to_numpy()
    return hac_tstat(ex, lags=lags)


def _annualised_sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def bootstrap_sharpe_diff(
    a: pd.Series,
    b: pd.Series,
    rf: pd.Series | None = None,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 304,
) -> dict:
    """Circular block bootstrap CI for the Sharpe *difference* (arm ``a`` − arm ``b``).

    Resamples the two aligned **excess-of-cash** return series *jointly* in circular blocks
    (preserving cross-correlation and volatility clustering), recomputes each arm's
    annualised Sharpe per resample, and returns the point difference, its 95% CI, and the
    bootstrap fraction of resamples in which ``a``'s Sharpe exceeds ``b``'s.
    """
    idx = a.index.intersection(b.index)
    ra = a.reindex(idx).to_numpy(dtype=float)
    rb = b.reindex(idx).to_numpy(dtype=float)
    if rf is not None:
        f = rf.reindex(idx).fillna(0.0).to_numpy(dtype=float)
        ra = ra - f
        rb = rb - f
    n = len(idx)
    point = _annualised_sharpe(ra) - _annualised_sharpe(rb)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    a_wins = 0
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        offsets = (starts[:, None] + np.arange(block)[None, :]) % n
        sel = offsets.reshape(-1)[:n]
        sa, sb = _annualised_sharpe(ra[sel]), _annualised_sharpe(rb[sel])
        diffs[i] = sa - sb
        if sa > sb:
            a_wins += 1
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "point": float(point),
        "ci95": (float(lo), float(hi)),
        "frac_a_wins": a_wins / n_boot,
        "n": n, "block": block, "n_boot": n_boot,
    }

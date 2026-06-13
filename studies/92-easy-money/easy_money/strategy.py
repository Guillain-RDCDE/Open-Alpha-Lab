"""The short-vol carry strategy and its honest tail accounting — Study 92 (Easy-Money).

The folk claim: 'VIX futures are almost always in contango, so a long-vol ETP like
VIXY bleeds lower every single day. Just short it (or hold the inverse) and harvest
the roll — it's near-free money, a steady carry.' We implement the short and confront
it with the only thing that decides whether it is free money or a paid-for risk
premium: **the tail**.

- ``decay_stats`` — documents the *real, large* decay of the long-ETP itself (its
  CAGR, total wipeout, worst/best day). The carry the short would harvest is real.
- ``short_vol_backtest`` — short the long-ETP (equivalently, hold the inverse), net
  of a **one-way trade cost** and a **daily stock-borrow fee** (shorts pay borrow,
  every day, exactly once). Returns the full net daily series plus the tail
  statistics that the headline Sharpe hides: **skew, excess kurtosis, worst single
  day, and max drawdown**.
- ``sized_to_survive`` — the honest sizing question: scale the short so the worst
  historical day costs at most a chosen fraction of capital, and see what the
  steady drip is worth once the position is small enough to survive the spike.

Execution convention is the desk's: the short is sized on information known at the
close of day *t*, so it earns the return of *t+1* — one ``shift``, applied once.
Costs are one-way per turnover unit; borrow is charged daily on the gross short
notional.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# The decay of the long-vol ETP itself — is the carry real?
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def decay_stats(close: pd.Series) -> dict:
    """Headline decay statistics of a (long-vol) ETP NAV series.

    Quantifies how much the long-ETP loses just by existing: total return over the
    window, annualised CAGR, annualised vol, and the single worst/best day. A large
    negative CAGR here IS the contango carry the short would collect — the carry is
    real; the rest of the study is whether you can keep it.
    """
    close = close.dropna().astype(float)
    ret = close.pct_change().dropna()
    n = len(ret)
    years = n / TRADING_DAYS
    total = float(close.iloc[-1] / close.iloc[0] - 1.0)
    cagr = float((close.iloc[-1] / close.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "worst_day": float(ret.min()),
        "best_day": float(ret.max()),
        "n_days": n,
        "years": years,
        "final_frac_of_start": float(close.iloc[-1] / close.iloc[0]),
    }


# ---------------------------------------------------------------------------
# The short-vol carry backtest, net of costs + borrow, with tail stats
# ---------------------------------------------------------------------------
def short_vol_backtest(
    close: pd.Series,
    leverage: float = 1.0,
    cost_bps: float = 5.0,
    borrow_bps_annual: float = 300.0,
    lag: int = 1,
) -> dict:
    """Short the long-vol ETP at constant ``leverage`` of capital; harvest the bleed.

    The book is short ``leverage`` units of the ETP (so it *earns* ``-leverage *
    ret`` of the underlying — a falling ETP is a gain). We charge:

    - **borrow**: ``borrow_bps_annual`` per year on the gross short notional, accrued
      daily — shorts pay to borrow the shares, every day, exactly once.
    - **trade cost**: ``cost_bps`` one-way on the turnover needed to keep the notional
      at ``leverage`` as the price moves (rebalancing the short). Entry is charged
      once at the start.

    One execution lag: the position is known at close *t* and earns *t+1*'s return.
    Returns net daily series + headline stats AND the tail stats the Sharpe hides:
    skew, excess kurtosis, worst single day, max drawdown.
    """
    ret = close.pct_change().fillna(0.0)
    # Short position earns the negative of the underlying return, lagged one day.
    pos = pd.Series(-leverage, index=close.index)
    pos = pos.shift(lag).fillna(0.0)
    gross = pos.abs()

    # Daily borrow on the gross short notional.
    borrow = gross * (borrow_bps_annual * 1e-4) / TRADING_DAYS
    # Turnover: rebalancing the constant-leverage short as price moves, plus entry.
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * cost_bps * 1e-4

    net = pos * ret - borrow - cost
    return _stats(net, gross)


def _stats(net: pd.Series, gross: pd.Series) -> dict:
    net = net.astype(float)
    equity = (1.0 + net).cumprod()
    n = len(net)
    years = n / TRADING_DAYS
    final = float(equity.iloc[-1])
    # Equity can go non-positive in a blow-up; guard the CAGR.
    if final > 0 and years > 0:
        cagr = float(final ** (1.0 / years) - 1.0)
    else:
        cagr = -1.0
    sd = net.std(ddof=1)
    vol = float(sd * np.sqrt(TRADING_DAYS))
    sharpe = float(net.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "skew": float(_skew(net.to_numpy())),
        "excess_kurt": float(_excess_kurt(net.to_numpy())),
        "worst_day": float(net.min()),
        "best_day": float(net.max()),
        "final": final,
        "avg_gross": float(gross.mean()),
    }


# ---------------------------------------------------------------------------
# Moments — local, no scipy dependency for the core stats
# ---------------------------------------------------------------------------
def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return float("nan")
    return float(np.mean(((x - m) / s) ** 3))


def _excess_kurt(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return float("nan")
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


# ---------------------------------------------------------------------------
# Position sizing that survives the tail
# ---------------------------------------------------------------------------
def sized_to_survive(
    close: pd.Series,
    max_loss_on_worst_day: float = 0.25,
    cost_bps: float = 5.0,
    borrow_bps_annual: float = 300.0,
) -> dict:
    """Size the short so its worst single day costs at most ``max_loss_on_worst_day``.

    A short of leverage L loses ``L * up_move`` on a day the ETP jumps up by
    ``up_move``. To cap the worst-day loss at ``max_loss_on_worst_day`` we set
    ``L = max_loss_on_worst_day / worst_up_move`` using the realised worst up-day in
    the sample. We then re-run the carry at that survivable leverage and report what
    the steady drip is actually worth once the tail can't kill you.

    Returns the chosen leverage and the backtest dict at that leverage.
    """
    ret = close.pct_change().dropna()
    worst_up = float(ret.max())  # the largest single-day *rise* of the long-ETP
    lev = max_loss_on_worst_day / worst_up if worst_up > 0 else float("nan")
    bt = short_vol_backtest(
        close, leverage=lev, cost_bps=cost_bps, borrow_bps_annual=borrow_bps_annual
    )
    return {"leverage": lev, "worst_up_move": worst_up, "backtest": bt}


# ---------------------------------------------------------------------------
# Inference: HAC t-stat on a return series
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x`` (local, no quantlab dep)."""
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

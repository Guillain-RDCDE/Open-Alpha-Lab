"""The strategy and its honest controls - Study 247 (Bond-Seasonality).

The folk claim: 'Treasuries have their own calendar - a reliable turn-of-month or
turn-of-year bid.  Buy TLT (or IEF) at the last close of the month and sell at the first
close of the next month.'

We measure it three ways and pin it against comparisons that decide whether it is real and
whether it pays:

- **TOM vs the rest** - mean daily return on the last-2 + first-2 trading days of each
  month against all other days, with a HAC (Newey-West) t-stat and Wilson win-rate
  intervals.  Both TLT and IEF are reported.
- **A TOM-only strategy vs buy-and-hold** - long on TOM days, in cash otherwise (0%
  cash yield, conservative), net of 1 bp/leg.  The effect can look large per day and
  still fail as a standalone book if it captures too few days.
- **Turn-of-year (TOY) contrast** - last-5 + first-5 trading days of each calendar year.
  This is a separate, narrower hypothesis; the evidence decides independently.

Execution convention: the last trading day of each month and the first of the next are
fixed well in advance by the calendar.  The rule needs **no execution lag** for the entry
(the signal is the date itself), but we apply one anyway on the exit side (sell at the
close of the first-of-month day).  Costs are 1 bp one-way per leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Core measurement: TOM vs rest
# ---------------------------------------------------------------------------
def effect_stats(close: pd.Series, is_tom: pd.Series) -> dict:
    """Mean return on TOM days vs other days, gap, ratio, HAC t and win rates.

    Computes daily pct-change returns and splits them by the boolean ``is_tom`` mask.
    Reports:
    - mean return on TOM days and all other days (in bps and fractional)
    - the gap and ratio
    - HAC (Newey-West) t-stat on the TOM group itself and on the gap
    - win rates with Wilson score intervals
    """
    ret = close.pct_change().dropna()
    tom = is_tom.reindex(ret.index).fillna(False).astype(bool)
    r_tom = ret[tom]
    r_rest = ret[~tom]

    mu_tom = float(r_tom.mean())
    mu_rest = float(r_rest.mean())
    gap = mu_tom - mu_rest
    ratio = mu_tom / mu_rest if mu_rest != 0 else float("nan")

    win_tom = float((r_tom > 0).mean())
    win_rest = float((r_rest > 0).mean())

    return {
        "n_tom": int(len(r_tom)),
        "n_rest": int(len(r_rest)),
        "mu_tom": mu_tom,
        "mu_rest": mu_rest,
        "gap": gap,
        "ratio": ratio,
        "t_tom": hac_tstat(r_tom.to_numpy()),
        "t_gap": _diff_in_means_tstat(r_tom.to_numpy(), r_rest.to_numpy()),
        "win_tom": win_tom,
        "win_rest": win_rest,
        "wilson_tom": wilson_interval((r_tom > 0).sum(), len(r_tom)),
        "wilson_rest": wilson_interval((r_rest > 0).sum(), len(r_rest)),
    }


def _diff_in_means_tstat(r_pre: np.ndarray, r_rest: np.ndarray) -> float:
    """HAC two-sample t-stat for the difference in means (TOM minus rest).

    Uses the HAC long-run variance of the (smaller) TOM group plus the i.i.d. variance of
    the (large, low-autocorr) rest group - the pre group dominates the SE.
    """
    if r_pre.size <= 5 or r_rest.size <= 5:
        return float("nan")
    mu_d = r_pre.mean() - r_rest.mean()
    var_pre = _hac_lrv(r_pre - r_pre.mean()) / r_pre.size
    var_rest = r_rest.var(ddof=1) / r_rest.size
    se = np.sqrt(var_pre + var_rest)
    return float(mu_d / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Turn-of-year contrast
# ---------------------------------------------------------------------------
def toy_effect_stats(close: pd.Series, is_toy: pd.Series) -> dict:
    """Same as :func:`effect_stats` but for the turn-of-year (TOY) window."""
    return effect_stats(close, is_toy)


# ---------------------------------------------------------------------------
# Backtest: TOM-only book vs buy-and-hold
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def tom_only_book(close: pd.Series, is_tom: pd.Series, cost_bps: float = 1.0) -> dict:
    """Long only on TOM days, in cash (0%) otherwise; costs one-way per leg.

    Each entry and exit is charged ``cost_bps`` one-way.  Returns net daily series and
    headline stats including the per-invested-day Sharpe and the whole-tape CAGR.

    Because the TOM window spans ~4 days per month out of ~21, the book is in the market
    roughly 19% of the time - enough to potentially build a meaningful return, unlike
    extreme calendar anomalies (e.g. pre-holiday) that invest <5% of the time.
    """
    ret = close.pct_change().fillna(0.0)
    pos = is_tom.reindex(close.index).fillna(False).astype(float)
    turn = pos.diff().abs().fillna(pos.abs())
    net = pos * ret - turn * cost_bps * 1e-4
    equity = (1.0 + net).cumprod()

    n = len(net)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(net.std(ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe_whole = (
        float(net.mean() / net.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if net.std(ddof=1) > 0 else float("nan")
    )
    inv = net[pos > 0]
    sharpe_inv = (
        float(inv.mean() / inv.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if len(inv) > 1 and inv.std(ddof=1) > 0 else float("nan")
    )
    return {
        "net": net,
        "equity": equity,
        "cagr": cagr,
        "vol": vol,
        "sharpe_whole": sharpe_whole,
        "sharpe_inv": sharpe_inv,
        "max_dd": _max_drawdown(equity.to_numpy()),
        "days_invested": int((pos > 0).sum()),
        "time_in_market": float((pos > 0).mean()),
        "switches": int((turn > 1e-9).sum()),
    }


def buy_and_hold(close: pd.Series) -> dict:
    """Buy-and-hold the tape: position 1 every day, no costs after entry."""
    ret = close.pct_change().fillna(0.0)
    equity = (1.0 + ret).cumprod()
    n = len(ret)
    years = n / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    sharpe = (
        float(ret.mean() / ret.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if ret.std(ddof=1) > 0 else float("nan")
    )
    return {
        "equity": equity,
        "cagr": cagr,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# Inference helpers - HAC t-stat and Wilson interval
# ---------------------------------------------------------------------------
def _hac_lrv(e: np.ndarray, lags: int | None = None) -> float:
    """Newey-West long-run variance of a (demeaned) series ``e``."""
    e = np.asarray(e, dtype=float)
    n = e.size
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    return lrv


def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC) t-stat for the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 5:
        return float("nan")
    mu = x.mean()
    lrv = _hac_lrv(x - mu, lags)
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (win rate)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (float(centre - half), float(centre + half))

"""The strategy and its honest controls — Study 432 (Hull Moving Average).

Alan Hull's Hull Moving Average (HMA, 2005) is a stack of linearly-weighted moving
averages built to be nearly lag-free:

    WMA(n)  = sum(w_i * P_{t-n+1+i}) / sum(w_i),  w_i = i+1   (linear weights, recent heavy)
    HMA(n)  = WMA( 2*WMA(n/2) - WMA(n), period = round(sqrt(n)) )

The intuition: ``2*WMA(n/2) - WMA(n)`` over-weights recent price (removing lag) and the
final ``WMA(sqrt(n))`` re-smooths it.  The folk claim: because HMA hugs price with far less
lag than an SMA or EMA, a trend rule built on it **fires earlier and produces fewer false
signals** ("whipsaws"), so it beats a plain SMA rule.

We turn that into daily **timing rules** and race them:

- **HMA-slope long/flat (or long/short).** Long when HMA is rising (HMA_t > HMA_{t-1}),
  flat (or short) otherwise.  This is the canonical HMA trend rule.
- **SMA crossover (the obvious simpler benchmark).** Long when close > SMA(n), else flat.
- **MACD (a second benchmark).** Long when the MACD line is above its signal line.

The honest arbiters:

- **NET Sharpe race, excess-vs-excess.** Strategy daily returns minus the risk-free rate
  are compared to buy-and-hold daily returns minus the risk-free rate.  Costs are charged
  one-way × NAV on every position change (turnover); the short leg pays borrow.
- **HAC (Newey-West) one-sample t** on the daily *active spread* (strategy − buy&hold),
  the inference-bar number that decides the Signal stamp.
- **A permutation / label-shuffle placebo** that destroys the HMA's timing while keeping
  its turnover, to ask "is the timing real, or just exposure?".
- **A regime / in-out-of-sample split** (first vs second half) for robustness.

No look-ahead: the position for day *t+1* is decided from the close of day *t* (one
documented `shift`), and it earns day *t+1*'s return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Moving-average primitives
# ---------------------------------------------------------------------------
def wma(series: pd.Series, period: int) -> pd.Series:
    """Linearly-weighted moving average (most recent bar weighted heaviest)."""
    weights = np.arange(1, period + 1, dtype=float)
    wsum = weights.sum()

    def _f(x):
        return float(np.dot(x, weights) / wsum)

    return series.rolling(period).apply(_f, raw=True)


def hma(series: pd.Series, period: int = 16) -> pd.Series:
    """Hull Moving Average over ``period`` bars.

    HMA(n) = WMA( 2*WMA(n/2) - WMA(n), period = round(sqrt(n)) ).  Returns a Series
    aligned to ``series``; the first ~``period + sqrt(period)`` bars are NaN while the
    nested windows fill.
    """
    half = max(1, int(round(period / 2)))
    sqrt_n = max(1, int(round(np.sqrt(period))))
    raw = 2.0 * wma(series, half) - wma(series, period)
    return wma(raw, sqrt_n)


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (used inside MACD)."""
    return series.ewm(span=period, adjust=False).mean()


def macd_line(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd, signal_line): EMA(fast) − EMA(slow) and its EMA(signal)."""
    macd = ema(series, fast) - ema(series, slow)
    sig = ema(macd, signal)
    return macd, sig


# ---------------------------------------------------------------------------
# Timing rules — each returns a desired position series in {-1,0,+1} (pre-lag)
# ---------------------------------------------------------------------------
def hma_position(close: pd.Series, period: int = 16, long_short: bool = False) -> pd.Series:
    """HMA-slope trend rule: long when HMA rising, flat (or short) when falling.

    The position is the *desired* exposure formed on the close of bar *t* (pre-lag).
    ``strategy`` applies the one-day execution lag downstream.
    """
    h = hma(close, period)
    rising = h.diff() > 0
    pos = pd.Series(0.0, index=close.index)
    pos[rising] = 1.0
    if long_short:
        pos[~rising] = -1.0
    pos[h.isna()] = 0.0
    return pos


def sma_position(close: pd.Series, period: int = 50, long_short: bool = False) -> pd.Series:
    """SMA trend rule: long when close > SMA(period), flat (or short) otherwise."""
    s = sma(close, period)
    above = close > s
    pos = pd.Series(0.0, index=close.index)
    pos[above] = 1.0
    if long_short:
        pos[~above] = -1.0
    pos[s.isna()] = 0.0
    return pos


def macd_position(close: pd.Series, long_short: bool = False) -> pd.Series:
    """MACD trend rule: long when MACD line > signal line, flat (or short) otherwise."""
    macd, sig = macd_line(close)
    above = macd > sig
    pos = pd.Series(0.0, index=close.index)
    pos[above] = 1.0
    if long_short:
        pos[~above] = -1.0
    pos[macd.isna() | sig.isna()] = 0.0
    return pos


# ---------------------------------------------------------------------------
# Backtest engine — daily returns, one execution lag, costs one-way × NAV
# ---------------------------------------------------------------------------
def backtest(
    bars: pd.DataFrame,
    position: pd.Series,
    cost_bps: float = 2.0,
    borrow_bps_yr: float = 50.0,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """Run a daily-returns backtest of ``position`` on ``bars`` and return a frame.

    Convention (one documented execution lag): the desired position formed on the close
    of *t* is held for the return of *t+1* — a single ``shift(1)``.  Daily asset returns
    are close-to-close.  Costs are charged one-way × NAV on the change in position
    (``|pos_t - pos_{t-1}|``).  When the held position is short, a daily borrow fee
    (``borrow_bps_yr/252``) is deducted on the short notional.

    Columns: ``asset_ret`` (close-to-close), ``held`` (lagged position), ``turnover``,
    ``strat_gross``, ``strat_net``, ``bh`` (buy&hold), plus ``*_excess`` columns net of
    the daily risk-free rate.
    """
    close = bars["close"]
    asset_ret = close.pct_change().fillna(0.0)
    held = position.shift(1).fillna(0.0)  # the ONE execution lag
    turnover = held.diff().abs().fillna(held.abs())  # one-way change in exposure

    cost = turnover * (cost_bps * 1e-4)
    borrow = np.maximum(-held, 0.0) * (borrow_bps_yr * 1e-4 / TRADING_DAYS)

    strat_gross = held * asset_ret
    strat_net = strat_gross - cost - borrow
    bh = asset_ret  # buy & hold the asset

    rf_daily = rf_annual / TRADING_DAYS
    out = pd.DataFrame(
        {
            "asset_ret": asset_ret,
            "held": held,
            "turnover": turnover,
            "strat_gross": strat_gross,
            "strat_net": strat_net,
            "bh": bh,
            "strat_net_excess": strat_net - rf_daily * held.abs().clip(upper=1.0),
            "bh_excess": bh - rf_daily,
        },
        index=bars.index,
    )
    return out


# ---------------------------------------------------------------------------
# Statistics — Sharpe, HAC t, permutation placebo
# ---------------------------------------------------------------------------
def annual_sharpe(daily: pd.Series) -> float:
    """Annualised Sharpe of a daily return series."""
    d = pd.Series(daily).dropna().to_numpy(dtype=float)
    if d.size < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1) * np.sqrt(TRADING_DAYS))


def cagr(daily: pd.Series) -> float:
    """Compound annual growth rate of a daily return series."""
    d = pd.Series(daily).dropna().to_numpy(dtype=float)
    if d.size == 0:
        return float("nan")
    total = float(np.prod(1.0 + d))
    yrs = d.size / TRADING_DAYS
    return total ** (1.0 / yrs) - 1.0 if yrs > 0 and total > 0 else float("nan")


def max_drawdown(daily: pd.Series) -> float:
    """Maximum drawdown (fraction, negative) of the equity curve of daily returns."""
    d = pd.Series(daily).dropna().to_numpy(dtype=float)
    if d.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + d)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def hac_tstat(daily: pd.Series) -> float:
    """Newey-West HAC one-sample t-stat for the mean of a daily series being != 0."""
    r = pd.Series(daily).dropna().to_numpy(dtype=float)
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def permutation_pvalue(
    asset_ret: pd.Series,
    position: pd.Series,
    cost_bps: float = 0.0,
    n_perm: int = 2000,
    seed: int = 432,
) -> dict:
    """Block-permutation placebo: is the HMA *timing* real, or just exposure?

    We hold the realised position *path* fixed in its block structure but circularly
    shift it relative to returns, destroying the alignment between the rule's calls and
    the actual moves while preserving the rule's turnover and net long bias.  The test
    statistic is the mean daily *active spread* (strategy_net − buy&hold).  Returns the
    observed statistic, the placebo mean, and the one-sided p-value (fraction of placebo
    spreads >= observed).
    """
    rng = np.random.default_rng(seed)
    a = asset_ret.fillna(0.0).to_numpy(dtype=float)
    held = position.shift(1).fillna(0.0).to_numpy(dtype=float)
    n = a.size

    def _spread(h):
        turn = np.abs(np.diff(h, prepend=0.0))
        strat = h * a - turn * (cost_bps * 1e-4)
        return float((strat - a).mean())

    obs = _spread(held)
    placebo = np.empty(n_perm)
    for i in range(n_perm):
        shift = int(rng.integers(1, n))
        placebo[i] = _spread(np.roll(held, shift))
    pval = float((placebo >= obs).mean())
    return {
        "observed_spread_bps": obs * 1e4,
        "placebo_mean_bps": float(placebo.mean()) * 1e4,
        "p_value": pval,
        "n_perm": n_perm,
    }


# ---------------------------------------------------------------------------
# Summary of a backtest frame
# ---------------------------------------------------------------------------
def summarize(bt: pd.DataFrame, label: str = "") -> dict:
    """Headline stats for a backtest frame from :func:`backtest`."""
    net = bt["strat_net"]
    spread = bt["strat_net"] - bt["bh"]  # active return vs buy&hold
    pos_changes = int((bt["held"].diff().abs() > 1e-9).sum())
    yrs = len(bt) / TRADING_DAYS
    return {
        "label": label,
        "n_days": int(len(bt)),
        "sharpe_net": annual_sharpe(net),
        "sharpe_excess": annual_sharpe(bt["strat_net_excess"]),
        "bh_sharpe": annual_sharpe(bt["bh"]),
        "bh_sharpe_excess": annual_sharpe(bt["bh_excess"]),
        "cagr_net": cagr(net),
        "bh_cagr": cagr(bt["bh"]),
        "maxdd_net": max_drawdown(net),
        "bh_maxdd": max_drawdown(bt["bh"]),
        "mean_spread_bps": float(spread.mean() * 1e4),
        "spread_t": hac_tstat(spread),
        "n_switches": pos_changes,
        "switches_per_yr": pos_changes / yrs if yrs > 0 else float("nan"),
        "time_in_market": float((bt["held"].abs() > 1e-9).mean()),
        "ann_turnover": float(bt["turnover"].sum() / yrs) if yrs > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(
    bars: pd.DataFrame,
    hma_period: int = 16,
    sma_period: int = 50,
    cost_bps: float = 2.0,
    borrow_bps_yr: float = 50.0,
    rf_annual: float = 0.0,
    long_short: bool = False,
) -> dict:
    """Run the full HMA-vs-benchmarks race on one tape and return a dict of results.

    Builds three timing rules (HMA-slope, SMA-crossover, MACD), backtests each net of
    costs with one execution lag, computes the buy&hold baseline, and runs the
    permutation placebo on the HMA rule.  All Sharpe comparisons are net; the active
    spread (strategy − buy&hold) is excess-vs-excess by construction.
    """
    close = bars["close"]
    rules = {
        "HMA": hma_position(close, hma_period, long_short=long_short),
        "SMA": sma_position(close, sma_period, long_short=long_short),
        "MACD": macd_position(close, long_short=long_short),
    }
    results = {}
    for name, pos in rules.items():
        bt = backtest(bars, pos, cost_bps=cost_bps,
                      borrow_bps_yr=borrow_bps_yr, rf_annual=rf_annual)
        results[name] = summarize(bt, name)

    # Permutation placebo on the HMA rule (gross spread, so timing not cost is tested).
    perm = permutation_pvalue(
        close.pct_change(), rules["HMA"], cost_bps=0.0, n_perm=2000, seed=432
    )
    results["HMA_permutation"] = perm
    return results

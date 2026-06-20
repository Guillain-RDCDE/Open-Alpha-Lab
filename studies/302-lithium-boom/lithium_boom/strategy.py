"""The strategy and its honest controls — Study 302 (Lithium-Boom).

The believer's pitch: *the battery-metals super-cycle was a once-in-a-decade trend; ride it
with simple trend-following and you'd have caught the boom and stepped aside before the
crash.* We implement the canonical **time-series-momentum / trend** rule on the theme ETF
(LIT) and two miners (ALB, SQM):

    Hold the asset when its price is **above** its own N-day moving average (the trend is
    up); sit in **cash** (the risk-free leg) otherwise. Rebalance daily, signal known at the
    close of day *t* earns the return of day *t+1* (one ``shift``, applied once).

Three honest pins decide whether this is anything more than "the metal went up, then down":

1. **Buy-and-hold the same asset.** A trend overlay is only interesting if it beats just
   owning the thing — and, because it sits in cash part of the time, we race **excess-of-cash
   Sharpe vs excess-of-cash Sharpe** so the comparison is apples-to-apples.
2. **Costs.** Every flip in/out of the market pays a one-way cost on NAV; we sweep it.
3. **A random-timing control.** The same number of in/out switches placed at random dates —
   if the trend rule beats a coin that trades equally often, the *timing* carried information.

Look-ahead: the moving-average signal is formed from closes up to and including day *t*; the
position it implies is held over day *t+1*. Exactly one ``shift(1)`` enforces that — no
second silent lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Indicators / signal
# ---------------------------------------------------------------------------
def sma(close: pd.Series, n: int) -> pd.Series:
    """Simple moving average of period ``n``."""
    return close.rolling(n, min_periods=n).mean()


def trend_signal(bars: pd.DataFrame, sma_n: int = 200) -> pd.Series:
    """1.0 when close > SMA(``sma_n``) (trend up → hold), else 0.0 (→ cash).

    Stamped on the *close* of bar *t*. The caller applies the single execution lag.
    """
    close = bars["close"]
    sig = (close > sma(close, sma_n)).astype(float)
    sig[sma(close, sma_n).isna()] = np.nan  # undefined during warmup
    return sig


def random_timing_signal(n: int, on_fraction: float, seed: int = 0) -> np.ndarray:
    """A control 0/1 path that is 'on' roughly ``on_fraction`` of the time, at random.

    Used as the timing-skill control: same exposure budget as the trend rule, but placed
    on random days. If trend beats this, the *timing* (not just the lower average exposure)
    carried information.
    """
    rng = np.random.default_rng(seed)
    return (rng.random(n) < on_fraction).astype(float)


# ---------------------------------------------------------------------------
# Backtest engine — one lag, costs one-way x NAV
# ---------------------------------------------------------------------------
def backtest(
    bars: pd.DataFrame,
    signal: pd.Series | np.ndarray,
    cost_bps: float = 0.0,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """Run a long/flat overlay and return a per-day frame of returns.

    The position over day *t+1* is ``signal`` known at the close of *t* — a single
    ``shift(1)``. When flat (signal 0) the book earns the daily risk-free rate
    ``rf_annual / 252``. ``cost_bps`` is charged **one-way on NAV** at every change in
    position (a full flip in or out is one one-way trade).

    Columns: ``asset_ret`` (the asset's daily return), ``pos`` (the *lagged*, held
    position), ``strat_gross``, ``strat_net``.
    """
    close = bars["close"]
    asset_ret = close.pct_change().fillna(0.0)

    sig = pd.Series(np.asarray(signal, dtype=float), index=bars.index)
    pos = sig.shift(1)  # the one and only execution lag
    pos = pos.fillna(0.0)

    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR
    # Gross strategy return: in-market days earn the asset, flat days earn cash.
    strat_gross = pos * asset_ret + (1.0 - pos) * rf_daily

    # Cost: one-way on NAV at each position change.
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * (cost_bps * 1e-4)
    strat_net = strat_gross - cost

    return pd.DataFrame(
        {
            "asset_ret": asset_ret,
            "pos": pos,
            "strat_gross": strat_gross,
            "strat_net": strat_net,
        },
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------
def sharpe(returns: np.ndarray, rf_daily: float = 0.0) -> float:
    """Annualised Sharpe of a daily-return series (excess of ``rf_daily``)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    ex = r - rf_daily
    sd = ex.std(ddof=1)
    if sd == 0 or r.size < 2:
        return float("nan")
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def cagr(returns: np.ndarray) -> float:
    """Compound annual growth rate from a daily simple-return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    growth = np.prod(1.0 + r)
    yrs = r.size / TRADING_DAYS_PER_YEAR
    if yrs <= 0 or growth <= 0:
        return float("nan")
    return float(growth ** (1.0 / yrs) - 1.0)


def max_drawdown(returns: np.ndarray) -> float:
    """Worst peak-to-trough drawdown of the equity curve (a negative number)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def hac_tstat(returns: np.ndarray) -> float:
    """Newey-West HAC t-stat that the mean daily return differs from zero."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 10:
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


def block_bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_boot: int = 1000,
    block: int = 21,
    seed: int = 302,
    rf_daily: float = 0.0,
) -> tuple[float, float]:
    """Circular block-bootstrap 95% CI for the annualised Sharpe.

    Block resampling preserves the volatility clustering that i.i.d. resampling destroys.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        out[b] = sharpe(r[idx[:n]], rf_daily=rf_daily)
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return (float(lo), float(hi))


def summarize(returns: np.ndarray, rf_daily: float = 0.0) -> dict:
    """Headline annualised statistics for a daily-return stream."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    return {
        "n_days": int(r.size),
        "cagr": cagr(r),
        "sharpe": sharpe(r, rf_daily=rf_daily),
        "vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if r.size > 1 else float("nan"),
        "max_dd": max_drawdown(r),
        "hac_t": hac_tstat(r - rf_daily),
    }

"""The Linear-Regression-Channel detector, its timing rules, and the honest arbiters.

The folk recipe (TradingView's "Linear Regression Channel", every charting platform):
fit a straight line to the last *N* closes by ordinary least squares; the **slope** of that
line is the trend. When the slope points **up**, you're in an uptrend — be long; when it
points **down**, step aside (or go short). The pitch is that the regression slope is a
*smoother, leading* read on the trend than a lagging moving average — so the claim worth
testing is sharp: **does the regression slope beat a moving average?**

We implement three things and pit them against each other on the same tape:

1. **The indicator** — :func:`rolling_slope`, the OLS slope of log-price over a rolling
   window (annualised, so it reads as a trend rate). The classic LRC.
2. **The timing rules** — :func:`slope_signal` (long when slope > 0, else flat; an
   optional long/short variant) and the benchmark :func:`sma_signal` (long when price is
   above its *N*-day SMA — the canonical trend filter the LRC claims to improve on).
3. **The honest arbiters** — every rule is converted to a daily position, lagged **one
   bar** (signal known at the close of *t* earns the return of *t+1*), charged one-way
   costs × NAV turnover, and raced **excess-of-buy-and-hold** so "it beat the market" is
   actually tested, not "it made money in a bull market." The Signal axis is a HAC
   (Newey-West) *t* on the daily *active* return (strategy minus buy-and-hold), plus a
   sign-block permutation placebo. Buy-and-hold's own excess return is the benchmark the
   *t* is measured against — the race is excess-vs-excess.

No look-ahead: the slope/SMA use closes up to and including *t*; the position they imply is
applied to the return from *t* to *t+1* via a single ``shift(1)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# The indicator — rolling OLS slope of log-price
# ---------------------------------------------------------------------------
def rolling_slope(close: pd.Series, window: int = 60, annualise: bool = True) -> pd.Series:
    """OLS slope of log(close) regressed on a 0..window-1 time index, rolling.

    This is the slope of the Linear Regression Channel's centre line. With log-price on
    the y-axis the slope is a per-bar continuously-compounded growth rate; multiplying by
    ``TRADING_DAYS`` (``annualise=True``) reads it as an annualised trend rate. Valid from
    bar ``window-1`` onward (earlier rows are NaN).

    Closed-form OLS slope for an evenly-spaced x = 0..n-1:
        slope = sum((x-xbar)(y-ybar)) / sum((x-xbar)^2),
    and sum((x-xbar)^2) = n(n^2-1)/12 is a constant, so the rolling computation is exact
    and fast (no per-window lstsq).
    """
    y = np.log(close.to_numpy(dtype=float))
    n = int(window)
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    denom = float(x @ x)  # n(n^2-1)/12

    out = np.full(y.size, np.nan)
    for i in range(n - 1, y.size):
        yw = y[i - n + 1 : i + 1]
        out[i] = float(x @ (yw - yw.mean())) / denom
    s = pd.Series(out, index=close.index, name="slope")
    return s * TRADING_DAYS if annualise else s


def rolling_sma(close: pd.Series, window: int = 60) -> pd.Series:
    """Simple moving average over ``window`` closes (the benchmark trend filter)."""
    return close.rolling(window, min_periods=window).mean()


# ---------------------------------------------------------------------------
# Timing rules -> daily positions (already lag-aware via the caller)
# ---------------------------------------------------------------------------
def slope_signal(close: pd.Series, window: int = 60, mode: str = "long_flat") -> pd.Series:
    """Position from the regression slope.

    ``mode="long_flat"`` → 1.0 when slope > 0 else 0.0 (long/flat — the usual deployment).
    ``mode="long_short"`` → +1.0 when slope > 0 else -1.0 (the symmetric variant).

    The position is the *desired exposure at the close of t*; the caller lags it one bar.
    """
    sl = rolling_slope(close, window=window)
    if mode == "long_flat":
        pos = (sl > 0).astype(float)
    elif mode == "long_short":
        pos = np.sign(sl).astype(float)
    else:
        raise ValueError(f"mode must be 'long_flat' or 'long_short', got {mode!r}")
    pos[sl.isna()] = np.nan
    return pos.rename("pos")


def sma_signal(close: pd.Series, window: int = 60, mode: str = "long_flat") -> pd.Series:
    """Position from the price-vs-SMA trend filter (the benchmark the LRC claims to beat)."""
    sma = rolling_sma(close, window=window)
    above = close > sma
    if mode == "long_flat":
        pos = above.astype(float)
    elif mode == "long_short":
        pos = np.where(above, 1.0, -1.0)
        pos = pd.Series(pos, index=close.index)
    else:
        raise ValueError(f"mode must be 'long_flat' or 'long_short', got {mode!r}")
    pos = pd.Series(np.asarray(pos, dtype=float), index=close.index)
    pos[sma.isna()] = np.nan
    return pos.rename("pos")


# ---------------------------------------------------------------------------
# Backtest engine — one execution lag, one-way costs × NAV turnover
# ---------------------------------------------------------------------------
def backtest(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 1.0,
    borrow_bps_yr: float = 50.0,
    rf_annual: float = 0.0,
) -> pd.DataFrame:
    """Daily P&L for a position series on a price tape; returns gross/net/excess columns.

    Convention (the desk's one documented lag): the position known at the close of *t*
    earns the asset return from *t* to *t+1* — a single ``shift(1)`` on the position.

    Costs: one-way ``cost_bps`` × |Δposition| (turnover as a fraction of NAV) charged on the
    bar the position changes. Shorts pay ``borrow_bps_yr`` financing per day held short.
    The risk-free rate ``rf_annual`` is the cash return earned when flat (and the excess
    benchmark) — applied **once**.

    Columns: ``asset_ret`` (buy-and-hold daily), ``gross``, ``net`` (strategy daily net),
    ``bh_excess`` (buy-and-hold minus rf), ``strat_excess`` (strategy net minus rf),
    ``active`` (strat_excess minus bh_excess — the head-to-head race).
    """
    ret = close.pct_change().fillna(0.0)
    pos = position.shift(1)  # the one and only execution lag
    valid = pos.notna()
    pos = pos.fillna(0.0)

    rf_daily = rf_annual / TRADING_DAYS
    borrow_daily = borrow_bps_yr * 1e-4 / TRADING_DAYS

    # Long/flat book: invested fraction earns pos*ret, the un-invested fraction earns cash.
    # (pos in {0,1} for long_flat; for long_short pos in {-1,+1} so the cash term is 0.)
    uninvested = (1.0 - pos.clip(lower=0.0, upper=1.0))
    gross = pos * ret + uninvested * rf_daily

    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * cost_bps * 1e-4
    borrow = np.clip(-pos, 0.0, None) * borrow_daily  # only the short fraction pays borrow
    net = gross - cost - borrow

    bh_excess = ret - rf_daily
    strat_excess = net - rf_daily
    active = strat_excess - bh_excess

    df = pd.DataFrame(
        {
            "asset_ret": ret,
            "pos": pos,
            "turnover": turnover,
            "gross": gross,
            "net": net,
            "bh_excess": bh_excess,
            "strat_excess": strat_excess,
            "active": active,
        }
    )
    return df.loc[valid]


def sharpe(daily: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualised Sharpe of a daily return series (mean/std × sqrt(periods))."""
    r = daily.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods))


def hac_t(daily: pd.Series) -> float:
    """Newey-West (HAC) t-stat that the mean of ``daily`` differs from zero.

    Lag length is the Newey-West rule of thumb, floor(4(n/100)^(2/9)). Used on the
    **active** (strategy minus buy-and-hold) daily return — the inference-bar number.
    """
    r = daily.to_numpy(dtype=float)
    r = r[np.isfinite(r)]
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


# ---------------------------------------------------------------------------
# Permutation placebo — block-shuffle the positions, re-race
# ---------------------------------------------------------------------------
def permutation_pvalue(
    close: pd.Series,
    position: pd.Series,
    n_draws: int = 2000,
    block: int = 21,
    cost_bps: float = 1.0,
    seed: int = 439,
    metric: str = "active_mean",
) -> dict:
    """Block-permutation placebo for a timing rule's edge over buy-and-hold.

    The observed statistic is the mean daily **active** return (strategy net minus
    buy-and-hold). The null shuffles the position series in contiguous **blocks** (so the
    timing's own persistence/turnover profile is preserved but its *alignment* with returns
    is destroyed), re-runs the backtest, and recomputes the statistic. The p-value is the
    share of shuffled books whose active mean ``>=`` the observed one.

    Block shuffling (not i.i.d.) keeps the autocorrelation the HAC t respects; an i.i.d.
    shuffle would understate the null's spread.
    """
    rng = np.random.default_rng(seed)
    base = backtest(close, position, cost_bps=cost_bps)
    obs = float(base["active"].mean())

    pos = position.dropna()
    idx = pos.index
    vals = pos.to_numpy(dtype=float)
    n = vals.size
    n_blocks = int(np.ceil(n / block))

    ge = 0
    draws = np.empty(n_draws)
    for d in range(n_draws):
        order = rng.permutation(n_blocks)
        chunks = [vals[b * block : (b + 1) * block] for b in order]
        shuffled = np.concatenate(chunks)[:n]
        sp = pd.Series(shuffled, index=idx)
        bt = backtest(close, sp.reindex(close.index), cost_bps=cost_bps)
        m = float(bt["active"].mean())
        draws[d] = m
        if m >= obs:
            ge += 1
    return {
        "obs": obs,
        "p_value": (ge + 1) / (n_draws + 1),
        "draws": draws,
        "metric": metric,
    }


# ---------------------------------------------------------------------------
# One-shot summary for a single rule on a single tape
# ---------------------------------------------------------------------------
def evaluate(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 1.0,
    label: str = "rule",
) -> dict:
    """Headline numbers for one timing rule on one tape (net, excess-vs-excess).

    Returns Sharpe of the net strategy excess, Sharpe of buy-and-hold excess, the gap,
    the HAC t on the daily active return, time-in-market, annual turnover (round-trips),
    and net/gross CAGR.
    """
    bt = backtest(close, position, cost_bps=cost_bps)
    n = len(bt)
    yrs = n / TRADING_DAYS

    def cagr(daily):
        g = float((1.0 + daily).prod())
        return g ** (1.0 / yrs) - 1.0 if yrs > 0 and g > 0 else float("nan")

    return {
        "label": label,
        "n_days": int(n),
        "years": float(yrs),
        "sharpe_strat": sharpe(bt["strat_excess"]),
        "sharpe_bh": sharpe(bt["bh_excess"]),
        "sharpe_gap": sharpe(bt["strat_excess"]) - sharpe(bt["bh_excess"]),
        "active_t": hac_t(bt["active"]),
        "active_mean_bps": float(bt["active"].mean() * 1e4),
        "time_in_mkt": float((bt["pos"] > 0).mean()),
        "turnover_yr": float(bt["turnover"].sum() / yrs) if yrs > 0 else float("nan"),
        "cagr_net": cagr(bt["net"]),
        "cagr_gross": cagr(bt["gross"]),
        "cagr_bh": cagr(bt["asset_ret"]),
    }


# ---------------------------------------------------------------------------
# Orchestrator — slope vs SMA vs buy-and-hold on a tape (or a panel)
# ---------------------------------------------------------------------------
def run_experiment(
    close: pd.Series,
    window: int = 60,
    cost_bps: float = 1.0,
    mode: str = "long_flat",
    do_permutation: bool = False,
    n_draws: int = 2000,
) -> dict:
    """Race the regression-slope rule against the SMA rule and buy-and-hold on one tape.

    Returns a dict with the slope-rule eval, the SMA-rule eval, and (optionally) the
    block-permutation p-value for the slope rule's active edge over buy-and-hold.
    """
    sl_pos = slope_signal(close, window=window, mode=mode)
    sma_pos = sma_signal(close, window=window, mode=mode)

    out = {
        "window": window,
        "cost_bps": cost_bps,
        "mode": mode,
        "slope": evaluate(close, sl_pos, cost_bps=cost_bps, label="slope"),
        "sma": evaluate(close, sma_pos, cost_bps=cost_bps, label="sma"),
    }
    if do_permutation:
        out["slope_perm"] = permutation_pvalue(
            close, sl_pos, n_draws=n_draws, cost_bps=cost_bps
        )
    return out

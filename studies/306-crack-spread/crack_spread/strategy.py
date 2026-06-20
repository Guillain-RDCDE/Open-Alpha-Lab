"""The strategy and its honest controls — Study 306 (Crack-Spread).

The folklore: a **fat / rising crack spread predicts refiner stocks**, so you can *time*
a refiner basket (VLO/MPC/PSX) from the margin. We implement the two natural timing rules
and grade them by the inference bar:

1. ``level`` timer — be long the basket when the crack spread sits above its trailing
   median (a "margins are fat" regime filter), else in cash.
2. ``momentum`` timer — be long when the crack *rose* over the past ``lookback`` days
   (a "margins are improving" rule), else in cash.

The decisive comparison is **excess-of-cash vs excess-of-cash**: a part-time-in-cash timer
raced against always-long buy-and-hold must be compared on excess returns, or the cash
drag manufactures a verdict. We also run the **predictive regression** that the whole claim
rests on — does *yesterday's* crack change forecast *today's* basket return? — with a HAC
(Newey-West) *t* on the slope, the number that decides REAL vs WEAK/NONE.

Look-ahead: the signal is known at the **close of bar t** (the crack level/change through
*t*) and earns the basket return of **t+1** — exactly one ``shift``, applied once. Costs
are charged one-way × NAV on each change of position (a flat→long or long→flat switch is
one one-way trade).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Signals (known at the close of bar t)
# ---------------------------------------------------------------------------
def level_signal(crack: pd.Series, window: int = 252) -> pd.Series:
    """Long when the crack level is above its trailing-``window`` median, else flat.

    The median through *t* is causal (uses only data up to and including *t*). Returns a
    0/1 position Series stamped on bar *t* (the engine applies the one-day execution lag).
    """
    med = crack.rolling(window, min_periods=window // 2).median()
    return (crack > med).astype(float).fillna(0.0)


def momentum_signal(crack: pd.Series, lookback: int = 20) -> pd.Series:
    """Long when the crack rose over the past ``lookback`` days, else flat.

    ``crack_t > crack_{t-lookback}`` — a "margins are improving" rule, causal at *t*.
    """
    return (crack > crack.shift(lookback)).astype(float).fillna(0.0)


# ---------------------------------------------------------------------------
# Backtest engine — one execution lag, costs one-way × NAV
# ---------------------------------------------------------------------------
def book_returns(
    position: pd.Series,
    stock_ret: pd.Series,
    cost_bps: float = 0.0,
) -> pd.Series:
    """Strategy daily *log* returns from a 0/1 ``position`` known at the close of *t*.

    Convention (one lag, applied once here): the position chosen at the close of *t* earns
    the basket return of *t+1*, so ``position`` is shifted forward by one bar. Costs are
    one-way × NAV: ``cost_bps`` basis points are deducted on each bar where the *traded*
    position changes (a 0→1 or 1→0 switch costs one one-way fee).
    """
    pos = position.shift(1).fillna(0.0)            # signal at close t → return of t+1
    gross = pos * stock_ret
    turn = pos.diff().abs().fillna(pos.abs())      # |Δposition| = one-way turnover
    cost = turn * (cost_bps * 1e-4)
    return gross - cost


def buy_and_hold(stock_ret: pd.Series) -> pd.Series:
    """Always-long daily log returns (the benchmark the timer must beat)."""
    return stock_ret.copy()


# ---------------------------------------------------------------------------
# Honest stats
# ---------------------------------------------------------------------------
def _hac_lags(n: int) -> int:
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC *t*-stat that the mean of ``x`` differs from zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = _hac_lags(n)
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def _regress_on_dcrack(frame: pd.DataFrame, lag: int) -> dict:
    """Regress ``stock_ret_t`` on the (standardised) crack change at lag ``lag``.

    ``lag=1`` → yesterday's change (predictive); ``lag=0`` → today's change (coincident).
    Returns the slope (per 1-σ of crack change), its HAC (Newey-West) *t*, R², and n.
    """
    x = frame["crack"].diff().shift(lag)
    y = frame["stock_ret"]
    d = pd.concat([x, y], axis=1).dropna()
    d.columns = ["x", "y"]
    n = len(d)
    if n < 6:
        return {"slope": float("nan"), "tstat": float("nan"), "r2": float("nan"), "n": n}
    xv = ((d["x"] - d["x"].mean()) / (d["x"].std() + 1e-12)).to_numpy()
    yv = d["y"].to_numpy()
    xc = xv - xv.mean()
    beta = float(np.cov(xv, yv, ddof=1)[0, 1] / np.var(xv, ddof=1))
    resid = yv - (yv.mean() + beta * xc)
    score = xc * resid                          # HAC t on the slope via the score
    sxx = float(np.sum(xc ** 2))
    lags = _hac_lags(n)
    s = float(score @ score)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(score[k:] @ score[:-k])
    se = np.sqrt(s) / sxx if sxx > 0 else float("nan")
    tstat = beta / se if se and np.isfinite(se) and se > 0 else float("nan")
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": beta, "tstat": float(tstat), "r2": r2, "n": n}


def predictive_regression(frame: pd.DataFrame) -> dict:
    """Does *yesterday's* crack change forecast *today's* basket return? (lag 1)

    The slope's HAC *t* is the inference-bar number for the Signal axis (predictive, not
    coincident) — REAL needs |t| ≥ 2 here on the real tape.
    """
    return _regress_on_dcrack(frame, lag=1)


def contemporaneous_regression(frame: pd.DataFrame) -> dict:
    """The *coincident* relation: ``stock_ret_t`` on ``Δcrack_t`` (same day, lag 0).

    The relation the folklore confuses for predictability — large and significant, but NOT
    tradable (you don't know today's crack before today's close). The foil, not the edge.
    """
    return _regress_on_dcrack(frame, lag=0)


def sharpe(daily_log_ret: pd.Series) -> float:
    """Annualised Sharpe of a daily log-return series (excess assumed already removed)."""
    r = daily_log_ret.dropna().to_numpy()
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def annualised(daily_log_ret: pd.Series) -> float:
    """Annualised geometric return (%/yr) from daily log returns."""
    r = daily_log_ret.dropna()
    if r.empty:
        return float("nan")
    return float((np.exp(r.mean() * TRADING_DAYS_PER_YEAR) - 1.0) * 100.0)


def block_bootstrap_ci(
    daily_log_ret: pd.Series,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 306,
    stat=sharpe,
) -> tuple[float, float]:
    """Circular block-bootstrap 95% CI for a per-series ``stat`` (default Sharpe).

    A 21-day (≈1 month) circular block preserves the autocorrelation the i.i.d. bootstrap
    would destroy.
    """
    r = daily_log_ret.dropna().to_numpy()
    n = r.size
    if n < block * 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        stats[b] = stat(pd.Series(r[idx[:n]]))
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return (float(lo), float(hi))


def summarize_timer(
    frame: pd.DataFrame,
    position: pd.Series,
    cost_bps: float = 0.0,
) -> dict:
    """Headline stats for a timer vs always-long buy-and-hold, raced excess-vs-excess.

    Because the timer parks in cash part of the time, the fair race is the timer's
    return over what cash would have earned vs buy-and-hold's return over the same cash.
    With log returns and a flat risk-free assumption here (cash earns 0), the excess is
    the series itself — but we make the comparison explicit by reporting the **active
    return** (timer − buy-and-hold) and its HAC *t*: a positive, significant active return
    is the only thing that certifies the timer adds value over just holding the basket.
    """
    timer = book_returns(position, frame["stock_ret"], cost_bps=cost_bps)
    bh = buy_and_hold(frame["stock_ret"])
    active = (timer - bh).dropna()
    pos = position.shift(1).fillna(0.0)
    return {
        "timer_ann": annualised(timer),
        "bh_ann": annualised(bh),
        "timer_sharpe": sharpe(timer),
        "bh_sharpe": sharpe(bh),
        "active_ann": annualised(active),
        "active_t": hac_tstat(active.to_numpy()),
        "pct_in_market": float(pos.mean() * 100.0),
        "n_switches": int(pos.diff().abs().fillna(0).sum()),
        "n_days": int(len(frame)),
    }

"""The cross-sectional engine and the survivor-vs-full delta — Study 345 (Survivorship-Bias).

We run ONE strategy — a simple cross-sectional book that ranks names on a signal each
month, goes long the top fraction and short the bottom fraction, equal-weight, rebalanced
monthly — on two views of the SAME panel:

- **FULL** — every name, including the ones that delist. A short-momentum / long-quality
  rule that *avoids* the doomed names is rewarded; a naive long rule eats the wipe-outs.
- **SURVIVORS** — only the names still alive at the end (a current-membership universe
  projected back). The losers were deleted before the backtest saw them.

The strategy's gross/net Sharpe and HAC-t on each view, and the **delta between them**, is
the measurement. A positive delta means survivorship *inflated* the read; a sign flip means
it *manufactured* or *inverted* the edge.

Conventions (the audit's honesty checks):
- **One execution lag.** The signal is the trailing ``lookback``-month total return known
  *at the close of month t*; it earns the return of month ``t+1`` — a single ``shift(1)``,
  applied once, stated here. A delisting in ``t+1`` is therefore felt by a position taken
  on ``t``'s signal (no peeking past the death).
- **Costs are one-way × NAV.** Turnover is ``0.5 * sum|w_t - w_{t-1}|`` (one-way), charged
  at ``cost_bps`` per unit traded; the long-short book is dollar-neutral and the short leg
  pays ``borrow_bps`` annual financing on its gross.
- **Excess-vs-excess.** A dollar-neutral long-short book is self-financing; its return *is*
  the spread, so no cash leg to net out. The long-only variant is compared excess-of-cash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Signal & weights
# ---------------------------------------------------------------------------
def momentum_signal(returns: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """Trailing ``lookback``-month cumulative return, computed from data up to month t.

    NaN cells (a name not yet listed or already delisted) propagate to NaN signal, so a
    dead name simply drops out of the ranking that month — exactly as it would in reality.
    """
    logr = np.log1p(returns)
    cum = logr.rolling(lookback, min_periods=lookback).sum()
    return np.expm1(cum)


def cross_section_weights(
    signal_row: pd.Series, frac: float = 0.2, long_short: bool = True
) -> pd.Series:
    """Equal-weight long top ``frac`` / short bottom ``frac`` of the *available* names.

    Only names with a finite signal that month are ranked (delisted names are excluded).
    Long-short returns a dollar-neutral book (+ on longs sum to 1, − on shorts sum to 1, so
    gross exposure 2, net 0). Long-only returns weights summing to 1 over the top fraction.
    """
    s = signal_row.dropna()
    n = len(s)
    out = pd.Series(0.0, index=signal_row.index)
    if n < 5:
        return out
    k = max(1, int(round(frac * n)))
    order = s.sort_values()
    losers = order.index[:k]
    winners = order.index[-k:]
    out[winners] = 1.0 / k
    if long_short:
        out[losers] = -1.0 / k
    return out


# ---------------------------------------------------------------------------
# Backtest engine (one lag, one-way costs × NAV, borrow on shorts)
# ---------------------------------------------------------------------------
def backtest(
    returns: pd.DataFrame,
    lookback: int = 12,
    frac: float = 0.2,
    long_short: bool = True,
    direction: int = -1,
    cost_bps: float = 0.0,
    borrow_bps: float = 0.0,
) -> pd.Series:
    """Net monthly returns of the cross-sectional book on ``returns``.

    ``direction``: ``-1`` is the *contrarian* "buy the past losers" rule (the textbook
    survivorship trap — a loser that recovers looks like an edge, a loser that delists is
    invisible on the survivors tape); ``+1`` is the momentum "buy the past winners" rule.

    The signal at month t (trailing ``lookback`` return) sets weights HELD over month t+1
    (one shift). Turnover is one-way × NAV; the short leg pays ``borrow_bps``/yr financing.
    A delisting (the −80% wipe in the synthetic tape, a real bankruptcy on the live tape)
    is borne by whatever position the t-signal put on it — no look-ahead around the death.
    """
    sig = direction * momentum_signal(returns, lookback=lookback)
    dates = returns.index
    w_prev = pd.Series(0.0, index=returns.columns)
    out = pd.Series(np.nan, index=dates)
    monthly_borrow = borrow_bps * 1e-4 / MONTHS_PER_YEAR

    for i in range(len(dates)):
        if i == 0:
            continue
        # Signal known at the close of t-1 (one lag); weights held over month t.
        target = cross_section_weights(sig.iloc[i - 1], frac=frac, long_short=long_short)
        target = target.reindex(returns.columns).fillna(0.0)

        # Gross return of the held book over month t. A delisted name has NaN return: if we
        # held it (we shouldn't, the signal was NaN → weight 0) treat as 0; otherwise the
        # wipe-out month is a finite (large negative) return and IS felt.
        r_t = returns.iloc[i]
        contrib = (target * r_t).where(target != 0.0, 0.0)
        gross = float(contrib.fillna(0.0).sum())

        turnover = 0.5 * float((target - w_prev).abs().sum())   # one-way
        cost = turnover * cost_bps * 1e-4
        borrow = float(target[target < 0].abs().sum()) * monthly_borrow
        out.iloc[i] = gross - cost - borrow
        w_prev = target

    return out


# ---------------------------------------------------------------------------
# Metrics & inference
# ---------------------------------------------------------------------------
def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def summarize(monthly_ret: pd.Series, periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Headline metrics for one monthly return series (Sharpe is excess-of-zero here)."""
    r = monthly_ret.dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in ("n", "cagr", "sharpe", "max_dd", "mean_ann")}
    total = float((1.0 + r).prod())
    n_years = n / periods_per_year
    cagr = total ** (1.0 / n_years) - 1.0 if (n_years > 0 and total > 0) else float("nan")
    ann_mean = r.mean() * periods_per_year
    ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_mean / ann_vol if ann_vol > 0 else float("nan")
    equity = (1.0 + r).cumprod().to_numpy()
    return {
        "n": int(n),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_dd": _max_drawdown(equity),
        "mean_ann": float(ann_mean),
    }


def hac_tstat(r: pd.Series) -> dict:
    """Newey-West HAC t-stat that the mean of ``r`` differs from zero."""
    x = r.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 3:
        return {"mean": float("nan"), "tstat": float("nan"), "n": n}
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wk * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return {
        "mean": float(mu),
        "tstat": float(mu / se) if se > 0 else float("nan"),
        "n": n,
        "mean_ann": float(mu * MONTHS_PER_YEAR),
    }


def block_bootstrap_ci(
    r: pd.Series, block: int = 6, n_boot: int = 2000, seed: int = 345, alpha: float = 0.05
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of ``r`` (preserves autocorrelation)."""
    x = r.dropna().to_numpy(dtype=float)
    n = x.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = x[idx][:n].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# The study's spine — run the SAME strategy on FULL vs SURVIVORS
# ---------------------------------------------------------------------------
def compare_views(
    returns_full: pd.DataFrame,
    returns_survivors: pd.DataFrame,
    lookback: int = 12,
    frac: float = 0.2,
    long_short: bool = False,
    direction: int = -1,
    cost_bps: float = 0.0,
    borrow_bps: float = 0.0,
    seed: int = 345,
) -> dict:
    """Run the strategy on both views and report the survivorship delta.

    Returns a bundle with the FULL and SURVIVORS net return series, their summaries and
    HAC tests, the bootstrap CI on each, and:
      - ``delta_sharpe``   = Sharpe(survivors) − Sharpe(full)
      - ``delta_mean_ann`` = annualised mean(survivors) − mean(full)
      - ``sign_flip``      = True if the FULL and SURVIVORS mean returns have opposite sign
                             (the bias *manufactured* or *inverted* the edge).
    """
    rf = backtest(returns_full, lookback=lookback, frac=frac, long_short=long_short,
                  direction=direction, cost_bps=cost_bps, borrow_bps=borrow_bps)
    rs = backtest(returns_survivors, lookback=lookback, frac=frac, long_short=long_short,
                  direction=direction, cost_bps=cost_bps, borrow_bps=borrow_bps)

    sf, ss = summarize(rf), summarize(rs)
    tf, ts = hac_tstat(rf), hac_tstat(rs)
    bf = block_bootstrap_ci(rf, seed=seed)
    bs = block_bootstrap_ci(rs, seed=seed)

    sign_flip = bool(np.sign(tf["mean"]) != np.sign(ts["mean"])
                     and abs(tf["mean"]) > 0 and abs(ts["mean"]) > 0)

    return {
        "full": rf, "survivors": rs,
        "summary_full": sf, "summary_survivors": ss,
        "test_full": tf, "test_survivors": ts,
        "boot_full": bf, "boot_survivors": bs,
        "delta_sharpe": float(ss["sharpe"] - sf["sharpe"]),
        "delta_mean_ann": float(ts["mean_ann"] - tf["mean_ann"]),
        "sign_flip": sign_flip,
    }

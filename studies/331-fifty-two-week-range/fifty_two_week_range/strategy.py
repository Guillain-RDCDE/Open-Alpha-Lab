"""The strategy and its honest controls — Study 331 (Fifty-Two-Week-Range).

The folk claim (George & Hwang 2004, *nearness to the 52-week high*) generalised: the
**range position**

    pos(t) = (close(t) - low_52w(t)) / (high_52w(t) - low_52w(t))   ∈ [0, 1]

anchors on *both* the 52-week high and low at once. Believers say it is a *better*
momentum read than distance-to-the-high alone, because a stock at 0.95 of a wide range
that has also pulled clear of its low is "more confirmed" than one merely close to a
high it keeps brushing.

This study is therefore not a re-run of study 236 (high ratio ``close/high_52w``) or
study 202 (low proximity). It is a **horse race**: range-position vs the high ratio,
asking whether the second anchor (the low) earns its keep. Two tests share one
forward-return engine.

1. **Quintile spread of each signal** — rank the cross-section by each signal, form
   equal-weight quintiles, measure the Q5−Q1 forward-return spread and its HAC *t*.
   The honest comparison is **spread vs spread** on the *same* dates: if range-position
   beats the high ratio, its spread is larger with a significant *difference*.

2. **Cross-sectional spanning regression** — regress forward returns on range-position
   *controlling for* the high ratio (and vice-versa). The incremental coefficient on
   range-position, with its HAC *t*, is the cleanest answer to "does the low add
   anything the high doesn't already say?" A near-zero incremental coefficient means
   the two signals are redundant and the extra anchor is decoration.

Costs: turnover is one-way × NAV; a per-period round-trip cost (``cost_bps``) is charged
on the long-short book, and a long-only top-decile variant pays the same one-way cost on
its turnover. Shorts in the L/S spread pay borrow via the round-trip charge.

One execution lag, documented: the signal is formed on closes up to *t*; the forward
return is ``log(close[t+hold] / close[t])`` — it starts at *t*'s close and is realised
``hold`` bars later. No second silent shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------
def rolling_high_low(panel: pd.DataFrame, window: int = 252):
    """Trailing 52-week high and low of the close panel (inclusive of *t*)."""
    hi = panel.rolling(window, min_periods=window).max()
    lo = panel.rolling(window, min_periods=window).min()
    return hi, lo


def range_position(panel: pd.DataFrame, window: int = 252, eps: float = 1e-9):
    """52-week **range position**: (close - low) / (high - low) ∈ [0, 1].

    The George-Hwang *nearness* measure that anchors on both endpoints. NaN during the
    ``window``-bar warm-up; clipped to [0, 1] for the degenerate zero-range case.
    """
    hi, lo = rolling_high_low(panel, window)
    pos = (panel - lo) / (hi - lo + eps)
    return pos.clip(0.0, 1.0)


def high_ratio(panel: pd.DataFrame, window: int = 252, eps: float = 1e-9):
    """The study-236 / George-Hwang **high-only** ratio: close / high_52w ∈ (0, 1].

    The competitor in the horse race — it ignores the low entirely.
    """
    hi, _ = rolling_high_low(panel, window)
    return (panel / (hi + eps)).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------
def forward_returns(panel: pd.DataFrame, hold_days: int = 5) -> pd.DataFrame:
    """Forward log-return over ``hold_days`` for each stock: log(close[t+h]/close[t]).

    No look-ahead: the return starting at bar *t* is only realised at *t + hold_days*;
    the last ``hold_days`` rows are NaN. This is the one (and only) execution lag.
    """
    return panel.apply(lambda s: np.log(s.shift(-hold_days) / s))


# ---------------------------------------------------------------------------
# Quintile spread backtest (one signal)
# ---------------------------------------------------------------------------
def quintile_spread(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    n_quintiles: int = 5,
    rebalance_freq: int = 5,
    cost_bps: float = 0.0,
) -> dict[str, pd.Series]:
    """Cross-sectional quintile sort on ``signal``; Q5−Q1 spread per rebalancing date.

    On each date stocks are ranked ascending by the signal (rank 1 = lowest), bucketed
    into ``n_quintiles``; each quintile's return is the equal-weight mean forward return
    of its members. The spread is **Q5 − Q1** (top minus bottom). A round-trip
    ``cost_bps`` is deducted from the *spread* (the L/S book, where shorts pay borrow).

    Returns a dict: ``Q1..Q{n}``, ``spread_gross``, ``spread_net``, each a Series
    indexed by the signal date.
    """
    dates = signal.index[signal.notna().all(axis=1) & fwd.notna().all(axis=1)]
    dates = dates[::rebalance_freq]

    q_lists: dict[str, list] = {f"Q{q}": [] for q in range(1, n_quintiles + 1)}
    spread_g, idx = [], []

    for t in dates:
        sx = signal.loc[t]
        fr = fwd.loc[t]
        if sx.isna().any() or fr.isna().any():
            continue
        ranks = sx.rank(method="first")
        n = len(ranks)
        q_size = n / n_quintiles
        for q in range(1, n_quintiles + 1):
            mask = (ranks > (q - 1) * q_size) & (ranks <= q * q_size)
            q_lists[f"Q{q}"].append(
                float(fr[mask].mean()) if mask.sum() else float("nan")
            )
        spread_g.append(q_lists[f"Q{n_quintiles}"][-1] - q_lists["Q1"][-1])
        idx.append(t)

    di = pd.DatetimeIndex(idx, name="date")
    out = {k: pd.Series(v, index=di, name=k) for k, v in q_lists.items()}
    out["spread_gross"] = pd.Series(spread_g, index=di, name="spread_gross")
    out["spread_net"] = out["spread_gross"] - cost_bps * 1e-4
    out["spread_net"].name = "spread_net"
    return out


# ---------------------------------------------------------------------------
# Long-only top decile vs equal-weight baseline
# ---------------------------------------------------------------------------
def long_top_decile(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    decile_frac: float = 0.20,
    rebalance_freq: int = 5,
    cost_bps: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Long the top fraction by ``signal`` vs the equal-weight basket baseline.

    Returns ``(strat_net, baseline)`` aligned Series of per-period log returns. The
    strategy pays a one-way ``cost_bps`` on each rebalance (a crude turnover proxy: a
    full reconstitution); the baseline is the cost-free equal-weight mean (the fair
    bar a long-only investor must beat).
    """
    dates = signal.index[signal.notna().all(axis=1) & fwd.notna().all(axis=1)]
    dates = dates[::rebalance_freq]

    strat, base, idx = [], [], []
    for t in dates:
        sx = signal.loc[t]
        fr = fwd.loc[t]
        if sx.isna().any() or fr.isna().any():
            continue
        cutoff = sx.quantile(1.0 - decile_frac)
        mask = sx >= cutoff
        if mask.sum() == 0:
            continue
        strat.append(float(fr[mask].mean()) - cost_bps * 1e-4)
        base.append(float(fr.mean()))
        idx.append(t)

    di = pd.DatetimeIndex(idx, name="date")
    return (
        pd.Series(strat, index=di, name="top_quintile_net"),
        pd.Series(base, index=di, name="baseline"),
    )


# ---------------------------------------------------------------------------
# Cross-sectional spanning regression — the discriminating test
# ---------------------------------------------------------------------------
def spanning_regression(
    target: pd.DataFrame,
    control: pd.DataFrame,
    fwd: pd.DataFrame,
    rebalance_freq: int = 5,
) -> pd.DataFrame:
    """Period-by-period cross-sectional OLS: fwd ~ 1 + z(target) + z(control).

    Both signals are cross-sectionally z-scored each date so the coefficients are
    directly comparable. The **incremental** slope on ``target`` (controlling for
    ``control``) answers: does ``target`` add forward information the ``control``
    doesn't already carry? Returns a frame with one row per rebalancing date:
    ``b_target``, ``b_control`` (Fama-MacBeth-style per-period slopes).
    """
    dates = target.index[
        target.notna().all(axis=1)
        & control.notna().all(axis=1)
        & fwd.notna().all(axis=1)
    ]
    dates = dates[::rebalance_freq]

    rows, idx = [], []
    for t in dates:
        y = fwd.loc[t].to_numpy(dtype=float)
        x1 = target.loc[t].to_numpy(dtype=float)
        x2 = control.loc[t].to_numpy(dtype=float)
        if not (np.isfinite(y).all() and np.isfinite(x1).all() and np.isfinite(x2).all()):
            continue
        z1 = _zscore(x1)
        z2 = _zscore(x2)
        if z1 is None or z2 is None:
            continue
        X = np.column_stack([np.ones_like(y), z1, z2])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        rows.append((beta[1], beta[2]))
        idx.append(t)

    return pd.DataFrame(
        rows, columns=["b_target", "b_control"],
        index=pd.DatetimeIndex(idx, name="date"),
    )


def _zscore(x: np.ndarray):
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return None
    return (x - x.mean()) / sd


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def hac_tstat(series: pd.Series) -> float:
    """Newey-West HAC *t*-stat of the mean of a return/coefficient series."""
    r = np.asarray(series.dropna(), dtype=float)
    n = r.size
    if n <= 5:
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


def summarize(ret: pd.Series, periods_per_year: float | None = None) -> dict:
    """Headline stats for a per-period return series: n, win-rate, mean (bps), ann %,
    HAC *t*. ``periods_per_year`` annualises (default inferred from the date spacing)."""
    r = np.asarray(ret.dropna(), dtype=float)
    n = r.size
    if n == 0:
        return {k: float("nan") for k in ["n", "win_rate", "mean_bps", "ann_pct", "tstat"]}
    if periods_per_year is None:
        step = ret.index.to_series().diff().median()
        step_days = step.days if hasattr(step, "days") and step.days else 7
        periods_per_year = 252.0 / max(step_days, 1)
    return {
        "n": int(n),
        "win_rate": float((r > 0).mean()),
        "mean_bps": float(r.mean() * 1e4),
        "ann_pct": float(r.mean() * periods_per_year * 100.0),
        "tstat": hac_tstat(ret),
    }


def block_bootstrap_ci(
    series: pd.Series,
    n_boot: int = 2000,
    block: int = 10,
    alpha: float = 0.05,
    seed: int = 331,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean (bps), preserving autocorrelation."""
    r = np.asarray(series.dropna(), dtype=float)
    n = r.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = r[idx[:n]].mean()
    lo = float(np.quantile(means, alpha / 2) * 1e4)
    hi = float(np.quantile(means, 1 - alpha / 2) * 1e4)
    return (lo, hi)


def diff_tstat(a: pd.Series, b: pd.Series) -> dict:
    """HAC *t* on the *difference* a − b on common dates (the spread-vs-spread race).

    Returns ``{"mean_diff_bps", "tstat", "n"}``. This is the honest way to claim one
    signal beats the other: test the paired difference, not two separate t-stats.
    """
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(df) <= 5:
        return {"mean_diff_bps": float("nan"), "tstat": float("nan"), "n": len(df)}
    d = df["a"] - df["b"]
    return {
        "mean_diff_bps": float(d.mean() * 1e4),
        "tstat": hac_tstat(d),
        "n": int(len(d)),
    }

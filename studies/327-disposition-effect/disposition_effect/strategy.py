"""The signal and its honest controls — Study 327 (Disposition-Effect).

The claim (Grinblatt & Han 2005): because disposition-prone holders sell winners too soon,
stocks with a large **unrealised capital gain** (high *overhang*) are held below fundamentals
and subsequently *out*perform; deep-underwater names under-perform. We harvest that with a
monthly cross-sectional sort on the capital-gains overhang ``g = (P - R)/P``.

We implement the comparisons that decide the verdict:

1. **Overhang quintile sort** — Q5 (deep-in-the-money) minus Q1 (underwater): the headline
   long-short hedge, equal-weighted, rebalanced monthly.
2. **Momentum control.** Overhang is mechanically correlated with 12-1 momentum (a winner is
   both high-overhang and high-momentum). The decisive test is whether the overhang hedge
   carries information *beyond* momentum — we cross-sectionally **orthogonalise** overhang to
   momentum each month and re-sort. If the residual hedge dies, the "disposition" premium was
   just momentum wearing a costume.
3. **Information coefficient (IC)** — the Spearman rank-correlation of overhang with the
   next-month return, year by year.

Costs: one-way bps × NAV, charged on the *turnover* of the long-short book (a full rebalance
of a long-short book is ~4× one-way notional). The hedge is dollar-neutral; we report gross
and net side by side and never call one the other.

No look-ahead: the overhang at month-end M (known at that close) earns the return of month
M+1. The data layer already aligns ``fwd_ret`` to M+1, so the engine applies **no further
shift** — one lag, applied once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cross-sectional helpers
# ---------------------------------------------------------------------------
def _zscore_row(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score of one month's signal (NaN-safe)."""
    s = s.astype(float)
    mu, sd = s.mean(), s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return s * 0.0
    return (s - mu) / sd


def orthogonalise(signal: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally regress ``signal`` on ``control`` each month; return the residual.

    For every month we fit ``signal = a + b*control + e`` across the firms with both values
    present and keep ``e`` (the part of the overhang that momentum does *not* explain). This
    is the control that asks: is the disposition premium anything more than momentum?
    """
    rows = {}
    for m in signal.index:
        if m not in control.index:
            continue
        s = signal.loc[m].astype(float)
        c = control.loc[m].astype(float)
        common = s.dropna().index.intersection(c.dropna().index)
        if len(common) < 5:
            continue
        x = c.loc[common].to_numpy()
        y = s.loc[common].to_numpy()
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        rows[m] = pd.Series(resid, index=common)
    out = pd.DataFrame(rows).T.reindex(columns=signal.columns)
    out.index.name = signal.index.name
    return out


# ---------------------------------------------------------------------------
# Core quintile long-short sort
# ---------------------------------------------------------------------------
def quintile_hedge(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quantiles: int = 5,
    min_firms: int = 15,
    long_high: bool = True,
) -> pd.DataFrame:
    """Monthly quantile sort of ``signal`` against ``fwd_ret`` (next-month return).

    Returns a month-indexed DataFrame with columns ``Q1..Qk`` (equal-weight bucket returns),
    ``hedge`` (top − bottom = Q5 − Q1 when ``long_high``), ``market`` (equal-weight universe),
    and ``n_firms``. Convention: high overhang → Q5 (the Grinblatt-Han long leg).
    """
    q_labels = [f"Q{i}" for i in range(1, n_quantiles + 1)]
    rows = {}
    for m in signal.index:
        if m not in fwd_ret.index:
            continue
        s = signal.loc[m].dropna()
        r = fwd_ret.loc[m].dropna()
        common = s.index.intersection(r.index)
        if len(common) < min_firms:
            continue
        s_, r_ = s.loc[common], r.loc[common]
        try:
            qbins = pd.qcut(s_.rank(method="first"), q=n_quantiles, labels=q_labels)
        except ValueError:
            continue
        means = {q: float(r_[qbins == q].mean()) for q in q_labels}
        if any(not np.isfinite(v) for v in means.values()):
            continue
        top, bot = (means[q_labels[-1]], means[q_labels[0]])
        means["hedge"] = (top - bot) if long_high else (bot - top)
        means["market"] = float(r_.mean())
        means["n_firms"] = float(len(common))
        rows[m] = means
    out = pd.DataFrame(rows).T.sort_index()
    out.index.name = "month"
    return out.astype(float)


def information_coefficient(
    signal: pd.DataFrame, fwd_ret: pd.DataFrame, min_firms: int = 15
) -> pd.Series:
    """Monthly Spearman IC: rank-correlation of ``signal`` with next-month returns."""
    from scipy.stats import spearmanr

    out = {}
    for m in signal.index:
        if m not in fwd_ret.index:
            continue
        s = signal.loc[m].dropna()
        r = fwd_ret.loc[m].dropna()
        common = s.index.intersection(r.index)
        if len(common) < min_firms:
            continue
        a, b = s.loc[common], r.loc[common]
        if a.nunique() < 2 or b.nunique() < 2:
            continue
        ic, _ = spearmanr(a, b)
        if np.isfinite(ic):
            out[m] = float(ic)
    return pd.Series(out, name="ic").sort_index()


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def apply_costs(hedge: pd.Series, cost_bps: float, turnover_per_leg: float = 2.0) -> pd.Series:
    """Net a monthly long-short hedge return for one-way trading costs × NAV.

    A dollar-neutral long-short hedge that fully rebalances each month turns over
    ``turnover_per_leg`` × NAV one-way **per leg** (default 2.0 → 4× one-way notional across
    both legs, the standard conservative assumption for a monthly quintile L/S). The drag is
    ``2 * turnover_per_leg * cost_bps * 1e-4`` per month.
    """
    drag = 2.0 * turnover_per_leg * cost_bps * 1e-4
    return hedge.astype(float) - drag


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def hac_tstat(series: pd.Series) -> float:
    """Newey-West (HAC) t-stat on the mean of a monthly series (auto lag rule)."""
    r = pd.Series(series).astype(float).dropna().to_numpy()
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, min(lags + 1, n)):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def block_bootstrap_ci(
    series: pd.Series, block: int = 6, n_boot: int = 2000,
    alpha: float = 0.05, seed: int = 327,
) -> tuple[float, float]:
    """Circular block-bootstrap CI for the mean of a monthly series.

    i.i.d. resampling would destroy the autocorrelation the HAC t-stat respects, so we
    resample contiguous blocks of length ``block`` (months) with wrap-around.
    """
    r = pd.Series(series).astype(float).dropna().to_numpy()
    n = r.size
    if n < block + 1:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        means[b] = r[idx][:n].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return (lo, hi)


def summary(series: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised stats for a monthly hedge (or IC) series, plus the HAC t-stat."""
    r = pd.Series(series).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in
                ("mean_m", "ann_ret", "ann_vol", "sharpe", "tstat", "hit_rate", "max_dd", "n")} | {"n": n}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    sharpe = (mu / sd) * np.sqrt(periods_per_year) if sd > 0 else float("nan")
    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean_m": mu,
        "ann_ret": mu * periods_per_year,
        "ann_vol": sd * np.sqrt(periods_per_year),
        "sharpe": sharpe,
        "tstat": hac_tstat(r),
        "hit_rate": float((r > 0).mean()),
        "max_dd": max_dd,
        "n": int(n),
    }

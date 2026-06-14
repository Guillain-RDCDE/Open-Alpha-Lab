"""The signal and its honest controls — Study 124 (Cash-Flow-Yield).

The claim: sorting S&P 500 stocks by **operating cash-flow yield** (OCF / market cap) and
going long the top quintile vs the bottom (or vs the equal-weight universe) generates alpha
beyond what earnings yield alone already captures.  The trade runs at annual frequency with
a one-year reporting lag.

We implement three comparisons that decide the verdict:

1. **OCF-yield quintile sort vs equal-weight universe** — the headline long-only spread.
2. **OCF-yield long-short hedge** (Q5 − Q1) — the pure cross-sectional signal.
3. **EY (earnings-yield) long-short hedge** — the control: does NI/price already do this?
   If the EY hedge and the OCF hedge are statistically identical, OCF adds no information
   over the better-known P/E inverse.

No look-ahead: signal for year Y uses fiscal-year-Y fundamentals (10-K filed by ~March of
Y+1); the forward return is the calendar-year-Y+1 return (so the signal is already stale
before the position is entered). Survivorship bias is named on every result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core quintile sort
# ---------------------------------------------------------------------------
def quintile_sort(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quintiles: int = 5,
    min_firms: int = 20,
) -> pd.DataFrame:
    """Annual quintile-sort of ``signal`` against ``fwd_ret`` (next-year return).

    Returns a DataFrame with columns ``Q1``, …, ``Q{n_quintiles}`` (the equal-weight
    mean return of each bucket), ``hedge`` (= top − bottom = Q5 − Q1 for long_high=True),
    ``market`` (the equal-weight universe return), and ``n_firms`` (median bucket count).

    Convention: **high signal → Q5** (we long high OCF yield, the value direction).
    Firms missing either signal or forward return in a year are silently dropped.
    """
    rows = {}
    q_labels = [f"Q{i}" for i in range(1, n_quintiles + 1)]
    for y in signal.index:
        if y not in fwd_ret.index:
            continue
        s = signal.loc[y].dropna()
        r = fwd_ret.loc[y].dropna()
        common = s.index.intersection(r.index)
        if len(common) < min_firms:
            continue
        s_, r_ = s.loc[common], r.loc[common]
        # Rank into quintiles (qcut on the signal).
        try:
            qbins = pd.qcut(s_, q=n_quintiles, labels=q_labels)
        except ValueError:
            continue
        means = {q: r_[qbins == q].mean() for q in q_labels}
        if any(np.isnan(v) for v in means.values()):
            continue
        means["hedge"] = means[q_labels[-1]] - means[q_labels[0]]  # Q5 − Q1
        means["market"] = r_.mean()
        means["n_firms"] = len(common)
        rows[y] = means
    out = pd.DataFrame(rows).T.sort_index()
    out.index.name = "year"
    return out.astype(float)


def hedge_vs_market(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.2,
    min_firms: int = 20,
) -> pd.DataFrame:
    """Annual top-``q`` vs equal-weight market spread.

    For each year, ranks firms by ``signal``, takes the top fraction ``q`` (high-yield),
    and records ``long`` (top-q mean), ``market`` (universe mean), and
    ``spread`` (= long − market). Returns a year-indexed DataFrame.
    """
    rows = {}
    for y in signal.index:
        if y not in fwd_ret.index:
            continue
        s = signal.loc[y].dropna()
        r = fwd_ret.loc[y].dropna()
        common = s.index.intersection(r.index)
        if len(common) < min_firms:
            continue
        s_, r_ = s.loc[common], r.loc[common]
        thresh = s_.quantile(1.0 - q)
        top_idx = s_[s_ >= thresh].index
        long_ret = r_.loc[top_idx].mean()
        mkt_ret = r_.mean()
        rows[y] = {"long": long_ret, "market": mkt_ret, "spread": long_ret - mkt_ret}
    out = pd.DataFrame(rows).T.sort_index()
    out.index.name = "year"
    return out.astype(float)


# ---------------------------------------------------------------------------
# Head-to-head: OCF yield vs earnings yield
# ---------------------------------------------------------------------------
def pairwise_ic(
    signal_a: pd.DataFrame,
    signal_b: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    min_firms: int = 20,
) -> pd.DataFrame:
    """Annual Spearman IC (information coefficient) for two competing signals.

    Returns a year × {ic_a, ic_b, ic_diff} DataFrame. A positive IC means the
    signal's cross-sectional rank correlates with next-year returns; the comparison
    tests whether OCF yield (A) has incremental IC over earnings yield (B) or vice versa.
    """
    from scipy.stats import spearmanr

    rows = {}
    for y in fwd_ret.index:
        ra_ok = y in signal_a.index
        rb_ok = y in signal_b.index
        if not ra_ok or not rb_ok:
            continue
        sa = signal_a.loc[y].dropna()
        sb = signal_b.loc[y].dropna()
        r = fwd_ret.loc[y].dropna()
        common = sa.index.intersection(sb.index).intersection(r.index)
        if len(common) < min_firms:
            continue
        sa_, sb_, r_ = sa.loc[common], sb.loc[common], r.loc[common]
        ic_a, _ = spearmanr(sa_, r_)
        ic_b, _ = spearmanr(sb_, r_)
        rows[y] = {"ic_a": ic_a, "ic_b": ic_b, "ic_diff": ic_a - ic_b}
    out = pd.DataFrame(rows).T.sort_index()
    out.index.name = "year"
    return out.astype(float)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summary(annual_series: pd.Series) -> dict:
    """Annualised stats for a yearly return (or IC) series.

    Returns mean, vol, Sharpe, HAC t-stat, hit-rate (fraction > 0), and max drawdown.
    The t-stat uses a Newey-West long-run variance with the standard lag rule
    ``floor(4*(n/100)^(2/9))`` — the same estimator as ``quantlab.analytics.mean_tstat_hac``
    but inlined here so the engine has no hard dependency on the quantlab import path during
    tests.
    """
    r = pd.Series(annual_series).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate",
                                    "max_drawdown", "n")}
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    sharpe = mu / sigma if sigma > 0 else np.nan

    # Newey-West HAC t-stat on the level series.
    e = r.to_numpy(dtype=float) - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, min(lags + 1, n)):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = mu / se if se > 0 else np.nan

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "mean": mu,
        "vol": sigma,
        "sharpe": sharpe,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": max_dd,
        "n": n,
    }

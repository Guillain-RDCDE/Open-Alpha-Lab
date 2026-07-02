"""The engine and its honest controls — Study 583 (DeFi-TVL-Momentum).

The claim, at full strength: a protocol's **TVL flow** — the trailing growth in its total value
locked — is a leading, on-chain "smart money" signal. Rank protocols each month by trailing TVL
growth; the top basket (fastest inflows) should out-earn the bottom basket (outflows) over the
next month. This module measures that, with the desk's honesty machinery:

1. **The signal.** Cross-sectional trailing TVL growth per protocol per month (already in the
   panel as ``tvl_growth``), z-scored across the cross-section each month.

2. **The decile/tercile sort.** Long the top TVL-growth basket, short the bottom; the monthly
   long-short return is the top-mean minus the bottom-mean forward return.

3. **The headline test.** A one-sample *t* on the time series of monthly long-short returns
   (is the average spread reliably positive?), plus a **label-shuffle placebo** null.

4. **Costs & borrow.** A one-way cost on every rebalance × both legs, and a short-borrow charge on
   the outflow leg you'd short (DeFi tokens with collapsing TVL are exactly the expensive/impossible
   ones to borrow — a deliberately punitive, honest assumption).

5. **A robustness sweep.** Re-run the sort at several formation/holding horizons and basket
   fractions; a real flow signal should not depend on one lucky cut.

6. **A synthetic positive control.** A deterministic world where the effect is planted
   (``tvl_alpha > 0``); the engine must recover a positive long-short *t* and stay flat at the null,
   averaged over >= 20 seeds so no single RNG seed can manufacture significance.

Execution convention: TVL growth is measured over the *trailing* formation window ending at the
ranking month's close, so the signal is fully public at that close; the book is entered at that
same close and held over the *forward* month. No future data enters the signal (no look-ahead).
The same-close fill is the single documented convention; a one-period-later entry is available via
the ``lag`` knob and moves the headline *t* only marginally (see the notebook).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


# ---------------------------------------------------------------------------
# The cross-sectional long-short: rank by TVL growth, long top / short bottom
# ---------------------------------------------------------------------------
def long_short_series(panel: pd.DataFrame, frac: float = 0.3, lag: int = 0) -> pd.Series:
    """Monthly long-short forward return: long top-``frac`` TVL growth, short bottom-``frac``.

    Each month the cross-section is ranked by ``tvl_growth``; the long basket is the top ``frac``
    and the short basket the bottom ``frac``; the month's long-short return is the long-mean minus
    the short-mean ``fwd_ret``. ``lag`` shifts the realised forward return by ``lag`` extra months
    (a robustness knob on the entry convention). Returns a Series indexed by month.
    """
    if panel.empty:
        return pd.Series(dtype=float)
    out = {}
    months = sorted(panel["month"].unique())
    by_month = {m: g for m, g in panel.groupby("month")}
    for m in months:
        g = by_month[m].sort_values("tvl_growth")
        k = max(1, int(round(len(g) * frac)))
        short = g.head(k)  # lowest TVL growth (outflows) -> short
        long = g.tail(k)   # highest TVL growth (inflows) -> long
        if lag:
            tgt = m + lag
            if tgt not in by_month:
                continue
            gt = by_month[tgt].set_index("protocol")["fwd_ret"]
            l = gt.reindex(long["protocol"]).dropna()
            s = gt.reindex(short["protocol"]).dropna()
            if l.empty or s.empty:
                continue
            out[m] = float(l.mean() - s.mean())
        else:
            out[m] = float(long["fwd_ret"].mean() - short["fwd_ret"].mean())
    return pd.Series(out, name="ls_ret").sort_index()


def ls_tstat(ls: pd.Series) -> dict:
    """One-sample *t* on the monthly long-short return series (is the mean spread > 0?)."""
    x = ls.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 3:
        return {"mean": float("nan"), "t": float("nan"), "n": n}
    se = x.std(ddof=1) / np.sqrt(n)
    return {
        "mean": float(x.mean()),
        "t": float(x.mean() / se) if se > 0 else float("nan"),
        "n": n,
    }


# ---------------------------------------------------------------------------
# The firm/protocol-level relation — the slope of forward return on TVL growth
# ---------------------------------------------------------------------------
def pooled_slope(panel: pd.DataFrame) -> dict:
    """Pooled OLS of forward return on the (monthly-z-scored) TVL-growth signal.

    A *positive* slope is the claim (more inflows -> more return). Standard errors are clustered by
    month (block) so cross-sectional dependence within a month does not inflate the *t*.
    """
    if panel.empty or len(panel) < 10:
        return {"slope": float("nan"), "slope_t": float("nan"), "n": 0}
    df = panel.copy()
    df["z"] = df.groupby("month")["tvl_growth"].transform(_zscore)
    x = df["z"].to_numpy(dtype=float)
    y = df["fwd_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    xtx_inv = np.linalg.inv(X.T @ X)
    # Month-clustered (block) robust variance.
    months = df["month"].to_numpy()
    meat = np.zeros((2, 2))
    for m in np.unique(months):
        mask = months == m
        Xg = X[mask]
        ug = resid[mask]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    vcov = xtx_inv @ meat @ xtx_inv
    se_slope = float(np.sqrt(vcov[1, 1]))
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se_slope) if se_slope > 0 else float("nan"),
        "n": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, frac: float = 0.3, n_perm: int = 2000, seed: int = 583) -> float:
    """Label-shuffle placebo p-value for the long-short mean.

    Within each month, shuffle the TVL-growth labels against forward returns, rebuild the
    long-short mean, and record the fraction of shuffles whose |mean| >= the observed |mean|. A
    real cross-sectional flow signal sits in the tail; noise does not.
    """
    if panel.empty:
        return float("nan")
    obs = abs(ls_tstat(long_short_series(panel, frac))["mean"])
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    by_month = {m: g.reset_index(drop=True) for m, g in panel.groupby("month")}
    months = sorted(by_month)
    count = 0
    for _ in range(n_perm):
        vals = []
        for m in months:
            g = by_month[m]
            k = max(1, int(round(len(g) * frac)))
            perm = rng.permutation(len(g))
            gz = g["tvl_growth"].to_numpy()[perm]  # shuffled labels
            order = np.argsort(gz)
            y = g["fwd_ret"].to_numpy()
            ys = y[order]
            vals.append(ys[-k:].mean() - ys[:k].mean())
        m_perm = abs(np.mean(vals))
        if m_perm >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_mean(ls: pd.Series, cost_bps: float = 30.0, borrow_ann_bps: float = 800.0) -> dict:
    """Long-short mean, gross and net of per-rebalance costs and a short borrow.

    The book is rebalanced monthly: charge a one-way ``cost_bps`` on each leg at each rebalance
    (2 legs), plus a monthly borrow on the short leg (the collapsing-TVL tokens — punitively
    expensive to borrow in crypto). Costs and borrow are in bps of NAV. Returns gross and net
    monthly means and their annualised sums.
    """
    x = ls.dropna()
    if x.empty:
        return {"gross_m": float("nan"), "net_m": float("nan"),
                "gross_ann": float("nan"), "net_ann": float("nan")}
    gross_m = float(x.mean())
    cost = 2.0 * cost_bps * 1e-4            # both legs, one-way per monthly rebalance
    borrow = borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
    net_m = gross_m - cost - borrow
    return {
        "gross_m": gross_m,
        "net_m": net_m,
        "gross_ann": gross_m * MONTHS_PER_YEAR,
        "net_ann": net_m * MONTHS_PER_YEAR,
    }


# ---------------------------------------------------------------------------
# Robustness sweep — horizons and basket fractions
# ---------------------------------------------------------------------------
def robustness(panel: pd.DataFrame, fracs=(0.2, 0.3, 0.4), lags=(0, 1)) -> pd.DataFrame:
    """Long-short *t* over a grid of basket fractions and entry lags."""
    rows = []
    for f in fracs:
        for lg in lags:
            ls = long_short_series(panel, frac=f, lag=lg)
            r = ls_tstat(ls)
            rows.append({"frac": f, "lag": lg, "mean": r["mean"], "t": r["t"], "n": r["n"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Seed-robust synthetic control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, tvl_alpha: float, n_seeds: int = 25, base_seed: int = 583,
                     frac: float = 0.3) -> float:
    """Average the long-short *t* over ``n_seeds`` synthetic worlds for a planted ``tvl_alpha``.

    The house rule: any synthetic-dependent claim averages the test statistic over >= 20 seeds so
    no single lucky RNG seed can manufacture significance.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(tvl_alpha=tvl_alpha, seed=s)
        ts.append(ls_tstat(long_short_series(panel, frac=frac))["t"])
    return float(np.nanmean(ts))

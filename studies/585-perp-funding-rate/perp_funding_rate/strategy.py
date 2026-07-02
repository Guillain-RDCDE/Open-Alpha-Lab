"""The engine and its honest controls — Study 585 (Perp-Funding-Rate).

The claim, at full strength (crypto-desk folklore; Coinglass/Amberdata research notes): a crypto
perpetual swap is tethered to spot by a periodic **funding rate** paid between longs and shorts.
When funding runs **hot** (large positive: longs pay shorts) the crowd is long-and-levered and a
long-squeeze / mean-reversion flush is due — so extreme funding is a **contrarian** signal for the
forward return. Symmetrically, extreme *negative* funding marks capitulated shorts and a
contrarian long. It is a positioning / mean-reversion signal, not a trend signal.

This module measures that, honestly, on a deterministic synthetic tape (funding history is not
free — see ``data.py``):

1. **The contrarian signal.** Standardize the funding rate (trailing z-score) so "extreme"
   is well-defined, then a position **short** when funding is hot (z high) and **long** when
   funding is cold (z low): ``position = -sign(z_funding)`` for the tercile-tail sort, and the
   continuous ``-z_funding`` for the firm-level slope.

2. **The forward-return relation.** OLS of forward return on standardized funding across the
   panel — the *sign* of the slope IS the claim (negative slope = contrarian effect: hot funding
   -> lower forward return).

3. **The tail long-short.** Sort periods into a *hot-funding* tail and a *cold-funding* tail;
   the contrarian trade is **long cold, short hot**. Report the spread (cold - hot), a two-sample
   (Welch) *t*, a label-shuffle placebo null, and costs + a short-side funding drag.

4. **A synthetic positive control.** A deterministic world where the contrarian effect is planted
   (``contrarian_beta < 0``); the engine must recover a negative slope / positive cold-minus-hot
   spread and stay flat at the null — averaged over >= 20 seeds (house rule).

Execution convention: funding is observed at period ``t`` (fully public at that close) and the
position is entered at that same close and held over the strictly-future forward window. No future
data enters the signal (no look-ahead). The same-bar fill is a documented convention; shifting
entry one period later leaves the sign and the stamp unchanged (the synthetic control uses the
identical convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Standardizing funding (trailing z, no look-ahead)
# ---------------------------------------------------------------------------
def trailing_z(funding: pd.Series, lookback: int = 90) -> pd.Series:
    """Trailing standardized funding: (f_t - mean_{t-lookback..t}) / std, using only past data.

    A rolling window (min 20 obs) keeps the z-score fully public at each period — "hot" and "cold"
    are defined relative to the recent past, not the whole sample.
    """
    mu = funding.rolling(lookback, min_periods=20).mean()
    sd = funding.rolling(lookback, min_periods=20).std(ddof=1)
    z = (funding - mu) / sd.replace(0.0, np.nan)
    return z.rename("z_trailing")


# ---------------------------------------------------------------------------
# The firm-level relation — the sign IS the claim
# ---------------------------------------------------------------------------
def funding_regression(panel: pd.DataFrame, z_col: str = "z_funding") -> dict:
    """OLS of forward return on standardized funding across the panel.

    Slope < 0 = the contrarian effect (hot funding -> lower forward return). Returns the slope, its
    t-stat, and the Pearson correlation between funding and forward return.
    """
    df = panel.dropna(subset=[z_col, "fwd_ret"])
    if len(df) < 4:
        return {"slope": float("nan"), "slope_t": float("nan"), "corr": float("nan"), "n": 0}
    x = df[z_col].to_numpy(dtype=float)
    y = df["fwd_ret"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, kk = X.shape
    dof = max(n - kk, 1)
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se_slope = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return {
        "slope": float(coef[1]),
        "slope_t": float(coef[1] / se_slope) if se_slope > 0 else float("nan"),
        "corr": corr,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# The tail long-short (long cold funding, short hot funding)
# ---------------------------------------------------------------------------
def tail_buckets(panel: pd.DataFrame, z_col: str = "z_funding", frac: float = 0.2) -> dict:
    """Sort by funding, return cold / hot bucket forward-return means and the contrarian spread.

    ``frac`` is the tail fraction in each bucket (0.2 = quintile tails). The contrarian claim
    predicts cold > hot (a *positive* cold-minus-hot spread: capitulated shorts bounce, crowded
    longs get flushed).

    Returns bucket means, the spread (cold - hot), bucket sizes, and the per-period forward returns
    of each bucket (for the two-sample t).
    """
    df = panel.dropna(subset=[z_col, "fwd_ret"])
    if df.empty:
        return {}
    df = df.sort_values(z_col)
    k = max(1, int(round(len(df) * frac)))
    cold = df.head(k)   # lowest funding (cold)
    hot = df.tail(k)    # highest funding (hot)
    cold_ret = cold["fwd_ret"]
    hot_ret = hot["fwd_ret"]
    return {
        "n": int(len(df)),
        "k": int(k),
        "cold_mean": float(cold_ret.mean()),
        "hot_mean": float(hot_ret.mean()),
        "spread": float(cold_ret.mean() - hot_ret.mean()),
        "cold_ret": cold_ret,
        "hot_ret": hot_ret,
    }


def long_short_tstat(buckets: dict) -> dict:
    """Two-sample (Welch) t on cold - hot forward returns.

    The contrarian trade's spread is a difference in means between the cold-funding and hot-funding
    tails, so the honest test is a Welch two-sample t. Returns the spread, the t-stat, and n.
    """
    if not buckets:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = buckets["cold_ret"].to_numpy(dtype=float)
    b = buckets["hot_ret"].to_numpy(dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return {"spread": float("nan"), "t": float("nan"), "n": na + nb}
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (a.mean() - b.mean()) / se if se > 0 else float("nan")
    return {"spread": float(a.mean() - b.mean()), "t": float(t), "n": na + nb}


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, z_col: str = "z_funding", frac: float = 0.2,
                   n_perm: int = 2000, seed: int = 585) -> float:
    """Label-shuffle placebo p-value for the cold-minus-hot spread.

    Shuffle the funding labels against forward returns ``n_perm`` times; the p-value is the
    fraction of shuffles whose |spread| >= the observed |spread|. A real relation sits in the
    tail; noise does not.
    """
    df = panel.dropna(subset=[z_col, "fwd_ret"])
    if len(df) < 10:
        return float("nan")
    obs = abs(tail_buckets(df, z_col=z_col, frac=frac)["spread"])
    rng = np.random.default_rng(seed)
    z = df[z_col].to_numpy(dtype=float)
    y = df["fwd_ret"].to_numpy(dtype=float)
    n = len(y)
    k = max(1, int(round(n * frac)))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        order = np.argsort(z[perm])
        ys = y[order]
        spread = ys[:k].mean() - ys[-k:].mean()
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + funding drag on the short
# ---------------------------------------------------------------------------
def net_spread(buckets: dict, cost_bps: float = 6.0, funding_drag_bps: float = 30.0) -> float:
    """Cold-minus-hot spread net of trading costs and the short-side funding drag.

    The contrarian book is rebalanced every funding period, so trading is expensive. We charge a
    round-trip one-way ``cost_bps`` on each of the two legs (4 crossings) plus a ``funding_drag_bps``
    penalty: shorting the *hot* leg means you *pay* funding to hold the short (funding is positive
    exactly when the trade says to short) — a deliberately punitive, per-rebalance drag.
    """
    if not buckets:
        return float("nan")
    gross = buckets["spread"]
    cost = 4.0 * cost_bps * 1e-4
    drag = funding_drag_bps * 1e-4
    return float(gross - cost - drag)


# ---------------------------------------------------------------------------
# Robustness — horizon sweep on the synthetic tape
# ---------------------------------------------------------------------------
def horizon_sweep(data_mod, contrarian_beta: float, horizons=(1, 3, 6, 9),
                  seed: int = 585) -> list[tuple]:
    """Re-run the firm-level slope-t and cold-minus-hot spread across forward horizons.

    Returns a list of ``(horizon, spread, slope_t)`` for a fixed planted ``contrarian_beta`` — a
    stable sign across horizons is the robustness the desk demands.
    """
    out = []
    for h in horizons:
        panel, _ = data_mod.synthetic_panel(contrarian_beta=contrarian_beta, horizon=h, seed=seed)
        reg = funding_regression(panel)
        b = tail_buckets(panel)
        out.append((int(h), float(b["spread"]), float(reg["slope_t"])))
    return out


# ---------------------------------------------------------------------------
# Seed-robust synthetic control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, contrarian_beta: float, n_seeds: int = 25,
                     base_seed: int = 585) -> float:
    """Average the funding-regression slope-t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean slope-t across seeds for a planted
    ``contrarian_beta`` (negative = the contrarian folklore).
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(contrarian_beta=contrarian_beta, seed=s)
        ts.append(funding_regression(panel)["slope_t"])
    return float(np.nanmean(ts))

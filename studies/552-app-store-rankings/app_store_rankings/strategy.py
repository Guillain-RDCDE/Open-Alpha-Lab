"""The engine and its honest controls — Study 552 (App-Store-Rankings).

The claim, at full strength: a consumer-tech company's **App Store ranking improvement** (its app
climbing the download charts) is an alt-data *nowcast* of demand, and if the market is slow to price
that demand, the rank-momentum leaders should **out-earn** the laggards cross-sectionally. The
tradable expression is a monthly long-short: **long the names whose apps climbed, short the names
whose apps sank**.

This module measures that with the standard alt-data cross-sectional toolkit:

1. **The signal.** ``rank_improve`` — the z-scored ranking improvement over the trailing month
   (fully known at the observation date; no look-ahead).

2. **The information coefficient (IC).** The cross-sectional **Spearman rank correlation** between
   the ranking-improvement signal and the *forward* return — the standard alt-data quality metric.
   Pooled across months, the mean IC and its *t*-stat (mean / SE over the monthly ICs) are the
   headline test. A robust IC needs |t| >= 2 on a *real* tape (which does not exist here).

3. **The long-short spread.** Sort each month's cross-section by ``rank_improve``, go long the top
   tercile / short the bottom tercile, and read the spread — reported **gross AND net** of one-way
   costs, with the **short leg paying borrow** (the sinking-app names are exactly the small/hard-to-
   borrow tail).

4. **A label-shuffle placebo null.** Shuffle the signal against forward returns; the p-value is the
   fraction of shuffles whose |mean IC| >= the observed. A real cross-sectional signal sits in the
   tail; noise does not.

5. **A multi-window robustness sweep.** Recompute the mean IC and its *t* over disjoint month
   blocks; a bankable signal keeps its sign and rough magnitude.

6. **A synthetic positive control.** A deterministic world where the effect is planted
   (``rank_alpha > 0``); the engine must recover a positive mean-IC-*t* and stay flat at the null —
   averaged over >= 20 seeds (house rule) so no single lucky RNG can manufacture significance.

Execution convention: the signal is a *trailing* ranking change, public at the observation close;
the position is entered at that same close and held over the forward month. No future data enters
the signal (no look-ahead). The same-close fill is one documented convention — a one-bar-later entry
on a monthly horizon moves the headline IC negligibly and changes no stamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Information coefficient — cross-sectional Spearman rank correlation
# ---------------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of a 1-D array (ties get the mean rank) — no scipy dependency."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        ranks = (sums / counts)[inv]
    return ranks


def spearman_ic(signal: np.ndarray, fwd_ret: np.ndarray) -> float:
    """Cross-sectional Spearman rank correlation between signal and forward return."""
    s = np.asarray(signal, dtype=float)
    r = np.asarray(fwd_ret, dtype=float)
    if len(s) < 3:
        return float("nan")
    rs, rr = _rankdata(s), _rankdata(r)
    if rs.std() == 0 or rr.std() == 0:
        return float("nan")
    return float(np.corrcoef(rs, rr)[0, 1])


def monthly_ic(history: pd.DataFrame) -> pd.Series:
    """One Spearman IC per month, indexed by month."""
    out = {}
    for m, g in history.groupby("month"):
        out[m] = spearman_ic(g["rank_improve"].to_numpy(), g["fwd_ret"].to_numpy())
    return pd.Series(out, name="ic").sort_index()


def ic_summary(history: pd.DataFrame) -> dict:
    """Pooled mean IC, its *t*-stat (mean / SE over monthly ICs), hit-rate, and n_months."""
    ics = monthly_ic(history).dropna()
    n = int(len(ics))
    if n < 2:
        return {"mean_ic": float("nan"), "t": float("nan"), "hit": float("nan"), "n_months": n}
    mean_ic = float(ics.mean())
    se = float(ics.std(ddof=1) / np.sqrt(n))
    t = mean_ic / se if se > 0 else float("nan")
    hit = float((ics > 0).mean())
    return {"mean_ic": mean_ic, "t": float(t), "hit": hit, "n_months": n}


# ---------------------------------------------------------------------------
# The long-short — gross and net of costs + borrow
# ---------------------------------------------------------------------------
def month_long_short(g: pd.DataFrame, frac: float = 1 / 3) -> float:
    """Top-tercile-minus-bottom-tercile forward-return spread for one month's cross-section."""
    df = g.sort_values("rank_improve")
    k = max(1, int(round(len(df) * frac)))
    short_leg = df.head(k)["fwd_ret"]     # apps that SANK (lowest improvement)
    long_leg = df.tail(k)["fwd_ret"]      # apps that CLIMBED (highest improvement)
    return float(long_leg.mean() - short_leg.mean())


def long_short_series(history: pd.DataFrame, frac: float = 1 / 3) -> pd.Series:
    """Monthly gross long-short spread series."""
    out = {m: month_long_short(g, frac) for m, g in history.groupby("month")}
    return pd.Series(out, name="ls").sort_index()


def long_short_summary(
    history: pd.DataFrame,
    frac: float = 1 / 3,
    cost_bps: float = 10.0,
    borrow_ann_bps: float = 100.0,
) -> dict:
    """Mean monthly long-short spread, *t*-stat, and the net figure after costs + borrow.

    Monthly rebalance: charge a round-trip one-way cost on **both** legs (entry + exit, long + short
    = 4 crossings of ``cost_bps``) each month, plus a short borrow on the sinking-app leg (annual,
    pro-rated to one month). Sinking-app names are exactly the small/illiquid/hard-to-borrow tail, so
    the borrow is deliberately punitive.
    """
    ls = long_short_series(history, frac).dropna()
    n = int(len(ls))
    if n < 2:
        return {"gross": float("nan"), "t": float("nan"), "net": float("nan"), "n_months": n}
    gross = float(ls.mean())
    se = float(ls.std(ddof=1) / np.sqrt(n))
    t = gross / se if se > 0 else float("nan")
    cost = 4.0 * cost_bps * 1e-4          # 4 one-way crossings per month
    borrow = borrow_ann_bps * 1e-4 / 12.0  # one month of borrow on the short leg
    net = gross - cost - borrow
    return {"gross": gross, "t": float(t), "net": float(net), "n_months": n}


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(history: pd.DataFrame, n_perm: int = 2000, seed: int = 552) -> float:
    """Label-shuffle placebo p-value for the pooled mean IC.

    Within each month, shuffle the signal against forward returns, recompute the mean IC across
    months, and repeat ``n_perm`` times. The p-value is the fraction of shuffles whose |mean IC| >=
    the observed |mean IC|. A real cross-sectional signal sits in the tail; noise does not.
    """
    obs = abs(ic_summary(history)["mean_ic"])
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    groups = [(g["rank_improve"].to_numpy(dtype=float), g["fwd_ret"].to_numpy(dtype=float))
              for _, g in history.groupby("month")]
    count = 0
    for _ in range(n_perm):
        ics = []
        for sig, ret in groups:
            perm = rng.permutation(len(ret))
            ics.append(spearman_ic(sig, ret[perm]))
        m = np.nanmean(ics)
        if abs(m) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Robustness — sign/magnitude across disjoint month blocks
# ---------------------------------------------------------------------------
def robustness_windows(history: pd.DataFrame, n_blocks: int = 4) -> list[dict]:
    """Split the months into ``n_blocks`` disjoint contiguous blocks; report each block's IC summary.

    Returns a list of dicts (one per block): ``{label, mean_ic, t, n_months}``.
    """
    months = sorted(history["month"].unique())
    if len(months) < n_blocks:
        return []
    chunks = np.array_split(np.array(months), n_blocks)
    out = []
    for ch in chunks:
        sub = history[history["month"].isin(ch.tolist())]
        s = ic_summary(sub)
        out.append({
            "label": f"months {int(ch[0])}-{int(ch[-1])}",
            "mean_ic": s["mean_ic"], "t": s["t"], "n_months": s["n_months"],
        })
    return out


# ---------------------------------------------------------------------------
# Synthetic positive control — seed-robust mean IC-t (house rule)
# ---------------------------------------------------------------------------
def synthetic_mean_ic_t(
    data_mod,
    rank_alpha: float,
    n_seeds: int = 25,
    n_months: int = 48,
    n_names: int = 30,
    signal_noise: float = 1.0,
    base_seed: int = 552,
) -> float:
    """Average the pooled mean-IC-*t* over ``n_seeds`` synthetic worlds for a planted ``rank_alpha``.

    House rule: any synthetic-dependent claim averages the test stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean of the pooled IC-*t* across seeds.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        hist = data_mod.synthetic_history(
            n_names=n_names, n_months=n_months, rank_alpha=rank_alpha,
            signal_noise=signal_noise, seed=s,
        )
        ts.append(ic_summary(hist)["t"])
    return float(np.nanmean(ts))

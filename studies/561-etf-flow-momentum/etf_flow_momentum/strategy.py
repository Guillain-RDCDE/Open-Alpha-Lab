"""The engine and its honest controls — Study 561 (ETF-Flow-Momentum).

The claim, at full strength: sector/asset ETFs that pull in the most *new money* (net
creation-unit inflows this month) go on to *outperform* their peers next month — flow momentum.
The reversal/trap view says the high-inflow ETFs are crowded and subsequently *lag*. Either way
the tradable expression is the same cross-sectional book — **long the top-flow ETFs, short the
bottom-flow ETFs** — and the *sign* of its spread is the whole question.

This module measures that, honestly, on the deterministic synthetic panel from ``data.py``:

1. **The flow signal.** Each month, the cross-section of net flows (fraction of prior-month
   assets). Rank funds by flow; the *inflow* leg is the top fraction, the *outflow* leg the
   bottom fraction.

2. **The spread portfolio.** Each month go long the inflow leg, short the outflow leg,
   equal-weight, dollar-neutral, and hold one month. This produces a monthly time series of
   long-short returns. The headline statistic is a **one-sample t** on that series (does the
   mean monthly spread differ from zero?), reported as an annualised return too.

3. **The placebo null.** Shuffle the flow *labels* against forward returns within each month
   many times; the p-value is the fraction of shuffles whose |mean spread| >= the observed. A
   real cross-sectional flow signal sits in the tail; noise does not.

4. **Costs + borrow.** The book rebalances monthly, so each leg pays a one-way cost every
   rebalance (in and out), and the short leg pays a borrow. ETFs are cheap to trade and cheap to
   borrow, so the frictions are modest but real and applied honestly.

5. **A robustness sweep.** Re-run the spread at several leg fractions (top/bottom 25 %, 33 %,
   50 %) and check the sign and t are stable.

6. **A synthetic positive control.** A deterministic world where the flow effect is planted
   (``flow_alpha != 0``); the engine must recover the right sign / a significant t and stay flat
   at the null — averaged over >= 20 seeds so no single lucky RNG seed can manufacture it.

Execution convention (one documented lag, no look-ahead): the flow signal at month ``t`` uses
only flows realised *through* month ``t``; the position is entered at the month-``t`` close and
held over month ``t+1`` (the ``fwd_ret`` column is already next-month's return). No future data
enters the signal. This is the single execution lag; there is no same-bar fill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# The monthly long-short spread series
# ---------------------------------------------------------------------------
def monthly_spread_series(panel: pd.DataFrame, frac: float = 0.33) -> pd.Series:
    """Monthly long-inflow / short-outflow spread return series.

    For each month: rank the ETFs by ``flow``, take the top ``frac`` (inflow leg) long and the
    bottom ``frac`` (outflow leg) short, equal-weight, and record the mean forward return of the
    inflow leg minus the outflow leg. ``frac`` is the tail fraction per leg (0.33 ≈ terciles for
    a ~16-ETF cross-section).

    Returns a ``pd.Series`` indexed by month of the long-short spread (before costs).
    """
    if panel.empty:
        return pd.Series(dtype=float)
    out = {}
    for m, g in panel.groupby("month"):
        if len(g) < 4:
            continue
        g = g.sort_values("flow")
        k = max(1, int(round(len(g) * frac)))
        low = g.head(k)   # outflow leg (short)
        high = g.tail(k)  # inflow leg (long)
        out[int(m)] = float(high["fwd_ret"].mean() - low["fwd_ret"].mean())
    return pd.Series(out, name="spread").sort_index()


def _one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd <= 0:
        return float("nan")
    return float(x.mean() / (sd / np.sqrt(n)))


def spread_stats(panel: pd.DataFrame, frac: float = 0.33) -> dict:
    """Headline stats of the flow long-short spread.

    Returns the mean monthly spread, its annualised value (×12), the one-sample t on the monthly
    series, the number of months, and the monthly hit rate (share of months the spread is > 0).
    A *positive* mean spread = flow momentum (inflows keep winning); *negative* = the reversal
    trap.
    """
    s = monthly_spread_series(panel, frac=frac)
    if s.empty:
        return {"mean_m": float("nan"), "ann": float("nan"), "t": float("nan"),
                "n_months": 0, "hit": float("nan")}
    arr = s.to_numpy(dtype=float)
    return {
        "mean_m": float(np.nanmean(arr)),
        "ann": float(np.nanmean(arr) * MONTHS_PER_YEAR),
        "t": _one_sample_t(arr),
        "n_months": int(np.isfinite(arr).sum()),
        "hit": float(np.mean(arr > 0)),
    }


# ---------------------------------------------------------------------------
# Firm-level pooled relation — the sign IS the claim
# ---------------------------------------------------------------------------
def pooled_slope(panel: pd.DataFrame) -> dict:
    """Pooled OLS of forward return on the *within-month* z-scored flow.

    De-mean flow and forward return within each month (to strip the common market factor), then
    regress. Slope > 0 = flow momentum (more inflow, more next-month return). Returns the slope,
    a clustered-by-month t (months are the independent units), and the pooled correlation.
    """
    if panel.empty or panel["month"].nunique() < 3:
        return {"slope": float("nan"), "t": float("nan"), "corr": float("nan"), "n": 0}
    df = panel.copy()
    # within-month z-score of flow, and within-month de-mean of forward return
    def _z(x):
        sd = x.std(ddof=0)
        return (x - x.mean()) / sd if sd > 0 else x * 0.0
    df["zflow"] = df.groupby("month")["flow"].transform(_z)
    df["dret"] = df.groupby("month")["fwd_ret"].transform(lambda x: x - x.mean())
    x = df["zflow"].to_numpy(dtype=float)
    y = df["dret"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4 or x.std() == 0:
        return {"slope": float("nan"), "t": float("nan"), "corr": float("nan"), "n": len(x)}
    # pooled slope + correlation via covariance
    slope = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    corr = float(np.corrcoef(x, y)[0, 1])
    # t from the per-month slope estimates (months = independent units — robust to cross-name
    # correlation within a month, the usual Fama-MacBeth logic)
    per = []
    for _, g in df.groupby("month"):
        gx = g["zflow"].to_numpy(dtype=float)
        gy = g["dret"].to_numpy(dtype=float)
        m = np.isfinite(gx) & np.isfinite(gy)
        gx, gy = gx[m], gy[m]
        if len(gx) >= 3 and gx.std() > 0:
            per.append(float(np.cov(gx, gy, ddof=1)[0, 1] / np.var(gx, ddof=1)))
    t = _one_sample_t(np.array(per)) if len(per) >= 2 else float("nan")
    return {"slope": slope, "t": t, "corr": corr, "n": len(x)}


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(panel: pd.DataFrame, frac: float = 0.33, n_perm: int = 2000,
                   seed: int = 561) -> float:
    """Label-shuffle placebo p-value for the mean monthly spread.

    Within each month, shuffle the flow labels against forward returns, recompute the spread
    series and its mean; the p-value is the fraction of shuffles whose |mean spread| >= the
    observed |mean spread|. A real signal sits in the tail; noise does not.
    """
    if panel.empty or panel["month"].nunique() < 3:
        return float("nan")
    obs = abs(np.nanmean(monthly_spread_series(panel, frac=frac).to_numpy(dtype=float)))
    rng = np.random.default_rng(seed)
    # pre-split months for speed
    groups = [g[["flow", "fwd_ret"]].to_numpy(dtype=float) for _, g in panel.groupby("month")
              if len(g) >= 4]
    count = 0
    for _ in range(n_perm):
        spreads = []
        for arr in groups:
            n = len(arr)
            k = max(1, int(round(n * frac)))
            perm = rng.permutation(n)
            flow = arr[:, 0][perm]
            ret = arr[:, 1]
            order = np.argsort(flow)
            rs = ret[order]
            spreads.append(rs[-k:].mean() - rs[:k].mean())
        if abs(float(np.mean(spreads))) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_annualised(panel: pd.DataFrame, frac: float = 0.33, cost_bps: float = 3.0,
                   borrow_ann_bps: float = 40.0) -> dict:
    """Gross and net annualised spread after monthly rebalance costs + short borrow.

    The book rebalances every month, so each leg pays a one-way ``cost_bps`` on entry and exit
    each month (4 crossings/month: long in+out, short in+out), and the short leg pays an annual
    ``borrow_ann_bps``. ETFs are cheap and liquid, so both are modest — but applied honestly.
    Returns ``{"gross_ann", "net_ann", "cost_ann", "borrow_ann"}`` in return units.
    """
    s = spread_stats(panel, frac=frac)
    gross_ann = s["ann"]
    cost_ann = 4.0 * cost_bps * 1e-4 * MONTHS_PER_YEAR  # per-month round trip, annualised
    borrow_ann = borrow_ann_bps * 1e-4
    net_ann = gross_ann - cost_ann - borrow_ann
    return {
        "gross_ann": float(gross_ann),
        "net_ann": float(net_ann),
        "cost_ann": float(cost_ann),
        "borrow_ann": float(borrow_ann),
    }


# ---------------------------------------------------------------------------
# Robustness — leg-fraction sweep
# ---------------------------------------------------------------------------
def robustness_sweep(panel: pd.DataFrame, fracs=(0.25, 0.33, 0.50)) -> pd.DataFrame:
    """Spread stats across several leg fractions; the sign/t should be stable."""
    rows = []
    for f in fracs:
        st = spread_stats(panel, frac=f)
        rows.append({"frac": f, "ann": st["ann"], "t": st["t"], "n_months": st["n_months"],
                     "hit": st["hit"]})
    return pd.DataFrame(rows).set_index("frac")


# ---------------------------------------------------------------------------
# Seed-robust synthetic control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_t(data_mod, flow_alpha: float, n_seeds: int = 25,
                     base_seed: int = 561, frac: float = 0.33,
                     idio_vol: float = 0.05) -> float:
    """Average the spread one-sample t over ``n_seeds`` synthetic worlds.

    The house rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single
    lucky RNG seed can manufacture significance. Returns the mean spread-t across seeds for a
    planted ``flow_alpha``.
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(flow_alpha=flow_alpha, idio_vol=idio_vol, seed=s)
        ts.append(spread_stats(panel, frac=frac)["t"])
    return float(np.nanmean(ts))




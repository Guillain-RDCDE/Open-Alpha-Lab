"""The Organizational-Capital strategy -- Eisfeldt & Papanikolaou (2013).

Construction (faithful to the paper's recipe, simplified for a survivor basket):

  1. **Org-capital stock by perpetual inventory.** SG&A is treated as investment in
     organizational capital. The stock accumulates and depreciates::

         OC_t = (1 - delta) * OC_{t-1} + SG&A_t / cpi_t

     with depreciation ``delta = 0.15`` (Eisfeldt-Papanikolaou). The initial stock is
     seeded as ``OC_0 = SG&A_1 / (g + delta)`` where ``g`` is the long-run SG&A growth rate
     (we use a fixed g = 0.10, as in the paper's steady-state seed). We omit the CPI deflator
     (real vs nominal makes a negligible difference to a *cross-sectional rank*).

  2. **Scale by assets.** The sort variable is ``oc/assets`` -- organizational capital per
     dollar of physical capital. (The paper scales by book assets to make the stock
     comparable across firm sizes.)

  3. **Annual cross-sectional sort.** Each year, rank the basket by ``oc/assets`` and split
     into a high-OC half (long) and a low-OC half (short).

  4. **Long-short, held 12 months.** Equal-weight each leg; the spread is high minus low.
     Eisfeldt-Papanikolaou predict a *positive* premium: high-OC firms bear more risk
     (their key input -- talent -- is mobile and can walk out the door) so they earn more.

Inference: monthly long-short returns, one-sample **Newey-West HAC t-stat** (the bar number),
a **label-shuffle placebo** (shuffle the oc/assets ranks within each year -> null), costs
(one-way bps x turnover, plus borrow on the short leg), and a deterministic synthetic control.

EXECUTION LAG: the rank uses fundamentals with fiscal year ending on or before the prior
June; the position is then entered at the close one trading day after the rank date and held
for the next 12 months. One lag, documented, no same-bar fills.

The basket is **survivorship-biased** (current large-caps still trading in 2026).
All positive results are upper-bound estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Eisfeldt-Papanikolaou perpetual-inventory parameters.
DELTA = 0.15           # org-capital depreciation rate
SEED_GROWTH = 0.10     # long-run SG&A growth for the steady-state initial stock

# Costs.
COST_BPS = 5.0         # one-way transaction cost per leg, basis points
BORROW_BPS_ANNUAL = 50.0  # annual borrow fee on the short leg, basis points

RF_ANNUAL = 0.03       # constant risk-free for excess-return framing


# ---------------------------------------------------------------------------
# Org-capital perpetual inventory
# ---------------------------------------------------------------------------
def org_capital_stock(
    fundamentals: pd.DataFrame,
    delta: float = DELTA,
    seed_growth: float = SEED_GROWTH,
) -> pd.DataFrame:
    """Perpetual-inventory org-capital stock from annual SG&A.

    ``fundamentals`` is a long frame with columns ``ticker, fyear, sga, assets``.
    Returns the same rows plus ``oc`` (the accumulated stock) and ``oc_assets`` (oc / assets).

    For each firm, sorted by fiscal year:
        OC_first = SG&A_first / (seed_growth + delta)        # steady-state seed
        OC_t     = (1 - delta) * OC_{t-1} + SG&A_t           # accumulate
    """
    df = fundamentals.sort_values(["ticker", "fyear"]).reset_index(drop=True)
    oc_vals = np.empty(len(df), dtype="float64")
    pos = 0
    for _, grp in df.groupby("ticker", sort=False):
        sga = grp["sga"].to_numpy(dtype="float64")
        oc_prev = sga[0] / (seed_growth + delta)
        oc_vals[pos] = oc_prev
        for i in range(1, len(sga)):
            oc_prev = (1.0 - delta) * oc_prev + sga[i]
            oc_vals[pos + i] = oc_prev
        pos += len(sga)
    out = df.copy()
    out["oc"] = oc_vals
    out["oc_assets"] = oc_vals / out["assets"].to_numpy(dtype="float64")
    out["fyear"] = out["fyear"].astype(int)
    return out[["ticker", "fyear", "sga", "assets", "oc", "oc_assets"]]


# ---------------------------------------------------------------------------
# Cross-sectional long-short
# ---------------------------------------------------------------------------
def long_short(
    oc: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    lag_days: int = 1,
    min_names: int = 10,
) -> pd.DataFrame:
    """Annual high-minus-low org-capital long-short on a monthly return panel.

    ``oc``: output of :func:`org_capital_stock` (must contain ``oc_assets`` per ticker-year).
    ``monthly_returns``: wide (period_or_datetime index x ticker) monthly returns.

    For each formation year Y, the rank uses fundamentals from fiscal year ``Y-1`` (already
    public by mid-Y -- the one-period lag). The portfolio is held over months of year Y.
    Returns a frame indexed by month with columns ``high, low, ls, n`` and an annual
    ``rebalance`` flag (first month of each holding year).

    ``lag_days`` documents the single execution lag in the prose (monthly data already enters
    the month after the signal is public); kept for the costs accounting.
    """
    rets = monthly_returns
    idx = rets.index
    if isinstance(idx, pd.PeriodIndex):
        years = np.asarray(idx.year)
        months_of = np.asarray(idx.month)
    else:
        idx = pd.DatetimeIndex(idx)
        years = np.asarray(idx.year)
        months_of = np.asarray(idx.month)

    cols = list(rets.columns)
    col_ix = {c: i for i, c in enumerate(cols)}
    R = rets.to_numpy(dtype="float64")  # (months x firms)

    rank_by_year = {y: g.set_index("ticker")["oc_assets"] for y, g in oc.groupby("fyear")}

    rows: list[dict] = []
    for i in range(R.shape[0]):
        rank = rank_by_year.get(years[i] - 1)  # fundamentals from prior fiscal year
        if rank is None:
            continue
        avail = [t for t in rank.index if t in col_ix]
        rk = rank.loc[avail].dropna()
        if len(rk) < min_names:
            continue
        vals = rk.to_numpy(dtype="float64")
        med = np.median(vals)
        ix = np.array([col_ix[t] for t in rk.index], dtype=int)
        hi_mask = vals > med
        lo_mask = ~hi_mask
        row = R[i, ix]
        hi = float(np.nanmean(row[hi_mask])) if hi_mask.any() else np.nan
        lo = float(np.nanmean(row[lo_mask])) if lo_mask.any() else np.nan
        rows.append({
            "month": idx[i], "high": hi, "low": lo, "ls": hi - lo,
            "n": int(len(rk)), "rebalance": bool(months_of[i] == 1),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("month")


# ---------------------------------------------------------------------------
# Costs: one-way bps x turnover + borrow on the short leg
# ---------------------------------------------------------------------------
def apply_costs(
    ls_returns: pd.Series,
    rebalance_flags: pd.Series,
    cost_bps: float = COST_BPS,
    borrow_bps_annual: float = BORROW_BPS_ANNUAL,
) -> pd.Series:
    """Net the long-short of trading and borrow costs.

    Trading: at each annual rebalance both legs are fully turned over in the worst case.
    We charge ``2 * cost_bps`` (one-way per leg, long + short) on the rebalance month -- a
    deliberately conservative 100% turnover assumption for an annual cross-sectional sort.

    Borrow: the short leg pays ``borrow_bps_annual`` per year, charged monthly (every month
    the short is open).
    """
    out = ls_returns.astype(float).copy()
    trade_cost = 2.0 * cost_bps / 1e4
    borrow_monthly = (borrow_bps_annual / 1e4) / 12.0
    out = out - borrow_monthly  # borrow charged every month
    out = out - rebalance_flags.astype(float) * trade_cost
    return out


# ---------------------------------------------------------------------------
# Newey-West HAC summary
# ---------------------------------------------------------------------------
def summary(monthly_returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised stats + Newey-West HAC t-stat for a monthly return series."""
    r = pd.Series(monthly_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mean_m = float(r.mean())
    vol_m = float(r.std(ddof=1))
    mean_ann = mean_m * periods_per_year
    vol_ann = vol_m * np.sqrt(periods_per_year)
    sr = mean_ann / vol_ann if vol_ann > 0 else np.nan

    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())

    arr = r.to_numpy()
    mu = arr.mean()
    e = arr - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else np.nan

    return {
        "mean": mean_ann, "vol": vol_ann, "sharpe": sr, "tstat": tstat,
        "hit_rate": float((r > 0).mean()), "max_drawdown": dd, "n": n,
    }


# ---------------------------------------------------------------------------
# Placebo: shuffle the org-capital labels within each year (null)
# ---------------------------------------------------------------------------
def placebo_pvalue(
    oc: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    n_shuffles: int = 500,
    seed: int = 527,
    min_names: int = 10,
) -> dict:
    """Label-shuffle placebo: how often does a shuffled OC rank beat the real long-short?

    Each iteration permutes ``oc_assets`` within each fiscal year (destroying the firm->rank
    link while preserving the cross-sectional distribution), reruns the long-short, and
    records its full-sample mean. The placebo p-value is the fraction of shuffles whose mean
    is at least as large (in absolute value) as the real mean.
    """
    real = long_short(oc, monthly_returns, min_names=min_names)
    if real.empty:
        return {"real_mean": np.nan, "p_value": np.nan, "n_shuffles": 0}
    real_mean = float(real["ls"].mean())

    # Vectorised null. Precompute, per month, the firm-column indices and the formation-year
    # rank vector once; each shuffle just permutes the rank-to-firm assignment within the
    # formation year and recomputes the high/low split with NumPy. Same null as the loop above
    # (shuffle oc/assets labels within each year), thousands of times faster.
    rets = monthly_returns
    cols = list(rets.columns)
    col_ix = {c: i for i, c in enumerate(cols)}
    if isinstance(rets.index, pd.PeriodIndex):
        years = np.asarray(rets.index.year)
    else:
        years = np.asarray(pd.DatetimeIndex(rets.index).year)
    R = rets.to_numpy(dtype="float64")  # (months x firms)

    rank_by_year = {y: g.set_index("ticker")["oc_assets"] for y, g in oc.groupby("fyear")}
    # For each month: the array of firm column-indices that have a rank this month, and their
    # ranks (formation = year-1).
    month_firm_ix: list[np.ndarray] = []
    month_ranks: list[np.ndarray] = []
    for mi in range(R.shape[0]):
        rank = rank_by_year.get(years[mi] - 1)
        if rank is None:
            month_firm_ix.append(np.empty(0, dtype=int)); month_ranks.append(np.empty(0))
            continue
        firms = [t for t in rank.index if t in col_ix]
        rk = rank.loc[firms].dropna()
        if len(rk) < min_names:
            month_firm_ix.append(np.empty(0, dtype=int)); month_ranks.append(np.empty(0))
            continue
        month_firm_ix.append(np.array([col_ix[t] for t in rk.index], dtype=int))
        month_ranks.append(rk.to_numpy(dtype="float64"))

    rng = np.random.default_rng(seed)
    null_means = np.full(n_shuffles, np.nan)
    for s in range(n_shuffles):
        # One permutation of ranks per formation year, shared across that year's months.
        year_perm: dict[int, np.ndarray] = {}
        per_month = np.full(R.shape[0], np.nan)
        for mi in range(R.shape[0]):
            ix = month_firm_ix[mi]
            if ix.size == 0:
                continue
            fy = years[mi] - 1
            ranks = month_ranks[mi]
            perm = year_perm.get(fy)
            if perm is None or perm.shape[0] != ranks.shape[0]:
                perm = rng.permutation(ranks.shape[0])
                year_perm[fy] = perm
            shuffled_rank = ranks[perm]
            med = np.median(shuffled_rank)
            hi_mask = shuffled_rank > med
            lo_mask = ~hi_mask
            row = R[mi, ix]
            hi = np.nanmean(row[hi_mask]) if hi_mask.any() else np.nan
            lo = np.nanmean(row[lo_mask]) if lo_mask.any() else np.nan
            per_month[mi] = hi - lo
        valid = per_month[~np.isnan(per_month)]
        null_means[s] = float(valid.mean()) if valid.size else np.nan

    null_means = null_means[~np.isnan(null_means)]
    if len(null_means) == 0:
        return {"real_mean": real_mean, "p_value": np.nan, "n_shuffles": 0}
    p = float((np.abs(null_means) >= abs(real_mean)).mean())
    return {
        "real_mean": real_mean, "p_value": p,
        "null_mean": float(null_means.mean()), "null_std": float(null_means.std()),
        "n_shuffles": int(len(null_means)),
    }


# ---------------------------------------------------------------------------
# Synthetic positive control -- faithful-engine power check ONLY
# ---------------------------------------------------------------------------
def synthetic_control(premiums=(-0.06, -0.03, 0.0, 0.03, 0.06, 0.10),
                      seed: int = 527, n_seeds: int = 12) -> pd.DataFrame:
    """Run the full pipeline on the synthetic world for a sweep of planted OC premiums.

    For each premium we average the long-short HAC t over ``n_seeds`` RNG seeds (so a single
    lucky seed cannot masquerade as signal -- the desk's seed-robustness rule). Returns a
    frame: ``premium, mean_ls_%/yr, mean_t``. This is a POWER CHECK ONLY -- never cited to
    support a real-tape stamp.
    """
    from . import data as _data

    rows = []
    for prem in premiums:
        means, ts = [], []
        for k in range(n_seeds):
            fund, monthly, _ = _data.synthetic_panel(oc_premium=prem, seed=seed + k)
            oc = org_capital_stock(fund)
            ls = long_short(oc, monthly)
            if ls.empty:
                continue
            s = summary(ls["ls"])
            means.append(s["mean"] * 100)
            ts.append(s["tstat"])
        rows.append({
            "premium": prem,
            "mean_ls_pct": float(np.mean(means)) if means else np.nan,
            "mean_t": float(np.mean(ts)) if ts else np.nan,
        })
    return pd.DataFrame(rows)

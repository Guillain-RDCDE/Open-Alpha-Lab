"""The engine and its honest controls — Study 572 (Capex-Cycle).

The claim, at full strength: a firm *accelerating* its capital expenditure relative to its asset
base — a **capex binge** — earns lower future returns than one *harvesting* cash (letting capex
intensity fall). This is the capex-*growth* cousin of the asset-growth anomaly (Cooper-Gulen-Schill
2008) and the capex-*level* anomaly (Titman-Wei-Xie 2004): the over-investment channel expressed as
the **change** in investment intensity — the phase of the capex cycle.

This module measures that, honestly, on what a retail (yfinance) stack can reach:

1. **The capex-cycle signal.** For each firm,
       capex_intensity_t = |CapEx_t| / TotalAssets_{t-1}
       capex_cycle       = capex_intensity_t - capex_intensity_{t-1}
   A positive ``capex_cycle`` is a binge (ramping); negative is harvesting.

2. **The cross-sectional IC (the headline).** Pearson (and Spearman) correlation between
   ``capex_cycle`` and forward return across the basket. The over-investment hypothesis predicts a
   **negative** IC (bingers underperform). We attach a *t*-stat to the Pearson IC.

3. **The tercile hedge.** Long the low-capex_cycle (harvesting) tercile, short the high
   (bingeing) tercile; report the spread, a two-sample (Welch) *t*, and costs + short borrow.

4. **A label-shuffle placebo** null on the hedge spread.

5. **A multi-window robustness sweep** (re-score at earlier fiscal splits; read the sign).

6. **A seed-robust synthetic positive control** (>=20 seeds): plant the effect and prove the
   engine recovers a negative IC-*t*, and stays flat at the null.

Execution convention: capex_cycle is scored from the *last two reported* fiscal years (fully
public at the scoring close), and the position is entered at that same close and held over the
*forward* window. No future data enters the signal (no look-ahead). The same-close fill is one
documented lag; on a ~2-year hold a one-bar-later entry moves the headline by a hair and changes
no stamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------
def forward_return(prices: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Simple total return per name over the forward window (start, end]."""
    px = prices.loc[start:end]
    if len(px) < 2:
        return pd.Series(dtype=float)
    return px.iloc[-1] / px.iloc[0] - 1.0


# ---------------------------------------------------------------------------
# The capex-cycle signal, from a cash-flow snapshot
# ---------------------------------------------------------------------------
def capex_cycle_signal(cf_entry: dict, upto_year: int | None = None) -> float | None:
    """capex_cycle = capex_intensity_t - capex_intensity_{t-1} from a cash-flow snapshot.

    ``cf_entry`` = ``{"years": [...], "capex": [...], "assets": [...]}`` oldest -> newest.
    capex_intensity_y = |CapEx_y| / TotalAssets_{y-1}. The change of that ratio across the two most
    recent fiscal years (or ending at ``upto_year`` if given) is the signal. Needs >=3 fiscal years
    (t, t-1, and the t-2 asset base). Returns ``None`` if not computable.
    """
    years = list(cf_entry.get("years", []))
    capex = list(cf_entry.get("capex", []))
    assets = list(cf_entry.get("assets", []))
    if len(years) < 3:
        return None
    idx = {y: i for i, y in enumerate(years)}
    if upto_year is None:
        t = years[-1]
    else:
        cand = [y for y in years if y <= upto_year]
        if not cand:
            return None
        t = max(cand)
    # need t, t-1, t-2 present as consecutive-ish reported years
    if t not in idx:
        return None
    it = idx[t]
    if it < 2:
        return None
    y_t, y_tm1, y_tm2 = years[it], years[it - 1], years[it - 2]
    a_tm1, a_tm2 = assets[it - 1], assets[it - 2]
    if a_tm1 == 0 or a_tm2 == 0:
        return None
    intensity_t = capex[it] / a_tm1
    intensity_tm1 = capex[it - 1] / a_tm2
    return float(intensity_t - intensity_tm1)


def build_panel(
    prices: pd.DataFrame,
    cashflow: dict,
    split_date: str,
    end_date: str | None = None,
    upto_year: int | None = None,
) -> pd.DataFrame:
    """Assemble the cross-sectional panel the sort/IC consume, from real inputs.

    - *capex_cycle* comes from the cash-flow snapshot (the accounting leg), scored up to
      ``upto_year`` (default: latest fiscal year in the snapshot).
    - *forward_ret* is the price return from ``split_date`` to ``end_date`` (the holding period).

    ``split_date`` is the calendar close where the signal becomes public and the trade is entered;
    ``upto_year`` fixes the last fiscal year used in the signal (so earlier windows use earlier
    fiscal data — no look-ahead). The single execution lag, made explicit.
    """
    names = [t for t in cashflow.keys() if t in prices.columns]
    if not names:
        return pd.DataFrame()
    end_date = end_date or str(prices.index[-1].date())
    fr = forward_return(prices[names], split_date, end_date)
    rows = {}
    for t in names:
        sig = capex_cycle_signal(cashflow[t], upto_year=upto_year)
        if sig is None or t not in fr.index or pd.isna(fr[t]):
            continue
        rows[t] = {"capex_cycle": sig, "forward_ret": float(fr[t])}
    panel = pd.DataFrame.from_dict(rows, orient="index")
    panel.index.name = "ticker"
    return panel.dropna()


# ---------------------------------------------------------------------------
# The cross-sectional IC (the headline)
# ---------------------------------------------------------------------------
def information_coefficient(panel: pd.DataFrame) -> dict:
    """Pearson IC (with t-stat) and Spearman IC between capex_cycle and forward return.

    The over-investment hypothesis predicts a **negative** IC (bingers underperform). The Pearson
    IC gets a t-stat via t = r * sqrt((n-2)/(1-r^2)).
    """
    if panel.empty or len(panel) < 4:
        return {"ic": float("nan"), "ic_t": float("nan"), "spearman": float("nan"), "n": 0}
    x = panel["capex_cycle"].to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    n = len(x)
    if x.std() == 0 or y.std() == 0:
        return {"ic": float("nan"), "ic_t": float("nan"), "spearman": float("nan"), "n": n}
    r = float(np.corrcoef(x, y)[0, 1])
    denom = max(1.0 - r * r, 1e-12)
    ic_t = r * np.sqrt((n - 2) / denom)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    sp = float(np.corrcoef(rx, ry)[0, 1])
    return {"ic": r, "ic_t": float(ic_t), "spearman": sp, "n": int(n)}


# ---------------------------------------------------------------------------
# The tercile hedge (long harvest / short binge)
# ---------------------------------------------------------------------------
def bucket_returns(panel: pd.DataFrame, frac: float = 0.3) -> dict:
    """Sort by capex_cycle; long the LOW (harvest) tercile, short the HIGH (binge) tercile.

    The anomaly predicts harvest > binge (a positive long-low/short-high spread). Returns bucket
    means, the spread, sizes, and per-name forward returns (for the two-sample t).
    """
    if panel.empty:
        return {}
    df = panel.sort_values("capex_cycle")
    k = max(1, int(round(len(df) * frac)))
    harvest = df.head(k)          # lowest capex_cycle (harvesting / cutting)
    binge = df.tail(k)            # highest capex_cycle (bingeing / ramping)
    h_ret = harvest["forward_ret"]
    b_ret = binge["forward_ret"]
    return {
        "n": int(len(df)),
        "k": int(k),
        "harvest_mean": float(h_ret.mean()),
        "binge_mean": float(b_ret.mean()),
        "spread": float(h_ret.mean() - b_ret.mean()),
        "harvest_ret": h_ret,
        "binge_ret": b_ret,
    }


def hedge_tstat(buckets: dict) -> dict:
    """Two-sample (Welch) t on harvest - binge forward returns."""
    if not buckets:
        return {"spread": float("nan"), "t": float("nan"), "n": 0}
    a = buckets["harvest_ret"].to_numpy(dtype=float)
    b = buckets["binge_ret"].to_numpy(dtype=float)
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
def placebo_pvalue(panel: pd.DataFrame, n_perm: int = 2000, seed: int = 572) -> float:
    """Label-shuffle placebo p-value for the harvest - binge hedge spread.

    Shuffle the capex_cycle *labels* against forward returns ``n_perm`` times; the p-value is the
    fraction of shuffles whose |spread| >= the observed |spread|.
    """
    if panel.empty or len(panel) < 6:
        return float("nan")
    obs = abs(bucket_returns(panel)["spread"])
    rng = np.random.default_rng(seed)
    x = panel["capex_cycle"].to_numpy(dtype=float)
    y = panel["forward_ret"].to_numpy(dtype=float)
    n = len(y)
    k = max(1, int(round(n * 0.3)))
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        order = np.argsort(x[perm])           # sort returns by shuffled capex_cycle
        ys = y[order]
        spread = ys[:k].mean() - ys[-k:].mean()
        if abs(spread) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Costs + borrow
# ---------------------------------------------------------------------------
def net_spread(buckets: dict, cost_bps: float = 5.0, borrow_ann_bps: float = 100.0,
               holding_years: float = 1.0) -> float:
    """Long-harvest / short-binge spread net of costs and short borrow.

    Book built once and held ``holding_years``: charge a one-way cost on each leg (entry + exit,
    both legs = 4 crossings) plus an annual short borrow on the bingeing leg (pro-rated).
    """
    if not buckets:
        return float("nan")
    gross = buckets["spread"]
    cost = 4.0 * cost_bps * 1e-4
    borrow = borrow_ann_bps * 1e-4 * holding_years
    return float(gross - cost - borrow)


# ---------------------------------------------------------------------------
# Robustness — seed-robust synthetic control (house rule for synthetic claims)
# ---------------------------------------------------------------------------
def synthetic_mean_ic_t(data_mod, binge_alpha: float, n_seeds: int = 25,
                        base_seed: int = 572) -> float:
    """Average the cross-sectional IC-t over ``n_seeds`` synthetic worlds.

    House rule: any synthetic-dependent claim averages the stat over >= 20 seeds so no single lucky
    seed manufactures significance. Returns the mean IC-t across seeds for a planted ``binge_alpha``
    (negative alpha -> negative IC-t: bingers underperform).
    """
    ts = []
    for s in range(base_seed, base_seed + n_seeds):
        panel, _ = data_mod.synthetic_panel(binge_alpha=binge_alpha, seed=s)
        ts.append(information_coefficient(panel)["ic_t"])
    return float(np.nanmean(ts))

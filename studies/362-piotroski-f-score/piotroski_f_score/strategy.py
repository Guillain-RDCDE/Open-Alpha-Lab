"""Strategy + inference for Study 362 — Piotroski F-Score.

Joseph Piotroski's F-Score (2000, *Value Investing: The Use of Historical Financial
Statement Information to Separate Winners from Losers*) is a **9-point** accounting-health
checklist. Each point is a binary 0/1; a firm's F-Score is their sum (0 = worst, 9 = best):

**Profitability (4 points)**
  1. ROA > 0                   (net income / total assets, current year positive)
  2. CFO > 0                   (operating cash flow positive)
  3. ΔROA > 0                  (ROA improved vs last year)
  4. Accruals: CFO > NetIncome (cash earnings exceed accounting earnings)

**Leverage / liquidity / source of funds (3 points)**
  5. ΔLeverage < 0             (long-term-debt / assets fell)
  6. ΔCurrentRatio > 0         (current ratio rose)
  7. No new shares issued      (shares outstanding did not rise)

**Operating efficiency (2 points)**
  8. ΔGrossMargin > 0          (gross margin rose)
  9. ΔAssetTurnover > 0        (sales / assets rose)

The pitch: high-F-score firms (8–9) beat low-F-score firms (0–1) — fundamentals
"separate winners from losers." We test it as a cross-sectional **long-minus-short**
portfolio: each fiscal year, long the high-score firms, short the low-score firms, and
hold next calendar year (a conservative reporting lag). We measure:

  * the **annual long-minus-short spread** (and the high / low / market legs);
  * a **Newey-West (HAC) t-stat** of the spread (short annual series);
  * a **placebo / bootstrap null** — random same-size portfolios, to see how often chance
    matches the high-minus-low spread;
  * the **monotonicity** of returns across the 0..9 score (is it actually ordered?);
  * **one calendar-year execution lag** (fiscal year y -> returns of year y+1) and a
    **one-way cost × turnover** charge with **borrow on the short leg**.

Survivorship is named on the Signal axis: the real basket survived to 2026, a mild
upward tilt; a control that plants a known edge proves the engine, not the market.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# F-score construction
# --------------------------------------------------------------------------- #
def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a / b.where(b != 0)


def compute_fscore(panels: dict) -> pd.DataFrame:
    """Assemble the 9-point Piotroski F-Score from a dict of (year × ticker) concept frames.

    Expects keys: NetIncomeLoss, Assets, NetCashProvidedByUsedInOperatingActivities,
    LiabilitiesCurrent, AssetsCurrent, LongTermDebtNoncurrent, GrossProfit, Revenues,
    CommonStockSharesOutstanding. Returns a (year × ticker) integer F-score (0..9), NaN
    where the prior-year comparison is unavailable. Year-over-year points compare a fiscal
    year to the one before it on the same ticker.
    """
    ni = panels["NetIncomeLoss"]
    assets = panels["Assets"]
    cfo = panels["NetCashProvidedByUsedInOperatingActivities"]
    cl = panels["LiabilitiesCurrent"]
    ca = panels["AssetsCurrent"]
    ltd = panels["LongTermDebtNoncurrent"]
    gp = panels["GrossProfit"]
    rev = panels["Revenues"]
    sh = panels["CommonStockSharesOutstanding"]

    # common ticker set across the metrics we need
    frames = [ni, assets, cfo, cl, ca, ltd, gp, rev, sh]
    common = set(assets.columns)
    for f in frames:
        common &= set(f.columns)
    tickers = sorted(common)
    years = sorted(set().union(*[set(f.index) for f in frames]))
    idx = pd.Index(years, name="year")

    def R(f):  # reindex onto the common grid
        return f.reindex(index=idx, columns=tickers)

    ni, assets, cfo, cl, ca, ltd, gp, rev, sh = (R(f) for f in
                                                 (ni, assets, cfo, cl, ca, ltd, gp, rev, sh))

    roa = _safe_div(ni, assets)
    lev = _safe_div(ltd, assets)                 # leverage
    cr = _safe_div(ca, cl)                       # current ratio
    margin = _safe_div(gp, rev)                  # gross margin
    turn = _safe_div(rev, assets)                # asset turnover

    p_roa = roa > 0                              # 1
    p_cfo = cfo > 0                              # 2
    p_droa = roa.diff() > 0                      # 3
    p_accr = cfo > ni                            # 4 (accruals quality)
    p_lev = lev.diff() < 0                       # 5
    p_cr = cr.diff() > 0                         # 6
    p_share = sh.diff() <= 0                     # 7 (no new shares)
    p_margin = margin.diff() > 0                 # 8
    p_turn = turn.diff() > 0                     # 9

    points = [p_roa, p_cfo, p_droa, p_accr, p_lev, p_cr, p_share, p_margin, p_turn]
    # a point counts only where its inputs are defined; require ROA & Assets present at all
    valid = roa.notna() & assets.notna() & roa.diff().notna()
    fscore = sum(p.astype(float) for p in points)
    fscore = fscore.where(valid)
    fscore.index.name = "year"
    return fscore


# --------------------------------------------------------------------------- #
# Portfolio construction
# --------------------------------------------------------------------------- #
def long_short(fscore: pd.DataFrame, fwd_ret: pd.DataFrame,
               hi: int = 7, lo: int = 4, min_names: int = 2) -> pd.DataFrame:
    """High-minus-low F-score portfolio, fiscal year y -> calendar-year y+1 returns.

    Long the firms with F-score >= ``hi``, short those with F-score <= ``lo``. Returns a
    frame indexed by signal year with columns ``high``, ``low``, ``market``, ``spread``
    (= high - low), and the leg counts. Years with too few names on either leg are skipped.
    """
    rows: dict[int, dict] = {}
    for y in fscore.index:
        s = fscore.loc[y].dropna()
        yr = int(y) + 1
        if yr not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[yr].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < min_names * 2:
            continue
        hi_mask = s >= hi
        lo_mask = s <= lo
        if hi_mask.sum() < min_names or lo_mask.sum() < min_names:
            continue
        hi_ret = nxt[hi_mask].mean()
        lo_ret = nxt[lo_mask].mean()
        mkt = nxt.mean()
        rows[int(y)] = {
            "high": float(hi_ret), "low": float(lo_ret), "market": float(mkt),
            "spread": float(hi_ret - lo_ret),
            "n": int(len(s)), "n_hi": int(hi_mask.sum()), "n_lo": int(lo_mask.sum()),
        }
    return pd.DataFrame(rows).T.sort_index()


def score_buckets(fscore: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.DataFrame:
    """Mean next-year return by F-score bucket {0-1, 2-3, 4-5, 6-7, 8-9}, pooled over years.

    Returns a frame indexed by bucket label with ``mean_ret`` and ``n`` (firm-years) —
    the monotonicity check (does return rise with the score?).
    """
    labels = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    pool: dict[str, list[float]] = {f"{a}-{b}": [] for a, b in labels}
    for y in fscore.index:
        s = fscore.loc[y].dropna()
        yr = int(y) + 1
        if yr not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[yr].dropna()
        s, nxt = s.align(nxt, join="inner")
        for a, b in labels:
            m = (s >= a) & (s <= b)
            pool[f"{a}-{b}"].extend(nxt[m].tolist())
    rows = {k: {"mean_ret": float(np.mean(v)) if v else np.nan, "n": len(v)}
            for k, v in pool.items()}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_tstat(series: pd.Series) -> dict:
    """Newey-West (HAC) t-stat of the mean of an annual return series.

    Short annual series: even one lag of autocorrelation inflates a naive t. Returns
    ``{mean, vol, sharpe, tstat, hit, n}``.
    """
    r = pd.Series(series).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {"mean": np.nan, "vol": np.nan, "sharpe": np.nan, "tstat": np.nan,
                "hit": np.nan, "n": n}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    t = float(mu / se) if se > 0 else float("nan")
    return {"mean": mu, "vol": sd, "sharpe": (mu / sd if sd > 0 else np.nan),
            "tstat": t, "hit": float((r > 0).mean()), "n": n}


def placebo_pvalue(fscore: pd.DataFrame, fwd_ret: pd.DataFrame, hi: int = 7, lo: int = 4,
                   n_draws: int = 20_000, seed: int = 362, min_names: int = 2) -> dict:
    """Placebo null: replace the F-score legs by random same-size portfolios.

    For each year we know the high-leg size and low-leg size. Draw random disjoint subsets
    of those sizes from the same universe, form the random spread, average over years, and
    repeat ``n_draws`` times. ``p`` = share of random spreads >= the observed mean spread —
    the honest answer to "could random stock-picking match the F-score?".
    """
    obs = long_short(fscore, fwd_ret, hi=hi, lo=lo, min_names=min_names)
    if obs.empty:
        return {"obs_spread": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan"), "n_years": 0}
    obs_spread = float(obs["spread"].mean())

    # precompute per-year (returns array, n_hi, n_lo)
    years = []
    for y in obs.index:
        s = fscore.loc[y].dropna()
        nxt = fwd_ret.loc[int(y) + 1].dropna()
        s, nxt = s.align(nxt, join="inner")
        years.append((nxt.to_numpy(), int(obs.loc[y, "n_hi"]), int(obs.loc[y, "n_lo"])))

    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        sp = []
        for arr, n_hi, n_lo in years:
            idx = rng.permutation(len(arr))
            hi_pick = arr[idx[:n_hi]]
            lo_pick = arr[idx[n_hi:n_hi + n_lo]]
            sp.append(hi_pick.mean() - lo_pick.mean())
        means[i] = np.mean(sp)
    p = float((means >= obs_spread).mean())
    return {"obs_spread": obs_spread, "placebo_mean": float(means.mean()),
            "p_value": p, "n_years": int(len(obs))}


def net_of_costs(ls: pd.DataFrame, cost_bps: float = 10.0, borrow_bps: float = 50.0) -> dict:
    """Charge the long-minus-short spread realistic frictions.

    The book turns over ~fully each year (annual rebalance): one-way cost on each leg
    (charged twice — open + close ≈ one round-trip per leg per year) plus a borrow fee on
    the short leg. Returns gross and net mean spread and the HAC t of the net series.
    """
    if ls.empty:
        return {"gross_mean": np.nan, "net_mean": np.nan, "net_t": np.nan, "n": 0}
    c = cost_bps / 1e4
    bor = borrow_bps / 1e4
    # two legs, one round-trip each per year = 4 one-way crossings -> 2 round-trips
    cost = 2.0 * (2.0 * c) + bor
    gross = ls["spread"]
    net = gross - cost
    h = hac_tstat(net)
    return {"gross_mean": float(gross.mean()), "net_mean": float(net.mean()),
            "net_t": h["tstat"], "cost_drag": float(cost), "n": int(len(gross))}


def summarize(fscore: pd.DataFrame, fwd_ret: pd.DataFrame, hi: int = 7, lo: int = 4,
              min_names: int = 2) -> dict:
    """Headline bundle: leg means, spread, HAC t, placebo p, costs."""
    ls = long_short(fscore, fwd_ret, hi=hi, lo=lo, min_names=min_names)
    spread_stats = hac_tstat(ls["spread"]) if not ls.empty else hac_tstat(pd.Series(dtype=float))
    high_stats = hac_tstat(ls["high"]) if not ls.empty else spread_stats
    pl = placebo_pvalue(fscore, fwd_ret, hi=hi, lo=lo, min_names=min_names)
    costs = net_of_costs(ls)
    return {
        "n_years": int(len(ls)),
        "high_mean": high_stats["mean"],
        "low_mean": float(ls["low"].mean()) if not ls.empty else np.nan,
        "market_mean": float(ls["market"].mean()) if not ls.empty else np.nan,
        "spread_mean": spread_stats["mean"],
        "spread_t": spread_stats["tstat"],
        "spread_hit": spread_stats["hit"],
        "placebo_p": pl["p_value"],
        "gross_mean": costs["gross_mean"],
        "net_mean": costs["net_mean"],
        "net_t": costs["net_t"],
    }

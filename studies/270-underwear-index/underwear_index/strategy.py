"""Strategy and analysis layer for Study 270 (Underwear-Index).

The Men's Underwear Index (Greenspan): a *dip* in men's-underwear sales is supposed
to flag (or lead) a recession. We turn that into two falsifiable tests:

  - **Recession prediction.** Does a year-Y dip in the index predict an NBER
    recession in year Y or Y+1? We compute hit-rate, the unconditional base rate
    (how often is any year a recession year anyway?), a Fisher exact test on the
    2x2 dip x recession table, and a permutation test on the dip labels.

  - **Market timing.** Does going to cash the year *after* a dip beat buy-and-hold
    the S&P 500? We build the no-look-ahead strategy (dip in year Y -> flat in Y+1)
    and report gross and net (after one-way costs) annualised returns vs B&H, plus a
    HAC (Newey-West) t-stat on the strategy-minus-benchmark return spread.

The killer here is tiny-n and base-rate neglect. There are only ~33 annual
observations and ~5 recession years in the window. With so few events any "dip ->
recession" coincidence is over-credited, and the market drifts up ~73% of years
regardless, so a "predict trouble" rule that sits out gives up the drift for free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

SEED = 270


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def dip_signal(df: pd.DataFrame) -> pd.Series:
    """Return +1 when the underwear index dipped that year (yoy < 0), else 0.

    A +1 is the Greenspan 'trouble' flag: it predicts a recession / a bad market
    in the *following* year (the no-look-ahead convention).
    """
    return (df["yoy"] < 0).astype(int).rename("dip_signal")


# ---------------------------------------------------------------------------
# Recession-prediction inference
# ---------------------------------------------------------------------------
def recession_stats(
    df: pd.DataFrame,
    horizon: str = "next",
    n_permutations: int = 10_000,
    seed: int = SEED,
) -> dict:
    """Test whether a dip predicts a recession (Fisher exact + permutation).

    Parameters
    ----------
    df : DataFrame with ``dip`` (bool), ``recession`` (bool, same year), and
         ``next_recession`` (bool, year+1) columns.
    horizon : 'same' uses the same-year recession flag (coincident view);
              'next' uses the year+1 flag (the leading-indicator view).
    n_permutations : permutation draws (shuffle dip labels).
    seed : RNG seed.

    Returns a dict with the 2x2 cell counts, dip-conditional and unconditional
    recession rates, the Fisher exact p-value, and a permutation p-value on the
    excess recession rate in dip years.
    """
    rec_col = "next_recession" if horizon == "next" else "recession"
    dip = df["dip"].to_numpy(dtype=bool)
    rec = df[rec_col].to_numpy(dtype=bool)

    # Drop rows with NaN outcome (the last year has no year+1 outcome).
    valid = ~pd.isna(df[rec_col].to_numpy())
    dip = dip[valid]
    rec = rec[valid]
    n = int(dip.sum() + (~dip).sum())

    a = int((dip & rec).sum())       # dip & recession
    b = int((dip & ~rec).sum())      # dip & no-recession
    c = int((~dip & rec).sum())      # no-dip & recession
    d = int((~dip & ~rec).sum())     # no-dip & no-recession

    n_dip = a + b
    n_nodip = c + d
    rec_rate_dip = a / n_dip if n_dip > 0 else float("nan")
    rec_rate_nodip = c / n_nodip if n_nodip > 0 else float("nan")
    uncond_rec_rate = (a + c) / n if n > 0 else float("nan")

    # Fisher exact test on the 2x2 table (one-sided: dip -> more recessions).
    _, fisher_p = scipy_stats.fisher_exact([[a, b], [c, d]], alternative="greater")

    # Permutation test: shuffle dip labels, recompute excess recession rate in
    # 'dip' years; p = fraction of shuffles with excess >= observed.
    obs_excess = (rec_rate_dip - rec_rate_nodip) if (n_dip > 0 and n_nodip > 0) else 0.0
    rng = np.random.default_rng(seed)
    perm_excess = np.empty(n_permutations, dtype=float)
    rec_int = rec.astype(int)
    for i in range(n_permutations):
        sh = rng.permuted(dip)
        nd = int(sh.sum())
        nnd = n - nd
        if nd == 0 or nnd == 0:
            perm_excess[i] = 0.0
            continue
        rd = rec_int[sh].mean()
        rnd = rec_int[~sh].mean()
        perm_excess[i] = rd - rnd
    perm_p = float((perm_excess >= obs_excess).mean())

    return {
        "horizon": horizon,
        "n": n,
        "a_dip_rec": a,
        "b_dip_norec": b,
        "c_nodip_rec": c,
        "d_nodip_norec": d,
        "n_dip": n_dip,
        "n_nodip": n_nodip,
        "rec_rate_dip": round(float(rec_rate_dip), 4),
        "rec_rate_nodip": round(float(rec_rate_nodip), 4),
        "uncond_rec_rate": round(float(uncond_rec_rate), 4),
        "excess_rec_rate": round(float(obs_excess), 4),
        "fisher_p": round(float(fisher_p), 4),
        "perm_p": round(float(perm_p), 4),
        "perm_excess": perm_excess,  # full distribution, for plots
    }


# ---------------------------------------------------------------------------
# Market-timing strategy (no look-ahead) + HAC t on the spread
# ---------------------------------------------------------------------------
def _newey_west_t(x: np.ndarray, lags: int = 1) -> float:
    """HAC (Newey-West) t-stat for H0: mean(x) = 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for L in range(1, lags + 1):
        if L >= n:
            break
        w = 1.0 - L / (lags + 1)
        cov = (e[L:] @ e[:-L]) / n
        var += 2.0 * w * cov
    se = np.sqrt(var / n)
    if se == 0:
        return float("nan")
    return float(mu / se)


def timing_backtest(
    df: pd.DataFrame,
    cost_bps: float = 10.0,
) -> dict:
    """Backtest the no-look-ahead 'dip -> go to cash next year' timing rule.

    A dip in year Y means we are FLAT the S&P in year Y+1 (and earn 0); otherwise
    we are long and earn ``next_sp500_return``. We compare against always-long
    buy-and-hold of ``next_sp500_return``. One-way trading costs (``cost_bps`` of
    NAV) are charged each year the position changes (in or out).

    Returns annualised gross/net strategy returns, B&H annualised return, turnover,
    the per-year spread series, and a Newey-West t-stat on (strategy - B&H).
    Price-only (Shiller nominal price return); no dividends. Survivorship is not a
    concern here (single index), which is noted on the Signal axis.
    """
    d = df.dropna(subset=["next_sp500_return"]).copy()
    bh = d["next_sp500_return"].to_numpy(dtype=float)
    in_market = (~d["dip"].to_numpy(dtype=bool)).astype(float)  # flat the year after a dip

    # Turnover: position changes between consecutive years (plus the initial entry).
    pos = in_market
    changes = np.abs(np.diff(np.concatenate([[0.0], pos])))
    turnover = float(changes.mean())
    cost = changes * (cost_bps * 1e-4)

    strat_gross = pos * bh
    strat_net = strat_gross - cost

    n_yrs = len(d)
    bh_ann = float((1.0 + bh).prod() ** (1.0 / n_yrs) - 1.0)
    strat_gross_ann = float((1.0 + strat_gross).prod() ** (1.0 / n_yrs) - 1.0)
    strat_net_ann = float((1.0 + strat_net).prod() ** (1.0 / n_yrs) - 1.0)

    spread = strat_net - bh
    nw_t = _newey_west_t(spread, lags=1)

    return {
        "n_years": n_yrs,
        "cost_bps": cost_bps,
        "turnover": round(turnover, 4),
        "bh_ann": round(bh_ann, 4),
        "strat_gross_ann": round(strat_gross_ann, 4),
        "strat_net_ann": round(strat_net_ann, 4),
        "spread_mean": round(float(np.nanmean(spread)), 4),
        "nw_t": round(float(nw_t), 3),
        "years": d["year"].to_numpy(),
        "bh": bh,
        "strat_net": strat_net,
    }


# ---------------------------------------------------------------------------
# Summarize — compact R-dict for notebooks / results.md
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame, n_permutations: int = 10_000, seed: int = SEED) -> dict:
    """One-stop summary dict for the real merged DataFrame.

    Returns a flat dict mirroring the R = dict(...) in build_notebooks.py and
    docs/results.md.
    """
    rs_next = recession_stats(df, horizon="next", n_permutations=n_permutations, seed=seed)
    rs_same = recession_stats(df, horizon="same", n_permutations=n_permutations, seed=seed)
    bt = timing_backtest(df, cost_bps=10.0)

    return {
        "n": rs_next["n"],
        "n_dips": rs_next["n_dip"],
        # leading-indicator (year+1) view
        "uncond_rec_pct": round(rs_next["uncond_rec_rate"] * 100, 1),
        "rec_rate_dip_pct": round(rs_next["rec_rate_dip"] * 100, 1),
        "rec_rate_nodip_pct": round(rs_next["rec_rate_nodip"] * 100, 1),
        "next_fisher_p": rs_next["fisher_p"],
        "next_perm_p": rs_next["perm_p"],
        # coincident (same-year) view
        "same_rec_rate_dip_pct": round(rs_same["rec_rate_dip"] * 100, 1),
        "same_fisher_p": rs_same["fisher_p"],
        "same_perm_p": rs_same["perm_p"],
        # timing
        "bh_ann_pct": round(bt["bh_ann"] * 100, 1),
        "strat_gross_pct": round(bt["strat_gross_ann"] * 100, 1),
        "strat_net_pct": round(bt["strat_net_ann"] * 100, 1),
        "turnover": bt["turnover"],
        "nw_t": bt["nw_t"],
    }

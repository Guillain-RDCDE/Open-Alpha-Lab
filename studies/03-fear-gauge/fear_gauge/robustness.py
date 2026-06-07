"""Robustness: the three ways a "buy the VIX" result can be a mirage.

1. Tiny effective sample — VIX events cluster violently (Feb–Mar 2020 alone is a
   quarter of the famous chart's events). :func:`block_bootstrap_excess` resamples
   *blocks* of days, preserving the clustering, for an honest CI on the excess.

2. A selected window — the chart that started this study runs **2016–2026**, which
   quietly excludes 2008 (and 1998, and the 2000–02 bear). :func:`window_sensitivity`
   recomputes the same excess on the cherry-picked window versus the full history,
   so you can watch the edge move when the crashes are put back.

3. The martingale — "double down at 50" is a sizing rule, and the window that sells
   it is exactly the one with no 2008. :func:`martingale_ruin` runs the add-on-the-
   way-down rule through history and reports how often it would have blown past a
   ruin drawdown before the bounce arrived.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .benchmark import forward_returns


# --- Standard-normal helpers (kept dependency-free; no scipy needed) --------- #

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF via Acklam's rational approximation (|err| < 1.2e-9)."""
    if not 0.0 < p < 1.0:
        return float("-inf") if p <= 0.0 else float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def block_bootstrap_excess(
    market: pd.DataFrame,
    signal: pd.Series,
    horizon: int = 21,
    block: int = 21,
    n_iter: int = 2000,
    seed: int = 0,
) -> dict:
    """Block-bootstrap confidence interval for the conditional excess return.

    Resamples contiguous blocks of (forward-return, is-event) pairs so the
    clustering of VIX events is preserved, then recomputes the excess
    (mean_cond - mean_uncond) on each synthetic history. A CI that straddles zero
    means the apparent edge is within sampling noise once clustering is respected.

    Returns ``mean, ci_low, ci_high, p_excess_le_0``.
    """
    rng = np.random.default_rng(seed)
    fwd = forward_returns(market, horizon)
    sig = signal.reindex(market.index).fillna(False).to_numpy()
    valid = ~np.isnan(fwd)
    fwd, sig = fwd[valid], sig[valid]
    n = len(fwd)
    if n == 0 or sig.sum() == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_excess_le_0": np.nan}

    n_blocks = int(np.ceil(n / block))
    excesses = np.empty(n_iter)
    for i in range(n_iter):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        f, s = fwd[idx], sig[idx]
        if s.sum() == 0:
            excesses[i] = np.nan
            continue
        excesses[i] = f[s].mean() - f.mean()

    excesses = excesses[~np.isnan(excesses)]
    return {
        "mean": float(np.mean(excesses)),
        "ci_low": float(np.percentile(excesses, 2.5)),
        "ci_high": float(np.percentile(excesses, 97.5)),
        "p_excess_le_0": float((excesses <= 0).mean()),
    }


# The window the famous chart uses, versus the deep sample that includes the
# crashes it omits. Boundaries are inclusive.
WINDOWS = {
    "chart (2016–2026)": ("2016-06-01", "2026-06-30"),
    "full history": (None, None),
    "ex-2016+ (pre-window)": (None, "2016-05-31"),
}


def window_sensitivity(
    market: pd.DataFrame,
    signal: pd.Series,
    horizon: int = 21,
) -> pd.DataFrame:
    """How the conditional excess moves when you change the sample window.

    Recomputes ``mean_cond``, ``mean_uncond`` and ``excess`` at ``horizon`` for each
    window in :data:`WINDOWS`. The point of the study in one table: if the excess is
    large on "chart (2016–2026)" and shrinks (or flips) on "full history", the edge
    was a property of the *window*, not the *signal*.

    Returns one row per window: ``start, end, n_events, mean_cond, mean_uncond,
    excess``.
    """
    sig_full = signal.reindex(market.index).fillna(False)
    rows = []
    for name, (start, end) in WINDOWS.items():
        m = market
        s = sig_full
        if start is not None:
            keep = m.index >= pd.Timestamp(start)
            m, s = m[keep], s[keep]
        if end is not None:
            keep = m.index <= pd.Timestamp(end)
            m, s = m[keep], s[keep]
        fwd = forward_returns(m, horizon)
        valid = ~np.isnan(fwd)
        ev = s.to_numpy() & valid
        pool = fwd[valid]
        cond = fwd[ev]
        rows.append({
            "window": name,
            "start": start or str(m.index.min().date()),
            "end": end or str(m.index.max().date()),
            "n_events": int(ev.sum()),
            "mean_cond": float(cond.mean()) if cond.size else np.nan,
            "mean_uncond": float(pool.mean()) if pool.size else np.nan,
            "excess": float(cond.mean() - pool.mean()) if cond.size else np.nan,
        })
    return pd.DataFrame(rows).set_index("window")


def martingale_ruin(
    market: pd.DataFrame,
    vix: pd.DataFrame,
    rung1: float = 30.0,
    rung2: float = 50.0,
    hold_days: int = 63,
    ruin_drawdown: float = -0.20,
) -> dict:
    """Stress-test the "buy at VIX 30, double down at 50" martingale.

    For each fresh cross of ``rung1`` we deploy one unit of capital; if VIX reaches
    ``rung2`` while the trade is open we deploy a *second* unit at the (lower) price
    — the "double down". We then **hold the blended position for a fixed
    ``hold_days``** (≈ a quarter by default) — the honest version of "wait for the
    bounce" — and track the deepest mark-to-market drawdown along the way, not just
    the first time it pokes back to break-even.

    Why fixed-hold and not exit-on-recovery: a buyer who closes the instant they're
    green never *experiences* the tail. The danger of doubling down is precisely the
    drawdown you must sit through (and, levered, get margin-called on) before the
    recovery the rule assumes. So the headline is ``p_ruin`` = the fraction of
    episodes whose worst intra-hold drawdown breaches ``ruin_drawdown``.

    A small ``p_ruin`` on the no-2008 window next to a larger one (and a fatter
    ``worst_drawdown``) on the full sample is the whole point of beat 6: the rule
    "works" exactly until the path the selling window leaves out.

    Returns ``n_episodes, p_ruin, p_doubled, mean_terminal, worst_drawdown,
    worst_terminal``.
    """
    close = market["Close"].reindex(vix.index).to_numpy()
    level = vix["Close"].to_numpy()
    n = len(level)
    rising = level >= rung1
    rising[1:] &= level[:-1] < rung1  # fresh crosses only

    episodes = []
    last = -10**9
    for i in np.flatnonzero(rising):
        if i - last < hold_days or i + hold_days >= n:
            continue
        last = i
        cost = 1.0                 # one unit of capital deployed at the cross
        shares = 1.0 / close[i]
        added = False
        worst = 0.0
        for j in range(i + 1, i + hold_days + 1):
            if not added and level[j] >= rung2:
                cost += 1.0                  # double down: a second unit...
                shares += 1.0 / close[j]     # ...at the current price
                added = True
            worst = min(worst, shares * close[j] / cost - 1.0)
        terminal = shares * close[i + hold_days] / cost - 1.0
        episodes.append({"worst": worst, "terminal": terminal, "doubled": added})

    if not episodes:
        return {"n_episodes": 0, "p_ruin": np.nan, "p_doubled": np.nan,
                "mean_terminal": np.nan, "worst_drawdown": np.nan,
                "worst_terminal": np.nan}

    df = pd.DataFrame(episodes)
    return {
        "n_episodes": int(len(df)),
        "p_ruin": float((df["worst"] <= ruin_drawdown).mean()),
        "p_doubled": float(df["doubled"].mean()),
        "mean_terminal": float(df["terminal"].mean()),
        "worst_drawdown": float(df["worst"].min()),
        "worst_terminal": float(df["terminal"].min()),
    }


def deflated_sharpe(best_sharpe: float, n_trials: int, n_obs: int) -> float:
    """Probabilistic Sharpe ratio deflated for the number of strategies tried.

    A simplified Bailey-López de Prado deflated Sharpe: the probability that the
    best observed (annualised) Sharpe over ``n_trials`` configs would NOT have
    arisen by chance under a true Sharpe of zero. ``n_obs`` is the number of return
    observations behind the Sharpe (e.g. trading days). Near 1 = unlikely a fluke;
    near 0.5 or below = consistent with picking the luckiest of many coin-flips.

    This is the discount that turns the family scan's headline "Sharpe 14 on 5
    trades" back into the noise it is.
    """
    if n_trials < 1 or n_obs < 2 or not np.isfinite(best_sharpe):
        return np.nan
    euler = 0.5772156649
    z = _norm_ppf(1 - 1.0 / n_trials) * (1 - euler) + _norm_ppf(1 - 1.0 / (n_trials * np.e)) * euler
    sr_daily = best_sharpe / np.sqrt(252)
    expected_max_daily = z / np.sqrt(n_obs)
    deflated = (sr_daily - expected_max_daily) * np.sqrt(n_obs)
    return float(_norm_cdf(deflated))


def split_sample(market: pd.DataFrame, frac: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological in-sample / out-of-sample split (no shuffling).

    Pick the best (trigger, exit) on the first ``frac`` of history, confirm it on
    the held-out tail. An edge that evaporates out-of-sample was data-mining.
    """
    cut = int(len(market) * frac)
    return market.iloc[:cut].copy(), market.iloc[cut:].copy()

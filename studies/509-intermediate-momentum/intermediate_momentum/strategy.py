"""Intermediate vs recent momentum -- Novy-Marx (2012).

Robert Novy-Marx ("Is momentum really momentum?", JFE 2012) showed that the cross-sectional
momentum premium is driven by the **intermediate** part of the formation window -- the
return from roughly twelve to seven months ago (t-12..t-7) -- rather than the **recent**
part (t-6..t-2). Sorting on intermediate past performance carries the drift; sorting on
recent past performance does not (and the recent window is contaminated by short-term
reversal).

Construction (monthly, equal-weight, dollar-neutral long-short):
  1. At each month-end rebalance date *t*, for every stock compute two formation-window
     returns from daily adjusted-close prices:
       - intermediate: cumulative return over [t-12mo, t-7mo)
       - recent:       cumulative return over [t-6mo, t-1mo)   (skip the most recent month)
  2. Rank cross-sectionally; long the top tercile, short the bottom tercile.
  3. Skip one trading day after *t* (execution lag), then hold the equal-weight long-short
     over the next calendar month.
  4. Report the two long-shorts side by side: does the intermediate one carry the drift?

Inference: one-sample / Newey-West HAC *t* on the monthly long-short series, a label-shuffle
placebo null (permute the cross-sectional ranks), and turnover-based costs with borrow on
the short leg. A deterministic synthetic positive control proves the engine is faithful.

The panel is **survivorship-biased** (current large-cap names projected backwards). All
positive results are upper-bound estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate trading days per month.
DAYS_PER_MONTH = 21

# Formation windows, in trading days back from the rebalance date.
# Intermediate: months t-12 .. t-7  ->  [12*21, 7*21) = [252, 147)
INT_FAR, INT_NEAR = 12 * DAYS_PER_MONTH, 7 * DAYS_PER_MONTH
# Recent: months t-6 .. t-1 (skip the most recent month for short-term reversal) -> [6*21, 1*21)
REC_FAR, REC_NEAR = 6 * DAYS_PER_MONTH, 1 * DAYS_PER_MONTH

# Tercile fraction for the long and short legs.
TERCILE = 1.0 / 3.0

# Execution lag (trading days after the signal date before the holding period starts).
EXEC_LAG = 1

# Costs.
COST_BPS = 5.0          # one-way transaction cost, bps of traded notional
BORROW_BPS_ANNUAL = 50.0  # annual borrow fee on the short leg


# ---------------------------------------------------------------------------
# Formation-window returns
# ---------------------------------------------------------------------------
def _window_return(prices: pd.DataFrame, t_idx: int, far: int, near: int) -> pd.Series:
    """Cumulative return for each stock over [t_idx-far, t_idx-near), using positional index.

    Returns NaN for stocks lacking valid endpoints. ``far`` > ``near`` (both look back).
    """
    i0 = t_idx - far
    i1 = t_idx - near
    if i0 < 0 or i1 <= i0:
        return pd.Series(np.nan, index=prices.columns)
    p0 = prices.iloc[i0]
    p1 = prices.iloc[i1]
    return (p1 / p0) - 1.0


# ---------------------------------------------------------------------------
# Long-short construction
# ---------------------------------------------------------------------------
def momentum_longshort(
    prices: pd.DataFrame,
    window: str = "intermediate",
    tercile: float = TERCILE,
    min_stocks: int = 18,
    exec_lag: int = EXEC_LAG,
) -> pd.DataFrame:
    """Build the monthly equal-weight dollar-neutral momentum long-short.

    ``window`` is "intermediate" (t-12..t-7) or "recent" (t-6..t-1). At each month-end the
    cross-section is ranked on the chosen formation-window return; long the top ``tercile``,
    short the bottom ``tercile``. After ``exec_lag`` trading days the equal-weight long-short
    is held for one calendar month.

    Returns a DataFrame indexed by rebalance date with columns:
        ``long_ret``  -- equal-weight return of the winner tercile over the hold
        ``short_ret`` -- equal-weight return of the loser tercile over the hold
        ``ls_ret``    -- long_ret - short_ret (gross long-short)
        ``turnover``  -- one-sided name turnover vs the previous rebalance (long+short)
        ``n``         -- number of ranked stocks
    """
    if window == "intermediate":
        far, near = INT_FAR, INT_NEAR
    elif window == "recent":
        far, near = REC_FAR, REC_NEAR
    else:
        raise ValueError("window must be 'intermediate' or 'recent'")

    daily = prices.pct_change()
    idx = prices.index

    # Month-end rebalance positions (positional indices of last trading day each month).
    month_keys = idx.to_period("M")
    last_pos = {}
    for pos, key in enumerate(month_keys):
        last_pos[key] = pos
    rebal_pos = sorted(last_pos.values())
    # Keep only rebalances with enough history.
    rebal_pos = [p for p in rebal_pos if p >= far]

    rows: list[dict] = []
    prev_long: set = set()
    prev_short: set = set()

    for k in range(len(rebal_pos) - 1):
        t_idx = rebal_pos[k]
        next_idx = rebal_pos[k + 1]

        sig = _window_return(prices, t_idx, far, near).dropna()
        if len(sig) < min_stocks:
            continue

        n_leg = max(1, int(round(len(sig) * tercile)))
        ranked = sig.sort_values()
        short_names = list(ranked.index[:n_leg])
        long_names = list(ranked.index[-n_leg:])

        # Holding period: exec_lag days after t_idx, through next rebalance (inclusive).
        h0 = t_idx + exec_lag
        h1 = next_idx
        if h0 >= h1 or h1 >= len(idx):
            continue
        hold = daily.iloc[h0 + 1 : h1 + 1]  # returns realised strictly after entry day
        if hold.empty:
            continue

        def _cum(names: list) -> float:
            sub = hold[names]
            comp = (1.0 + sub).prod() - 1.0
            return float(comp.mean())

        long_ret = _cum(long_names)
        short_ret = _cum(short_names)
        ls_ret = long_ret - short_ret

        cur_long, cur_short = set(long_names), set(short_names)
        if prev_long or prev_short:
            turn_long = len(cur_long - prev_long) / max(len(cur_long), 1)
            turn_short = len(cur_short - prev_short) / max(len(cur_short), 1)
            turnover = 0.5 * (turn_long + turn_short)
        else:
            turnover = 1.0
        prev_long, prev_short = cur_long, cur_short

        rows.append(
            {
                "date": idx[t_idx],
                "long_ret": long_ret,
                "short_ret": short_ret,
                "ls_ret": ls_ret,
                "turnover": turnover,
                "n": int(len(sig)),
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# Net-of-cost returns
# ---------------------------------------------------------------------------
def apply_costs(
    ls: pd.DataFrame,
    cost_bps: float = COST_BPS,
    borrow_bps_annual: float = BORROW_BPS_ANNUAL,
) -> pd.Series:
    """Net monthly long-short return: gross minus turnover-driven costs and short borrow.

    Each rebalance trades ``turnover`` of each leg (long + short = 2 legs), so traded
    notional per month ~ 2 * turnover * NAV. One-way cost = cost_bps applies to each side of
    each leg change. Borrow accrues monthly on the (full) short leg notional.
    """
    cost_rate = cost_bps / 1e4
    borrow_monthly = (borrow_bps_annual / 1e4) / 12.0
    # 2 legs, each turned over `turnover`, entered and exited -> 2*2 one-way trades scaled.
    trade_cost = ls["turnover"] * 2.0 * 2.0 * cost_rate
    return ls["ls_ret"] - trade_cost - borrow_monthly


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def summary(monthly_returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised statistics + one-sample Newey-West HAC *t* for a monthly return series."""
    r = pd.Series(monthly_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}

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
        "mean": mean_ann,
        "vol": vol_ann,
        "sharpe": sr,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd,
        "n": n,
    }


def placebo_pvalue(
    prices: pd.DataFrame,
    window: str = "intermediate",
    n_shuffles: int = 200,
    seed: int = 509,
    **ls_kwargs,
) -> dict:
    """Label-shuffle placebo: how often does a *randomly relabelled* sort match the real mean?

    We recompute the long-short but, at each rebalance, randomly permute which stocks are
    "winners" vs "losers" (destroying the signal while preserving the cross-sectional return
    distribution and the holding mechanics). The placebo p-value is the share of shuffled
    runs whose mean long-short return is >= the real mean.
    """
    real = momentum_longshort(prices, window=window, **ls_kwargs)
    if real.empty:
        return {"real_mean": np.nan, "p_value": np.nan, "n_shuffles": 0}
    real_mean = float(real["ls_ret"].mean())

    if window == "intermediate":
        far, near = INT_FAR, INT_NEAR
    else:
        far, near = REC_FAR, REC_NEAR
    tercile = ls_kwargs.get("tercile", TERCILE)
    min_stocks = ls_kwargs.get("min_stocks", 18)
    exec_lag = ls_kwargs.get("exec_lag", EXEC_LAG)

    daily = prices.pct_change()
    idx = prices.index
    month_keys = idx.to_period("M")
    last_pos: dict = {}
    for pos, key in enumerate(month_keys):
        last_pos[key] = pos
    rebal_pos = [p for p in sorted(last_pos.values()) if p >= far]

    rng = np.random.default_rng(seed)
    ge = 0
    done = 0
    for _ in range(n_shuffles):
        means = []
        for k in range(len(rebal_pos) - 1):
            t_idx = rebal_pos[k]
            next_idx = rebal_pos[k + 1]
            sig = _window_return(prices, t_idx, far, near).dropna()
            if len(sig) < min_stocks:
                continue
            names = list(sig.index)
            rng.shuffle(names)
            n_leg = max(1, int(round(len(names) * tercile)))
            short_names = names[:n_leg]
            long_names = names[-n_leg:]
            h0 = t_idx + exec_lag
            h1 = next_idx
            if h0 >= h1 or h1 >= len(idx):
                continue
            hold = daily.iloc[h0 + 1 : h1 + 1]
            if hold.empty:
                continue
            lr = float(((1.0 + hold[long_names]).prod() - 1.0).mean())
            srt = float(((1.0 + hold[short_names]).prod() - 1.0).mean())
            means.append(lr - srt)
        if not means:
            continue
        if float(np.mean(means)) >= real_mean:
            ge += 1
        done += 1

    p = ge / done if done else np.nan
    return {"real_mean": real_mean, "p_value": p, "n_shuffles": done}

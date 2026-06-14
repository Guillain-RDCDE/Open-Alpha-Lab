"""Strategy for Study 140 (Amihud-Illiquidity) — the annual ILLIQ quintile sort.

Amihud (2002) proposes ILLIQ = mean( |r_d| / DolVol_d ) as a simple proxy for price
impact / illiquidity. The premium claim: stocks with high ILLIQ earn higher returns in
the following year as compensation for bearing liquidity risk.

We test this with two honest controls:

1. **Equal-weight market baseline** — all stocks with valid ILLIQ in that year.
2. **Random-quintile control** — portfolios of identical size drawn randomly, to confirm
   any apparent Q5/Q1 spread is attributable to the ILLIQ signal rather than to portfolio
   concentration, size, or luck.

The verdict is carried by the HAC t-stat on the annual Q5 - Q1 hedge return series.

**Survivorship bias**: the panel is the current S&P 500 projected backwards. All spread
numbers are upper bounds. The premium famously lives in micro-caps, not large-caps, so
we expect a muted or absent effect here — any positive result should be discounted heavily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal transformation
# ---------------------------------------------------------------------------
def log_illiq(illiq: pd.DataFrame) -> pd.DataFrame:
    """Log-transform ILLIQ (natural log). ILLIQ is right-skewed; log makes it bell-shaped.

    Cells where ILLIQ <= 0 are set to NaN (should not occur on good data, but defensive).
    """
    return np.log(illiq.replace(0.0, np.nan))


# ---------------------------------------------------------------------------
# Quintile sort
# ---------------------------------------------------------------------------
def quintile_sort(
    illiq: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quintiles: int = 5,
    use_log: bool = True,
) -> pd.DataFrame:
    """Annual quintile sort on ILLIQ -> forward return by bucket.

    For each sort year y:
      - Rank stocks by ILLIQ (or log-ILLIQ) cross-sectionally into ``n_quintiles`` buckets.
      - Quintile 1 = most *liquid* (lowest ILLIQ), Quintile 5 = most *illiquid* (highest).
      - The return for each quintile is the equal-weight mean of next-year (y+1) returns.
      - Hedge = Q5 return - Q1 return (the claimed illiquidity premium, long illiquid / short liquid).
      - Market = equal-weight mean across all stocks with valid ILLIQ.

    Returns a DataFrame indexed by year with columns:
    ``q1, q2, q3, q4, q5, hedge, market, n, n_q``.
    """
    signal = log_illiq(illiq) if use_log else illiq

    rows: dict[int, dict] = {}
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < n_quintiles * 5 or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < n_quintiles * 5:
            continue

        # Cross-sectional quintile labels (qcut = equal-count bins)
        try:
            labels = pd.qcut(s, q=n_quintiles, labels=False, duplicates="drop")
        except ValueError:
            continue  # not enough distinct values

        q_rets = {}
        for q in range(n_quintiles):
            mask = labels == q
            r = nxt[mask].mean()
            q_rets[f"q{q + 1}"] = float(r) if not np.isnan(r) else float("nan")

        hedge = q_rets.get("q5", float("nan")) - q_rets.get("q1", float("nan"))
        market = float(nxt.mean())
        n_q = int((labels == 0).sum())  # typical quintile size

        row = {**q_rets, "hedge": float(hedge), "market": float(market),
               "n": int(len(s)), "n_q": n_q}
        rows[int(y)] = row

    return pd.DataFrame(rows).T.sort_index()


# ---------------------------------------------------------------------------
# Random-quintile control
# ---------------------------------------------------------------------------
def random_hedge_returns(
    illiq: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quintiles: int = 5,
    n_draws: int = 500,
    seed: int = 140,
) -> pd.Series:
    """Null distribution of Q5 - Q1 hedge returns from random quintile assignment.

    For each year, randomly assign stocks to quintiles (same sizes as the ILLIQ sort)
    ``n_draws`` times and record the resulting top-minus-bottom return. Returns a flat
    Series of all (year x draw) hedge returns — the empirical null for the ILLIQ hedge.
    """
    rng = np.random.default_rng(seed)
    signal = log_illiq(illiq)

    excesses: list[float] = []
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < n_quintiles * 5 or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < n_quintiles * 5:
            continue

        # Determine actual quintile sizes to keep sizes comparable
        try:
            labels = pd.qcut(s, q=n_quintiles, labels=False, duplicates="drop")
        except ValueError:
            continue
        n_top = int((labels == (n_quintiles - 1)).sum())
        n_bot = int((labels == 0).sum())
        nxt_arr = nxt.to_numpy()
        all_idx = np.arange(len(nxt_arr))
        for _ in range(n_draws):
            top_idx = rng.choice(all_idx, size=n_top, replace=False)
            remaining = np.setdiff1d(all_idx, top_idx)
            bot_idx = rng.choice(remaining, size=n_bot, replace=False)
            h = float(nxt_arr[top_idx].mean() - nxt_arr[bot_idx].mean())
            excesses.append(h)

    return pd.Series(excesses, name="random_hedge")


# ---------------------------------------------------------------------------
# Summary statistics with HAC t-stat
# ---------------------------------------------------------------------------
def summarize(annual_series: pd.Series, label: str = "") -> dict:
    """Headline statistics for an annual return series (HAC t-stat, Sharpe, hit rate).

    Uses Newey-West HAC standard error — appropriate for short annual series where
    even one-lag autocorrelation can inflate naive t-stats.
    """
    r = pd.Series(annual_series).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    # Newey-West HAC t-stat
    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "label": label,
        "mean": mu,
        "vol": std,
        "sharpe": sr,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": max_dd,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Convenience: per-quintile summary table
# ---------------------------------------------------------------------------
def quintile_summary(results: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.DataFrame:
    """Mean return, vol, Sharpe, and HAC t-stat for each quintile and the hedge.

    ``results`` is the output of ``quintile_sort``; ``fwd_ret`` is the forward return panel
    used to build the equal-weight market return.
    """
    cols = [c for c in results.columns if c.startswith("q")] + ["hedge", "market"]
    rows = []
    for col in cols:
        if col not in results.columns:
            continue
        s = summarize(results[col], label=col)
        rows.append(s)
    return pd.DataFrame(rows).set_index("label")

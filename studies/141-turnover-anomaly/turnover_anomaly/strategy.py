"""Strategy for Study 141 (Turnover-Anomaly) — Datar-Naik-Radcliffe (1998) turnover signal.

The hypothesis: stocks with high share-turnover (annual trading volume / shares outstanding)
**underperform** stocks with low share-turnover in the following year. Turnover is interpreted
as a liquidity/attention proxy: high turnover may reflect overvaluation driven by short-term
traders, divergence of opinion (Miller 1977), or the "attention" bias (Barber & Odean 2008).

Our implementation:

1. **Signal**: annual turnover = sum(daily volume) / weighted-average diluted shares.
2. **Portfolio**: each year, sort into quintiles; the low-turnover quintile (Q1) is the long
   leg, the high-turnover quintile (Q5) is the short leg, vs the equal-weight market.
3. **Honest controls**:
   - Equal-weight market (all valid-signal tickers) — the passive benchmark.
   - Shuffled-label control: each year randomly reassign turnover ranks to tickers, then
     sort into the same quintile portfolios. If the real signal beats a shuffle, it carries
     genuine cross-sectional information. With n_shuffles=200 this gives a robust null.
4. **Inference**: Newey-West HAC t-stat on the annual hedge return (Q1 − Q5, or Q1 − market);
   a t-stat |t| < 2 is noise, per the inference bar.

**Reporting lag**: signal from fiscal year y predicts returns in calendar year y+1.

**Survivorship bias**: the EDGAR panel covers only current S&P 500 members; results are
upper bounds on any live edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Quintile portfolio construction
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quintiles: int = 5,
    min_stocks: int = 20,
) -> pd.DataFrame:
    """Annual long-bottom / short-top quintile portfolio returns.

    For each signal year y, stocks are sorted into ``n_quintiles`` bins by turnover.
    Quintile 1 = low turnover (the theoretical long leg), quintile n = high turnover
    (theoretical short leg). Returns a DataFrame with columns ``q1``..``qN``, ``market``
    (equal-weight all valid stocks), and ``hedge`` (q1 − market), indexed by signal year.

    ``min_stocks`` is the minimum number of valid-signal stocks required to form quintiles
    in a given year (years below threshold are dropped).
    """
    rows: dict[int, dict] = {}
    for y in signal.index:
        s = signal.loc[y].dropna()
        if y not in fwd_ret.index or len(s) < min_stocks:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < min_stocks:
            continue

        labels = pd.qcut(s, q=n_quintiles, labels=False, duplicates="drop")
        mkt = float(nxt.mean())

        row: dict = {"market": mkt, "n": int(len(s))}
        for q in range(n_quintiles):
            mask = labels == q
            row[f"q{q+1}"] = float(nxt[mask].mean()) if mask.any() else float("nan")

        # Hedge: low-turnover (q1) minus market
        if not np.isnan(row.get("q1", float("nan"))):
            row["hedge"] = row["q1"] - mkt
        else:
            row["hedge"] = float("nan")

        # Spread: low minus high (q1 - qN)
        qn_key = f"q{n_quintiles}"
        if not np.isnan(row.get(qn_key, float("nan"))):
            row["spread"] = row["q1"] - row[qn_key]
        else:
            row["spread"] = float("nan")

        rows[int(y)] = row

    return pd.DataFrame(rows).T.sort_index()


# ---------------------------------------------------------------------------
# Shuffled-label null control
# ---------------------------------------------------------------------------
def shuffled_control(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quintiles: int = 5,
    n_shuffles: int = 200,
    seed: int = 141,
    min_stocks: int = 20,
) -> pd.DataFrame:
    """Distribution of Q1-minus-market returns under a shuffled-label null.

    For each shuffle, the cross-sectional signal ranks are randomly permuted within each
    year (destroying the turnover information while preserving the cross-sectional return
    distribution). The Q1 portfolio and market are computed on the permuted ranks.
    Returns a DataFrame (n_shuffles × n_years) of Q1−market annual returns.

    If Q1 in the real data systematically outperforms Q1 under the shuffle, the signal
    carries genuine predictive content. If the two are indistinguishable, the signal is noise.
    """
    rng = np.random.default_rng(seed)
    hedge_matrix = []

    for _ in range(n_shuffles):
        # Shuffle the signal ranks cross-sectionally within each year
        sig_perm = signal.copy()
        for y in sig_perm.index:
            row = sig_perm.loc[y].dropna()
            if len(row) < min_stocks:
                continue
            perm_idx = rng.permutation(len(row))
            sig_perm.loc[y, row.index] = row.to_numpy()[perm_idx]

        res = quintile_returns(sig_perm, fwd_ret, n_quintiles=n_quintiles, min_stocks=min_stocks)
        if "hedge" in res.columns:
            hedge_matrix.append(res["hedge"].rename(_))

    if not hedge_matrix:
        return pd.DataFrame()
    return pd.concat(hedge_matrix, axis=1).T


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(annual_returns: pd.Series) -> dict:
    """Headline statistics for an annual return series.

    Returns trade count, hit rate, mean return, vol, Sharpe, and a HAC t-stat
    (Newey-West) — the inference-bar number that decides whether the edge is
    distinguishable from zero. Short annual series have limited power, so the
    Newey-West lag is capped at 1 for n < 20.
    """
    r = pd.Series(annual_returns).astype(float).dropna()
    n = int(len(r))
    if n < 2:
        return {k: float("nan") for k in ("n", "mean", "vol", "sharpe", "tstat", "hit_rate")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = float(mu / std) if std > 0 else float("nan")

    # Newey-West HAC t-stat — lag floor from rule of thumb, capped at n//3
    e = r.to_numpy() - mu
    lags = min(max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))), n // 3)
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    return {
        "n": n,
        "mean": mu,
        "vol": std,
        "sharpe": sr,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Turnover quintile profile (for notebooks)
# ---------------------------------------------------------------------------
def quintile_profile(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quintiles: int = 5,
    min_stocks: int = 20,
) -> pd.DataFrame:
    """Mean forward return and mean turnover by quintile, pooled across years.

    Returns a DataFrame indexed by quintile (1 = low, N = high) with columns
    ``mean_fwd_ret`` and ``mean_turnover``, for the diagnostic scatter plot.
    """
    fwd_by_q: dict[int, list[float]] = {q: [] for q in range(1, n_quintiles + 1)}
    to_by_q: dict[int, list[float]] = {q: [] for q in range(1, n_quintiles + 1)}

    for y in signal.index:
        s = signal.loc[y].dropna()
        if y not in fwd_ret.index or len(s) < min_stocks:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < min_stocks:
            continue
        labels = pd.qcut(s, q=n_quintiles, labels=False, duplicates="drop")
        for q in range(n_quintiles):
            mask = labels == q
            if mask.any():
                fwd_by_q[q + 1].extend(nxt[mask].tolist())
                to_by_q[q + 1].extend(s[mask].tolist())

    rows = {}
    for q in range(1, n_quintiles + 1):
        f = fwd_by_q[q]
        t = to_by_q[q]
        rows[q] = {
            "mean_fwd_ret": float(np.mean(f)) if f else float("nan"),
            "mean_turnover": float(np.mean(t)) if t else float("nan"),
            "n_obs": len(f),
        }
    return pd.DataFrame(rows).T.sort_index()

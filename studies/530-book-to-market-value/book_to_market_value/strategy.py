"""The signal and its honest controls -- Study 530 (Book-To-Market-Value).

The claim (Fama & French, 1992; the canonical HML value factor): sorting stocks by
**book-to-market** (book equity / market cap) and going long the high-B/M quintile
(cheap "value") versus the low-B/M quintile (expensive "growth") earns a positive
cross-sectional premium -- the value premium.

We implement, in order:

1. ``quintile_sort``    -- annual B/M quintile buckets; Q5 = highest B/M (value),
                          Q1 = lowest B/M (growth). hedge = Q5 - Q1 = long value / short growth.
2. ``summary``          -- annualised stats + a Newey-West HAC t-stat (the inference number).
3. ``placebo_pvalue``   -- a label-shuffle null: permute the B/M ranks within each year and
                          recompute the hedge mean; p = fraction of |shuffled| >= |real|.
4. ``apply_costs``      -- one-way bps x NAV x turnover plus borrow on the short leg; gross vs net.
5. ``synthetic_control``-- plant a known HML premium and confirm the engine recovers it
                          (a faithful-engine / power check, NEVER a real-tape claim).

No look-ahead: signal year Y uses fiscal-year-Y book equity (10-K public by ~Y+1 spring);
the forward return is calendar-year (Y+1). Exactly one execution lag -- the position is
entered for the full year that begins after the fundamentals are public. Survivorship bias
is named on every result; magnitudes are upper bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# One-way transaction cost (bps of traded notional) and annual borrow on the short leg.
DEFAULT_COST_BPS = 10.0     # 10 bps one-way -- generous for liquid large-caps
DEFAULT_BORROW_BPS = 50.0   # 50 bps/yr borrow on the short (growth) leg


# ---------------------------------------------------------------------------
# Core quintile sort: long value (high B/M) vs short growth (low B/M)
# ---------------------------------------------------------------------------
def quintile_sort(
    bm: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quintiles: int = 5,
    min_firms: int = 15,
) -> pd.DataFrame:
    """Annual book-to-market quintile sort against next-year returns.

    Convention: **high B/M -> Q5 (value, the long leg)**, low B/M -> Q1 (growth, the short
    leg). For each year with at least ``min_firms`` names present in both panels, firms are
    cut into ``n_quintiles`` equal-count buckets by B/M; each bucket's equal-weight forward
    return is recorded.

    Returns a year-indexed DataFrame with columns ``Q1..Q{n}``, ``hedge`` (= Q5 - Q1 =
    long value / short growth), ``market`` (equal-weight universe return), ``long`` (= Q5),
    ``short`` (= Q1), and ``n_firms``.
    """
    rows: dict[int, dict] = {}
    q_labels = [f"Q{i}" for i in range(1, n_quintiles + 1)]
    for y in bm.index:
        if y not in fwd_ret.index:
            continue
        s = bm.loc[y].dropna()
        r = fwd_ret.loc[y].dropna()
        common = s.index.intersection(r.index)
        if len(common) < min_firms:
            continue
        s_, r_ = s.loc[common], r.loc[common]
        try:
            qbins = pd.qcut(s_, q=n_quintiles, labels=q_labels)
        except ValueError:
            continue
        means = {q: float(r_[qbins == q].mean()) for q in q_labels}
        if any(np.isnan(v) for v in means.values()):
            continue
        means["long"] = means[q_labels[-1]]      # Q5 = value
        means["short"] = means[q_labels[0]]       # Q1 = growth
        means["hedge"] = means["long"] - means["short"]
        means["market"] = float(r_.mean())
        means["n_firms"] = int(len(common))
        rows[y] = means
    out = pd.DataFrame(rows).T.sort_index()
    out.index.name = "year"
    return out.astype(float)


# ---------------------------------------------------------------------------
# Annualised summary statistics with a HAC t-stat
# ---------------------------------------------------------------------------
def summary(annual_series: pd.Series) -> dict:
    """Annualised stats for a yearly return series.

    Returns mean, vol, Sharpe, a Newey-West HAC t-stat, hit-rate (fraction > 0), max
    drawdown, and ``n``. The HAC t-stat (lag rule ``floor(4*(n/100)^(2/9))``) is the
    inference-bar number. For a yearly hedge series the data are already annual, so the
    mean is the annual mean.
    """
    r = pd.Series(annual_series).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    sharpe = mu / sigma if sigma > 0 else np.nan

    e = r.to_numpy(dtype=float) - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, min(lags + 1, n)):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = mu / se if se > 0 else np.nan

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "mean": mu, "vol": sigma, "sharpe": sharpe, "tstat": tstat,
        "hit_rate": float((r > 0).mean()), "max_drawdown": max_dd, "n": int(n),
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    bm: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_quintiles: int = 5,
    min_firms: int = 15,
    n_shuffles: int = 2000,
    seed: int = 530,
) -> dict:
    """Label-shuffle placebo: is the real hedge mean distinguishable from noise?

    Each shuffle randomly permutes the B/M values *within each year* (destroying any real
    cross-sectional B/M->return link while preserving the marginal B/M and return
    distributions), then recomputes the mean hedge. The two-sided p-value is the fraction
    of shuffles whose |mean hedge| >= the real |mean hedge|.

    Returns ``{real_mean, p_value, n_shuffles, null_mean, null_std}``.
    """
    real = quintile_sort(bm, fwd_ret, n_quintiles, min_firms)
    if real.empty:
        return {"real_mean": np.nan, "p_value": np.nan, "n_shuffles": 0,
                "null_mean": np.nan, "null_std": np.nan}
    real_mean = float(real["hedge"].mean())

    rng = np.random.default_rng(seed)
    null_means = np.empty(n_shuffles)
    for s in range(n_shuffles):
        shuffled = bm.copy()
        for y in shuffled.index:
            row = shuffled.loc[y]
            valid = row.dropna().index
            perm = rng.permutation(row.loc[valid].to_numpy())
            shuffled.loc[y, valid] = perm
        res = quintile_sort(shuffled, fwd_ret, n_quintiles, min_firms)
        null_means[s] = float(res["hedge"].mean()) if not res.empty else np.nan

    null_means = null_means[~np.isnan(null_means)]
    if null_means.size == 0:
        return {"real_mean": real_mean, "p_value": np.nan, "n_shuffles": 0,
                "null_mean": np.nan, "null_std": np.nan}
    p = float((np.abs(null_means) >= abs(real_mean)).mean())
    return {
        "real_mean": real_mean, "p_value": p, "n_shuffles": int(null_means.size),
        "null_mean": float(null_means.mean()), "null_std": float(null_means.std(ddof=1)),
    }


# ---------------------------------------------------------------------------
# Costs: one-way bps x NAV x turnover + borrow on the short leg
# ---------------------------------------------------------------------------
def apply_costs(
    sort: pd.DataFrame,
    bm: pd.DataFrame,
    n_quintiles: int = 5,
    min_firms: int = 15,
    cost_bps: float = DEFAULT_COST_BPS,
    borrow_bps: float = DEFAULT_BORROW_BPS,
) -> dict:
    """Charge realistic costs against the gross hedge return.

    The long-value / short-growth hedge re-sorts every year. Turnover is the fraction of
    names that *leave* the long (Q5) or short (Q1) bucket year-over-year. The two-sided
    traded notional each rebalance is ``(long_turnover + short_turnover)`` x NAV; we charge
    ``cost_bps`` one-way on it. The short (growth) leg also pays ``borrow_bps``/yr.

    Returns ``{gross_mean, net_mean, avg_turnover, cost_drag, borrow_drag, gross_t, net_t}``.
    """
    q_labels = [f"Q{i}" for i in range(1, n_quintiles + 1)]
    prev_long: set[str] | None = None
    prev_short: set[str] | None = None
    turnovers: list[float] = []

    for y in sort.index:
        s = bm.loc[y].dropna() if y in bm.index else pd.Series(dtype=float)
        if s.empty:
            continue
        try:
            qbins = pd.qcut(s, q=n_quintiles, labels=q_labels)
        except ValueError:
            continue
        long_set = set(s[qbins == q_labels[-1]].index)
        short_set = set(s[qbins == q_labels[0]].index)
        if prev_long is not None and long_set and prev_long:
            lt = 1.0 - len(long_set & prev_long) / max(len(long_set), 1)
            st = 1.0 - len(short_set & prev_short) / max(len(short_set), 1)
            turnovers.append((lt + st) / 2.0)
        prev_long, prev_short = long_set, short_set

    avg_turnover = float(np.mean(turnovers)) if turnovers else 1.0
    gross = sort["hedge"]
    s_gross = summary(gross)

    # Cost drag (annual): both legs trade -> 2 * turnover * cost; charged each year.
    cost_drag = 2.0 * avg_turnover * (cost_bps / 1e4)
    borrow_drag = borrow_bps / 1e4  # short leg borrow, full year
    net = gross - cost_drag - borrow_drag
    s_net = summary(net)

    return {
        "gross_mean": s_gross["mean"], "net_mean": s_net["mean"],
        "gross_t": s_gross["tstat"], "net_t": s_net["tstat"],
        "avg_turnover": avg_turnover, "cost_drag": cost_drag, "borrow_drag": borrow_drag,
        "gross_sharpe": s_gross["sharpe"], "net_sharpe": s_net["sharpe"],
    }


# ---------------------------------------------------------------------------
# Synthetic positive control -- faithful-engine / power check ONLY
# ---------------------------------------------------------------------------
def synthetic_control(
    premiums: tuple[float, ...] = (-0.05, 0.0, 0.05, 0.10),
    n_firms: int = 200,
    n_years: int = 24,
    seed: int = 530,
) -> pd.DataFrame:
    """Plant a known HML premium and confirm the quintile engine recovers it.

    This is a power / faithfulness check of the sort -- it proves the engine finds a value
    premium when one exists and returns noise when it does not. It is NEVER cited to support
    a real-tape stamp. Returns a DataFrame ``[premium, hedge_mean_%, tstat]``.
    """
    from .data import synthetic_panel

    rows = []
    for prem in premiums:
        bm, fwd, _ = synthetic_panel(
            n_firms=n_firms, n_years=n_years, hml_premium=prem, seed=seed
        )
        res = quintile_sort(bm, fwd, n_quintiles=5, min_firms=20)
        s = summary(res["hedge"]) if not res.empty else {"mean": np.nan, "tstat": np.nan}
        rows.append({"premium": prem, "hedge_mean_%": s["mean"] * 100, "tstat": s["tstat"]})
    return pd.DataFrame(rows)

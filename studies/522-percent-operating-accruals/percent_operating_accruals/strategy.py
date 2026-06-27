"""Strategy for Study 522 (Percent Operating Accruals) — Hafzalla-Lundholm-Van Winkle 2011.

**The percent-accruals anomaly.** HLVW (2011) take Sloan's operating accrual
(Net Income - Operating Cash Flow) and scale it by the **magnitude of earnings** rather
than by total assets. Their claim: percent accruals — the *fraction* of earnings that is
accrual-based — sort future returns more sharply than the asset-scaled Sloan accrual,
because it ranks firms by how much of their reported profit is *not* backed by cash.

**Signal (HLVW 2011)**::

    operating accruals  = Net Income - Operating Cash Flow
    percent accruals    = (Net Income - Operating Cash Flow) / |Net Income|

    HIGH percent accruals => earnings mostly non-cash => LOWER future returns
    LOW  percent accruals => earnings cash-backed      => HIGHER future returns

We sort the survivor basket into quintiles each fiscal year and go **long the low-percent-
accruals quintile, short the high** (Q1 - Q5). Distinct from Study 231 (Sloan, asset-scaled).

**Honest controls.**

1. Equal-weight market: all names with valid fundamentals that year.
2. Random-portfolio null: same-size random subsets, to ask whether the long leg's edge
   exceeds what concentration alone would give.
3. Label-shuffle placebo: permute the percent-accruals labels *within each year*, breaking
   the signal-return link while preserving the marginal return distribution. A real edge
   should die under the shuffle.

**Costs.** One-way bps x NAV x turnover, plus borrow on the short leg. Annual rebalance,
so turnover is modest. Gross and net are both reported.

**Reporting lag.** Fundamentals from fiscal year y predict calendar-year y+1 returns — a
conservative lag (the 10-K is not actionable until well into the following year). Exactly
one execution lag, no same-bar fills, no look-ahead.

**Survivorship bias.** The basket is a fixed set of names still trading in 2026. Every
real-tape number is an upper bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Signal construction
# --------------------------------------------------------------------------- #
def percent_accruals_signal(fund: pd.DataFrame, min_ni: float = 1e6) -> pd.DataFrame:
    """Build the HLVW percent-operating-accruals ratio from a long fundamentals frame.

    ``fund`` has columns ``ticker, year, net_income, cfo``. Computes::

        percent accruals = (net_income - cfo) / |net_income|

    Firm-years with |net_income| below ``min_ni`` are masked (the ratio explodes for
    near-zero earnings — HLVW exclude tiny-denominator firms). Returns a (year x ticker)
    DataFrame of percent-accruals ratios (higher = more accrual / less cash backing).
    """
    df = fund.copy()
    ni = df["net_income"].to_numpy(dtype=float)
    cfo = df["cfo"].to_numpy(dtype=float)
    accr = ni - cfo
    denom = np.abs(ni)
    with np.errstate(divide="ignore", invalid="ignore"):
        pa = np.where(denom >= min_ni, accr / denom, np.nan)
    df["pa"] = pa
    panel = df.pivot_table(index="year", columns="ticker", values="pa")
    panel.index.name = "year"
    panel.columns.name = None
    return panel.sort_index()


# --------------------------------------------------------------------------- #
# Portfolio construction and returns
# --------------------------------------------------------------------------- #
def quintile_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    min_names: int = 15,
) -> pd.DataFrame:
    """Annual quintile-sorted returns on percent-accruals signal vs next-year returns.

    Q1 = lowest percent accruals (most cash-backed) — the long leg.
    Q5 = highest percent accruals (least cash-backed) — the short leg.
    ``hedge`` = Q1 - Q5 (long low, short high) is the HLVW prediction.

    Returns a DataFrame with columns ``q1..q5``, ``market`` (equal-weight all), ``n``,
    and ``hedge``, indexed by signal year.
    """
    rows: dict[int, dict] = {}
    n_q = int(round(1 / q))
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < min_names or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < min_names:
            continue

        ranks = s.rank(method="first")
        bins = np.ceil(ranks / len(s) * n_q).astype(int).clip(1, n_q)
        row: dict = {"market": float(nxt.mean()), "n": int(len(s))}
        qrets = []
        for qi in range(1, n_q + 1):
            mask = (bins == qi).values
            qr = float(nxt[mask].mean()) if mask.any() else np.nan
            qrets.append(qr)
            row[f"q{qi}"] = qr
        row["hedge"] = qrets[0] - qrets[-1]  # long low-accruals, short high-accruals
        rows[int(y)] = row

    return pd.DataFrame(rows).T.sort_index()


def turnover_series(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    min_names: int = 15,
) -> pd.Series:
    """One-sided name turnover of the Q1 (long) basket from year to year.

    Fraction of the long basket that is replaced at each annual rebalance. Used to charge
    transaction costs. The short basket turns over similarly; we apply the same fraction
    to both legs for the cost charge.
    """
    n_q = int(round(1 / q))
    prev_long: set | None = None
    out: dict[int, float] = {}
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < min_names or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, _ = s.align(nxt, join="inner")
        if len(s) < min_names:
            continue
        ranks = s.rank(method="first")
        bins = np.ceil(ranks / len(s) * n_q).astype(int).clip(1, n_q)
        long_names = set(s.index[(bins == 1).values])
        if prev_long is not None and len(prev_long) > 0:
            replaced = len(long_names - prev_long) / len(prev_long)
            out[int(y)] = float(replaced)
        prev_long = long_names
    return pd.Series(out, name="turnover").sort_index()


def apply_costs(
    hedge: pd.Series,
    turnover: pd.Series,
    one_way_bps: float = 10.0,
    borrow_bps: float = 50.0,
) -> pd.Series:
    """Net the hedge return for transaction costs and short-leg borrow.

    - Transaction cost per year = 2 legs x one_way_bps x turnover (both legs rebalance).
    - Borrow = borrow_bps/yr on the short leg (always carried, full NAV on the short side).

    Returns the net hedge series.
    """
    t = turnover.reindex(hedge.index).fillna(turnover.mean() if len(turnover) else 0.0)
    trade_cost = 2.0 * (one_way_bps / 1e4) * t
    borrow = borrow_bps / 1e4
    return (hedge - trade_cost - borrow).rename("hedge_net")


# --------------------------------------------------------------------------- #
# Null distributions
# --------------------------------------------------------------------------- #
def random_portfolio_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 500,
    min_names: int = 15,
    seed: int = 522,
) -> pd.Series:
    """Random-portfolio excess-vs-market null (long-leg size). One value per (year x draw)."""
    rng = np.random.default_rng(seed)
    excesses: list[float] = []
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < min_names or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < min_names:
            continue
        n_pick = max(1, int(round(len(s) * q)))
        ret_arr = nxt.to_numpy()
        mkt = ret_arr.mean()
        for _ in range(n_draws):
            pick = rng.choice(len(ret_arr), size=n_pick, replace=False)
            excesses.append(float(ret_arr[pick].mean() - mkt))
    return pd.Series(excesses, name="random_excess")


def placebo_hedge_t(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_perm: int = 1000,
    min_names: int = 15,
    seed: int = 522,
) -> tuple[float, np.ndarray]:
    """Label-shuffle placebo: permute percent-accruals labels within each year.

    Breaks the signal->return link while preserving each year's marginal return distribution.
    Returns ``(p_value, null_tstats)`` where the p-value is the two-sided fraction of
    permuted hedge t-stats whose |t| meets or beats the real |t|. A real edge should die.
    """
    real_t = summary(quintile_returns(signal, fwd_ret, q=q, min_names=min_names)["hedge"])["tstat"]
    rng = np.random.default_rng(seed)
    null_t: list[float] = []
    for _ in range(n_perm):
        shuf = signal.copy()
        for y in shuf.index:
            row = shuf.loc[y].dropna()
            if len(row) < min_names:
                continue
            perm = rng.permutation(row.values)
            shuf.loc[y, row.index] = perm
        h = quintile_returns(shuf, fwd_ret, q=q, min_names=min_names)
        if "hedge" not in h or h["hedge"].dropna().empty:
            continue
        null_t.append(summary(h["hedge"])["tstat"])
    null_arr = np.array([t for t in null_t if np.isfinite(t)])
    if null_arr.size == 0 or not np.isfinite(real_t):
        return float("nan"), null_arr
    p = float((np.abs(null_arr) >= abs(real_t)).mean())
    return p, null_arr


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
def summary(annual_returns: pd.Series) -> dict:
    """Headline stats for an annual return series (HAC Newey-West t-stat, Sharpe, hit rate)."""
    r = pd.Series(annual_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

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

    return {"mean": mu, "vol": std, "sharpe": sr, "tstat": tstat,
            "hit_rate": float((r > 0).mean()), "max_drawdown": max_dd, "n": n}


def market_annual(fwd_ret: pd.DataFrame, years: list[int] | None = None) -> pd.Series:
    """Equal-weight annual market return across all tickers in the panel."""
    if years is not None:
        fwd_ret = fwd_ret.reindex(years)
    return fwd_ret.mean(axis=1).rename("market").dropna()


# --------------------------------------------------------------------------- #
# Synthetic positive control wrapper
# --------------------------------------------------------------------------- #
def synthetic_hedge_t(premium: float, seed: int = 522) -> float:
    """HAC t-stat of the Q1-Q5 hedge on a synthetic panel with a known ``premium``.

    A faithful engine check: ``premium = 0`` should give |t| typically < 2; a strongly
    negative ``premium`` (high accruals underperform) should light the hedge up positive.
    NEVER cited to support a real-tape stamp — only proves the sort/stat machinery works.
    """
    from .data import synthetic_panel
    acc, fwd, _ = synthetic_panel(premium=premium, seed=seed)
    h = quintile_returns(acc, fwd, q=0.20, min_names=15)
    return summary(h["hedge"])["tstat"]

"""Strategy for Study 199 (Sales-Growth) — Lakonishok, Shleifer & Vishny (1994).

**The LSV glamour/value anomaly (sales-growth version)**: firms with high past
one-year sales growth ("glamour" stocks) should earn LOW future returns because
investors over-extrapolate past revenue growth trajectories. A firm that grew
revenues by 40% last year looks exciting; investors bid it up beyond its
fundamental value. When the growth rate inevitably reverts, the stock disappoints.

**Signal construction**::

    SalesGrowth_y = Revenues_y / Revenues_{y-1} - 1

Computed cross-sectionally for each fiscal year y, then quintile-sorted.
The LSV prediction: **Q1 (low growth / value) > Q5 (high growth / glamour)**.

**Honest controls**:

1. Equal-weight market: all firms with valid sales-growth data in that year.
2. Random portfolio of the same quintile size (seeded), to test whether any
   apparent outperformance is attributable to the signal or simply to holding
   a small concentrated subset of the universe.

**Reporting lag**: fundamentals from fiscal year y are used to predict
calendar-year y+1 returns — a conservative lag that assumes the 10-K is not
actionable until well into the following year.

**Survivorship bias**: the EDGAR panel is the *current* S&P 500 projected back.
All results are upper bounds; the true live edge is weaker.

**Distinct from Study 44 (Asset-Growth)**: this study uses *revenue* (top-line
sales) growth, testing LSV's overextrapolation of operating momentum. Asset growth
(Cooper et al. 2008) uses total-asset expansion, a different mechanism focused on
overinvestment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def sales_growth_signal(rev: pd.DataFrame) -> pd.DataFrame:
    """Build year-over-year sales growth from a (year x ticker) Revenues frame.

    For each fiscal year y:

        SalesGrowth_y = Revenues_y / Revenues_{y-1} - 1

    Returns a (year x ticker) DataFrame of trailing one-year revenue growth rates.
    Firms with non-positive prior-year revenues are left as NaN (undefined growth).
    Firms with missing data in either year are also NaN.
    """
    years = sorted(rev.index)
    rows: dict[int, pd.Series] = {}
    for yr in years:
        if yr - 1 not in rev.index:
            continue
        curr = rev.loc[yr]
        prev = rev.loc[yr - 1]
        # Require positive prior revenues (avoid division by zero or sign flip)
        mask = (prev > 0) & curr.notna() & prev.notna()
        growth = (curr / prev - 1).where(mask)
        rows[int(yr)] = growth

    if not rows:
        return pd.DataFrame()

    sg = pd.DataFrame(rows).T
    sg.index = pd.Index(sorted(rows.keys()), name="year")
    sg.columns.name = None
    return sg


# ---------------------------------------------------------------------------
# Portfolio construction and returns
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
) -> pd.DataFrame:
    """Annual quintile-sorted returns on (year x ticker) sales-growth signal.

    ``q = 0.20`` gives quintiles (five bins). Q1 contains the slowest-growing
    firms ("value"), Q5 the fastest-growing ("glamour"). The LSV prediction is
    Q1 > Q5, so the hedge column is ``q1 - q5`` (long value, short glamour).

    The ``fwd_ret`` frame must be indexed by the *signal* year y and contain the
    forward one-year return (calendar year y+1) — alignment is the caller's
    responsibility (see ``data.fetch_panel``).

    Returns a DataFrame with columns ``q1`` through ``q5``, ``market`` (equal-weight
    all valid firms in that year), and ``hedge`` (q1 - q5), indexed by signal year.
    """
    rows: dict[int, dict] = {}
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < 20 or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < 20:
            continue

        n_q = int(round(1 / q))
        thresholds = [s.quantile(i * q) for i in range(n_q + 1)]
        quintile_rets = []
        for qi in range(n_q):
            if qi == n_q - 1:
                mask = s >= thresholds[qi]
            else:
                mask = (s >= thresholds[qi]) & (s < thresholds[qi + 1])
            quintile_rets.append(float(nxt[mask].mean()))

        mkt = float(nxt.mean())
        row: dict = {"market": mkt, "n": int(len(s))}
        for qi, qr in enumerate(quintile_rets, start=1):
            row[f"q{qi}"] = qr
        row["hedge"] = quintile_rets[0] - quintile_rets[-1]  # long value, short glamour
        rows[int(y)] = row

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).T.sort_index()


def random_portfolio_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 500,
    seed: int = 199,
) -> pd.Series:
    """Empirical distribution of random-portfolio excess returns (quintile-portfolio size).

    For each year, draw ``n_draws`` random subsets of the same number of stocks as the
    bottom quintile (low-growth/value portfolio). Returns a Series of excess-return-vs-market
    across all (year x draw) combinations — the null distribution for the low-growth
    portfolio's edge.
    """
    rng = np.random.default_rng(seed)
    excesses: list[float] = []
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < 20 or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < 20:
            continue
        n_q = max(1, int(round(len(s) * q)))
        ret_arr = nxt.to_numpy()
        mkt_mean = ret_arr.mean()
        for _ in range(n_draws):
            pick = rng.choice(len(ret_arr), size=n_q, replace=False)
            excesses.append(float(ret_arr[pick].mean() - mkt_mean))
    return pd.Series(excesses, name="random_excess")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summary(annual_returns: pd.Series) -> dict:
    """Headline statistics for an annual return series (HAC t-stat, Sharpe, hit rate).

    Uses Newey-West HAC standard error for the t-stat — appropriate for short annual
    series where even one-lag autocorrelation can inflate naive t-stats.
    """
    r = pd.Series(annual_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}

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
        "mean": mu,
        "vol": std,
        "sharpe": sr,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": max_dd,
        "n": n,
    }


def market_annual(fwd_ret: pd.DataFrame, years: list[int] | None = None) -> pd.Series:
    """Equal-weight annual market return across all tickers in the panel."""
    if years is not None:
        fwd_ret = fwd_ret.reindex(years)
    return fwd_ret.mean(axis=1).rename("market").dropna()

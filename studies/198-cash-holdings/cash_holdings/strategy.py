"""Strategy for Study 198 (Cash-Holdings) — Palazzo (2012).

**The cash-holdings anomaly**: firms with high cash-to-assets ratios should earn
higher future returns because they face high external financing costs. Cash hoarding is a
signal of financial constraints, and constrained firms bear more systematic risk — hence
the premium. Palazzo (2012) documents roughly +3 to +5%/yr outperformance for cash-rich
firms in the CRSP universe.

**Signal construction**::

    Cash-to-Assets = CashAndCashEquivalentsAtCarryingValue / Total Assets

High Cash-to-Assets = cash-rich = Palazzo's financial-constraint proxy. The claim is
HIGH Cash-to-Assets → HIGH future returns.

**Honest controls**:

1. Equal-weight market: all firms with valid fundamentals in that year.
2. Random portfolio of the same quintile size (seeded), to test whether any apparent
   outperformance is attributable to the cash signal or simply to holding a small
   concentrated subset of the universe.

**Reporting lag**: fundamentals from fiscal year y are used to predict calendar-year
y+1 returns — a conservative lag that assumes the 10-K is not actionable until well
into the following year.

**Survivorship bias**: the EDGAR panel is the *current* S&P 500 projected backwards.
All results are upper bounds; the true live edge is weaker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def cash_ratio_signal(
    assets: pd.DataFrame,
    cash: pd.DataFrame,
) -> pd.DataFrame:
    """Build the Cash-to-Assets ratio from (year × ticker) concept frames.

    Cash-to-Assets = CashAndCashEquivalentsAtCarryingValue / Total Assets

    Returns a (year × ticker) DataFrame of ratios (higher = more cash-rich).
    Missing values where the denominator is non-positive are left as NaN.
    Ratios > 1 are clamped to NaN (data errors / non-operating entities).
    """
    # Align tickers across the two panels
    common = set(assets.columns) & set(cash.columns)
    tickers = sorted(common)

    a = assets.reindex(columns=tickers)
    c = cash.reindex(columns=tickers)

    # Valid years: those where both panels have data
    valid_years = sorted(set(a.index) & set(c.index))

    rows: dict[int, pd.Series] = {}
    for yr in valid_years:
        a_yr = a.loc[yr].to_numpy(dtype=float)
        c_yr = c.loc[yr].to_numpy(dtype=float)

        # Mask non-positive denominators
        mask = (a_yr > 0) & np.isfinite(a_yr) & np.isfinite(c_yr)
        ratio_arr = np.where(mask, c_yr / np.where(mask, a_yr, 1.0), np.nan)
        # Clamp implausible values (ratio > 1 suggests data errors)
        ratio_arr = np.where((ratio_arr >= 0) & (ratio_arr <= 1.0), ratio_arr, np.nan)
        rows[int(yr)] = pd.Series(ratio_arr, index=tickers)

    ratio = pd.DataFrame(rows).T
    ratio.index = pd.Index(sorted(rows.keys()), name="year")
    ratio.columns.name = None
    return ratio


# ---------------------------------------------------------------------------
# Portfolio construction and returns
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
) -> pd.DataFrame:
    """Annual quintile-sorted returns on (year × ticker) cash-ratio signal vs next-year returns.

    ``q = 0.20`` gives quintiles. The fifth quintile (highest cash) should outperform
    the first (lowest cash / least cash-rich) if the anomaly is real (Palazzo 2012).

    Returns a DataFrame with columns ``q1`` through ``q5``, ``market`` (equal-weight all),
    and ``hedge`` (q5 − q1, long high-cash / short low-cash), indexed by the signal year.
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

        thresholds = [s.quantile(i * q) for i in range(int(1 / q) + 1)]
        n_q = int(round(1 / q))
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
        # Hedge: long HIGH cash (q5) / short LOW cash (q1) — the Palazzo claim
        row["hedge"] = quintile_rets[-1] - quintile_rets[0]
        rows[int(y)] = row

    return pd.DataFrame(rows).T.sort_index()


def random_portfolio_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 500,
    seed: int = 198,
) -> pd.Series:
    """Empirical distribution of random-portfolio excess returns (quintile-portfolio size).

    For each year, draw ``n_draws`` random subsets of the same number of stocks as the
    top quintile (high-cash portfolio). Returns a Series of excess-return-vs-market across
    all (year × draw) combinations — the null distribution for the high-cash portfolio's edge.
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
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n", "max_drawdown")}

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

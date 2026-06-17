"""Strategy for Study 231 (Sloan Accruals) — Sloan (1996).

**The accruals anomaly**: firms whose earnings are backed by accounting accruals rather
than real cash flows earn LOW future returns because earnings quality is low and investors
fail to discount the transitory component. A firm with high accruals reports earnings
that exceed its actual cash generation — the gap will eventually be corrected by
lower future earnings and disappointing stock performance.

**Signal construction (Sloan 1996, cash-flow statement version)**::

    Accruals       = Net Income - Operating Cash Flow
    Avg Assets     = (Assets_t + Assets_{t-1}) / 2
    Accruals_ratio = Accruals / Avg Assets

    HIGH ratio => earnings are accrual-heavy (low quality) => LOWER future returns
    LOW  ratio => earnings are cash-backed (high quality) => HIGHER future returns

The cash-flow statement version is cleaner than the original balance-sheet version
(which inferred accruals from working-capital changes) because SFAS 95 mandates direct
disclosure of operating cash flows since 1988.

**Honest controls**:

1. Equal-weight market: all firms with valid fundamentals in that year.
2. Random portfolio of the same quintile size (seeded), to test whether any apparent
   outperformance is attributable to the accruals signal or simply to concentration risk.

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
def accruals_signal(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the Sloan accruals ratio from a dict of (year x ticker) concept frames.

    Steps:
    1. Accruals       = Net Income - Operating Cash Flow
    2. Avg Assets     = (Assets_t + Assets_{t-1}) / 2
    3. Accruals_ratio = Accruals / Avg Assets

    Returns a (year x ticker) DataFrame of accruals ratios (higher = more accrual /
    less cash backing). Missing values where the average assets denominator is
    non-positive are left as NaN.
    """
    ni = panels["NetIncomeLoss"]
    cfo = panels["NetCashProvidedByUsedInOperatingActivities"]
    assets = panels["Assets"]

    # Align tickers across the three panels
    common = set(ni.columns) & set(cfo.columns) & set(assets.columns)
    tickers = sorted(common)

    ni = ni.reindex(columns=tickers)
    cfo = cfo.reindex(columns=tickers)
    assets = assets.reindex(columns=tickers)

    # Valid years: those where all three panels have data AND year-1 assets exist
    valid_years = sorted(
        set(ni.index) & set(cfo.index) & set(assets.index)
    )

    rows: dict[int, pd.Series] = {}
    for yr in valid_years:
        if yr - 1 not in assets.index:
            continue
        accruals_raw = ni.loc[yr] - cfo.loc[yr]
        avg_assets = (assets.loc[yr] + assets.loc[yr - 1]) / 2.0
        # Scale by average total assets; mask non-positive denominators
        mask = (avg_assets > 0).values
        acc_arr = accruals_raw.to_numpy(dtype=float)
        avg_arr = avg_assets.to_numpy(dtype=float)
        acc_scaled = np.where(mask, acc_arr / np.where(mask, avg_arr, 1.0), np.nan)
        rows[int(yr)] = pd.Series(acc_scaled, index=tickers)

    acc = pd.DataFrame(rows).T
    acc.index = pd.Index(sorted(rows.keys()), name="year")
    acc.columns.name = None
    return acc


# ---------------------------------------------------------------------------
# Portfolio construction and returns
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
) -> pd.DataFrame:
    """Annual quintile-sorted returns on (year x ticker) accruals signal vs next-year returns.

    ``q = 0.20`` gives quintiles. The first quintile (lowest accruals / highest cash
    backing) should outperform the fifth (highest accruals / lowest cash backing) if
    the Sloan anomaly is real.

    Returns a DataFrame with columns ``q1`` through ``q5``, ``market`` (equal-weight all),
    and ``hedge`` (q1 - q5, long low-accruals / short high-accruals), indexed by signal year.
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
        row["hedge"] = quintile_rets[0] - quintile_rets[-1]  # long low-acc, short high-acc
        rows[int(y)] = row

    return pd.DataFrame(rows).T.sort_index()


def random_portfolio_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 500,
    seed: int = 231,
) -> pd.Series:
    """Empirical distribution of random-portfolio excess returns (quintile-portfolio size).

    For each year, draw ``n_draws`` random subsets of the same number of stocks as the
    bottom quintile (low-accruals portfolio). Returns a Series of excess-return-vs-market
    across all (year x draw) combinations — the null distribution for the low-accruals
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
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")}

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

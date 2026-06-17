"""Strategy for Study 244 (Asset-Growth) — Cooper, Gulen & Schill (2008).

**The asset-growth anomaly**: firms with high total-asset growth (aggressive expansion)
should earn LOW future returns because managers over-invest when capital is cheap and
markets extrapolate past investment growth. The paper documents a robust cross-sectional
negative relationship between YoY total-asset growth and subsequent 12-month returns.

**Signal construction (Cooper, Gulen & Schill 2008)**::

    AG_t = (Total_Assets_t - Total_Assets_{t-1}) / Total_Assets_{t-1}

Higher AG = more aggressive asset expansion relative to prior year.
The paper predicts HIGH AG → LOW next-year return.

**Honest controls**:

1. Equal-weight market: all firms with valid fundamentals in that year.
2. Random portfolio of the same quintile size (seeded), to test whether any apparent
   outperformance is attributable to the AG signal or simply to holding a small
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
def ag_signal(assets: pd.DataFrame) -> pd.DataFrame:
    """Build the YoY total-asset growth rate from a (year × ticker) assets frame.

    Steps:
    1. Compute YoY growth: AG_t = (Assets_t - Assets_{t-1}) / Assets_{t-1}
    2. Require positive prior-year assets (mask non-positive denominators → NaN).

    Returns a (year × ticker) DataFrame of asset-growth rates (higher = more expansion).
    Missing values where the lagged denominator is non-positive are left as NaN.
    """
    tickers = sorted(assets.columns)
    a = assets.reindex(columns=tickers)

    valid_years = sorted(a.index)
    rows: dict[int, pd.Series] = {}
    for yr in valid_years:
        if yr - 1 not in a.index:
            continue
        cur = a.loc[yr].to_numpy(dtype=float)
        lag = a.loc[yr - 1].to_numpy(dtype=float)
        # Mask non-positive denominators
        mask = lag > 0
        ag_arr = np.where(mask, (cur - lag) / np.where(mask, lag, 1.0), np.nan)
        rows[int(yr)] = pd.Series(ag_arr, index=tickers)

    ag = pd.DataFrame(rows).T
    ag.index = pd.Index(sorted(rows.keys()), name="year")
    ag.columns.name = None
    return ag


# ---------------------------------------------------------------------------
# Portfolio construction and returns
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
) -> pd.DataFrame:
    """Annual quintile-sorted returns on (year × ticker) AG signal vs next-year returns.

    ``q = 0.20`` gives quintiles. The first quintile (lowest AG / slow growers) should
    outperform the fifth (highest AG / fast growers) if the anomaly is real.

    Returns a DataFrame with columns ``q1`` through ``q5``, ``market`` (equal-weight all),
    and ``hedge`` (q1 − q5, long low-AG / short high-AG), indexed by the signal year.
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
        row["hedge"] = quintile_rets[0] - quintile_rets[-1]  # long low-AG, short high-AG
        rows[int(y)] = row

    return pd.DataFrame(rows).T.sort_index()


def random_portfolio_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 500,
    seed: int = 244,
) -> pd.Series:
    """Empirical distribution of random-portfolio excess returns (quintile-portfolio size).

    For each year, draw ``n_draws`` random subsets of the same number of stocks as the
    bottom quintile (low-AG portfolio). Returns a Series of excess-return-vs-market across
    all (year × draw) combinations — the null distribution for the low-AG portfolio's edge.
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

"""Strategy for Study 121 (Magic-Formula) — Greenblatt's combined quality + cheapness rank.

The Magic Formula (Greenblatt 2005) ranks every stock on two axes:

- **Return on Capital (ROC)**: EBIT / Invested Capital, where Invested Capital ≈
  Net Working Capital + Net Fixed Assets ≈ Assets − LiabilitiesCurrent − Cash. Higher
  ROC means the business earns more per dollar of capital deployed (quality).
- **Earnings Yield (EY)**: EBIT / Enterprise Value, where EV ≈ StockholdersEquity +
  LongTermDebtNoncurrent − Cash (book-value EV proxy since market-cap data is not
  directly available in the shared EDGAR cache). Higher EY means you are paying less
  per dollar of earnings (cheapness).

Each firm is ranked on each axis; the ranks are summed (lower sum = worse, higher sum =
better). The top decile (best combined rank) is the long portfolio, tested against
(a) the equal-weight market and (b) a random portfolio of identical size.

**Honest controls**:

1. Equal-weight market: all firms with valid fundamentals in that year.
2. Random portfolio of the same decile size (seeded), to test whether any apparent
   outperformance is attributable to the rank signal or simply to holding a small
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
def magic_formula_rank(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the combined Magic-Formula rank from a dict of (year × ticker) concept frames.

    Steps:
    1. Compute ROC = EBIT / (Assets − LiabilitiesCurrent − Cash), require IC > 0.
    2. Compute EY  = EBIT / (Equity + LTD − Cash), require EV > 0.
    3. Cross-sectionally rank each metric (ascending, higher = better), sum the two ranks.

    Returns a (year × ticker) DataFrame of combined ranks (higher = better magic-formula stock).
    Missing values where either metric is undefined are left as NaN.
    """
    ebit = panels["OperatingIncomeLoss"]
    assets = panels["Assets"]
    liab_c = panels["LiabilitiesCurrent"]
    ltd = panels["LongTermDebtNoncurrent"]
    cash = panels["CashAndCashEquivalentsAtCarryingValue"]
    equity = panels["StockholdersEquity"]

    # Align tickers
    common = (
        set(ebit.columns)
        & set(assets.columns)
        & set(liab_c.columns)
        & set(ltd.columns)
        & set(cash.columns)
        & set(equity.columns)
    )
    tickers = sorted(common)

    eb = ebit.reindex(columns=tickers)
    a = assets.reindex(columns=tickers)
    lc = liab_c.reindex(columns=tickers)
    d = ltd.reindex(columns=tickers)
    c = cash.reindex(columns=tickers)
    eq = equity.reindex(columns=tickers)

    # Invested Capital (proxy) — must be positive for ROC to be meaningful
    ic = a - lc - c
    ic_safe = ic.where(ic > 0)  # negative IC is ambiguous; mask out

    # Enterprise Value (book proxy) — must be positive
    ev = eq + d - c
    ev_safe = ev.where(ev > 0)

    roc = eb / ic_safe
    ey = eb / ev_safe

    # Cross-sectional rank (ascending — highest = best rank)
    roc_rank = roc.rank(axis=1, ascending=True, na_option="keep")
    ey_rank = ey.rank(axis=1, ascending=True, na_option="keep")

    combined = roc_rank + ey_rank  # higher combined = better MF stock
    combined.index.name = "year"
    return combined


# ---------------------------------------------------------------------------
# Portfolio construction and returns
# ---------------------------------------------------------------------------
def quantile_hedge(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.10,
) -> pd.DataFrame:
    """Annual long-top / short-bottom decile hedge on (year × ticker) signal vs next-year returns.

    ``q`` is the tail fraction (default 0.10 = deciles). Returns a DataFrame with columns
    ``top``, ``bottom``, ``market`` (equal-weight all), and ``hedge`` (top − market), indexed
    by the signal year.
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

        hi_thresh = s.quantile(1.0 - q)
        lo_thresh = s.quantile(q)
        top_ret = nxt[s >= hi_thresh].mean()
        bot_ret = nxt[s <= lo_thresh].mean()
        mkt_ret = nxt.mean()

        if np.isnan(top_ret) or np.isnan(mkt_ret):
            continue
        rows[int(y)] = {
            "top": float(top_ret),
            "bottom": float(bot_ret) if not np.isnan(bot_ret) else float("nan"),
            "market": float(mkt_ret),
            "hedge": float(top_ret - mkt_ret),
            "n": int(len(s)),
            "n_top": int((s >= hi_thresh).sum()),
            "n_bot": int((s <= lo_thresh).sum()),
        }
    return pd.DataFrame(rows).T.sort_index()


def random_portfolio_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.10,
    n_draws: int = 500,
    seed: int = 121,
) -> pd.Series:
    """Empirical distribution of random-portfolio excess returns (top-decile size).

    For each year, draw ``n_draws`` random subsets of the same number of stocks as the
    top decile. Returns a Series of excess-return-vs-market across all (year × draw)
    combinations — the null distribution for the top-decile's edge.
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
        n_top = max(1, int(round(len(s) * q)))
        mkt = nxt.to_numpy()
        mkt_mean = mkt.mean()
        all_idx = np.arange(len(mkt))
        for _ in range(n_draws):
            pick = rng.choice(all_idx, size=n_top, replace=False)
            excesses.append(float(mkt[pick].mean() - mkt_mean))
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

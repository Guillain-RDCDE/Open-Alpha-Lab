"""Strategy and honest controls — Study 155 (Asset-Turnover).

The factor recipe: sort S&P 500 names on Asset Turnover (AT = Revenues / Assets, the
efficiency leg of the DuPont identity) from fiscal year y; go long the top quintile,
measure calendar-year y+1 returns.  The hypothesis is that high-AT (capital-efficient)
firms earn a persistent premium.

Two controls decide whether any premium is attributable to AT or just to something correlated:

1. **Equal-weight market** — all firms with valid fundamentals in the same year.  The
   question is whether AT top-quintile beats the universe, not just whether it earns a
   positive return (any long-only portfolio in a rising market earns a positive return).

2. **ROA comparison** — the same quintile sort on NetIncomeLoss / Assets (profitability)
   run head-to-head.  If AT and ROA deliver the same excess return, AT's DuPont story
   adds nothing beyond plain profitability.

**Reporting lag**: fundamentals from fiscal year y → returns in calendar year y+1.  This
is a conservative lag: even a December 31 fiscal year-end requires a 10-K within 60–90
days (i.e. by April), so using full-year y+1 gives at least 9 months for public
dissemination.

**Survivorship bias**: the EDGAR panel is the *current* S&P 500 projected backwards.
Positive results are upper bounds on the true live effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Portfolio construction and returns
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
) -> pd.DataFrame:
    """Annual top / bottom quintile returns and the equal-weight market, aligned to signal year.

    For each year with enough observations (>= 20 stocks with valid signal and forward
    returns), forms the top-q and bottom-q quantile portfolios and the equal-weight
    market.  Returns a DataFrame with columns ``top``, ``bottom``, ``market``, ``hedge``
    (top − market), and ``ls`` (top − bottom), indexed by the signal year.
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
            "ls": float(top_ret - (bot_ret if not np.isnan(bot_ret) else mkt_ret)),
            "n": int(len(s)),
            "n_top": int((s >= hi_thresh).sum()),
            "n_bot": int((s <= lo_thresh).sum()),
        }
    return pd.DataFrame(rows).T.sort_index()


def random_portfolio_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 500,
    seed: int = 155,
) -> pd.Series:
    """Empirical null distribution of random-portfolio excess returns (same size as top quintile).

    For each year, draw ``n_draws`` random subsets of the same number of stocks as the
    top quintile.  Returns a Series of (random subset mean − market mean) across all
    (year × draw) combinations — the null for "does any random concentrated portfolio of
    this size beat the market?".
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
        returns_arr = nxt.to_numpy()
        mkt_mean = returns_arr.mean()
        idx_all = np.arange(len(returns_arr))
        for _ in range(n_draws):
            pick = rng.choice(idx_all, size=n_top, replace=False)
            excesses.append(float(returns_arr[pick].mean() - mkt_mean))
    return pd.Series(excesses, name="random_excess")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(annual_returns: pd.Series) -> dict:
    """Headline statistics for an annual return series with HAC t-stat.

    Uses Newey-West long-run variance for the t-stat — appropriate for short annual
    panels where even one-lag autocorrelation can inflate naive OLS standard errors.
    Returns mean, vol, Sharpe, HAC t-stat, hit rate (fraction of positive years), max
    drawdown of a (1+r) cumulative product, and n.
    """
    r = pd.Series(annual_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate",
                                    "max_drawdown", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    # Newey-West HAC t-stat (Bartlett kernel, rule-of-thumb lag selection)
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

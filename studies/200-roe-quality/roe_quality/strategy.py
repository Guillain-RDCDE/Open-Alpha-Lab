"""Strategy for Study 200 (ROE-Quality) — return on equity as a quality factor.

**The quality story**: high return-on-equity firms reinvest capital efficiently and
should earn higher forward returns — either because they are genuinely higher-quality
businesses, or because quality is a compensated risk factor (similar to the AQR
Quality-Minus-Junk factor). The opposite of "junk": firms with low ROE, negative
earnings, and weak balance sheets should underperform.

**Signal construction**::

    ROE_y = NetIncomeLoss_y / StockholdersEquity_{y-1}

Winsorised cross-sectionally at the 1st / 99th percentile per year to dampen the effect
of near-zero equity book values. Firms with non-positive lagged book equity excluded.

**Gross profitability comparison (Novy-Marx 2013)**::

    GP_ratio_y = GrossProfit_y / Assets_y

Gross profit scaled by total assets is theoretically a cleaner measure of economic
profitability — closer to "cash earnings" and harder to manage. We include it as a
head-to-head baseline; if GP/Assets beats ROE on the same universe and horizon, ROE's
apparent signal may simply be a noisier version of the underlying quality premium.

**Honest controls**:

1. Equal-weight market: all firms with valid fundamentals in that year.
2. Random portfolio of the same quintile size (seeded), to test whether any apparent
   outperformance is attributable to the ROE signal or simply to holding a small
   concentrated subset of the universe.

**Reporting lag**: fundamentals from fiscal year y are used to predict calendar-year
y+1 returns — a conservative lag that assumes the 10-K is not actionable until well
into the following year.

**Survivorship bias**: the EDGAR panel is the *current* S&P 500 projected backwards.
All results are upper bounds in magnitude; the true live edge is weaker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def roe_signal(
    ni: pd.DataFrame,
    se: pd.DataFrame,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """Build the cross-sectionally winsorised ROE from (year × ticker) concept frames.

    Steps:
    1. For each fiscal year y, compute NI_y / SE_{y-1} (lagged equity denominator).
    2. Exclude firms with non-positive lagged book equity (ROE is undefined or
       economically misleading for deeply distressed firms with negative equity).
    3. Winsorise each year's cross-section at ``winsor_pct`` / ``1 - winsor_pct``
       to reduce leverage from near-zero denominators.

    Returns a (year × ticker) DataFrame of winsorised ROE (higher = higher quality).
    Missing values where the lagged denominator is non-positive are left as NaN.
    """
    # Align tickers across both panels
    common = sorted(set(ni.columns) & set(se.columns))
    ni_c = ni.reindex(columns=common)
    se_c = se.reindex(columns=common)

    valid_years = sorted(set(ni_c.index) & set(se_c.index))

    rows: dict[int, pd.Series] = {}
    for yr in valid_years:
        if yr - 1 not in se_c.index:
            continue
        ni_y = ni_c.loc[yr]
        se_lag = se_c.loc[yr - 1]

        # Exclude non-positive lagged equity
        valid_se = se_lag.where(se_lag > 0)
        roe_raw = ni_y / valid_se  # NaN where equity <= 0

        # Remove infinite values (edge case: equity very close to zero, missed by > 0 check)
        roe_raw = roe_raw.replace([np.inf, -np.inf], np.nan)

        # Cross-sectional winsorisation
        lo = roe_raw.quantile(winsor_pct)
        hi = roe_raw.quantile(1.0 - winsor_pct)
        rows[int(yr)] = roe_raw.clip(lo, hi)

    roe = pd.DataFrame(rows).T
    roe.index = pd.Index(sorted(rows.keys()), name="year")
    roe.columns.name = None
    return roe


def gp_signal(
    gp: pd.DataFrame,
    assets: pd.DataFrame,
    winsor_pct: float = 0.01,
) -> pd.DataFrame:
    """Build cross-sectionally winsorised Gross Profitability = GrossProfit / Assets.

    Novy-Marx (2013) defines this as GrossProfit / (TotalAssets). We winsorise at the
    same percentile as ROE for a level playing field in the head-to-head comparison.

    Returns a (year × ticker) DataFrame of GP/Assets (higher = more profitable).
    """
    common = sorted(set(gp.columns) & set(assets.columns))
    gp_c = gp.reindex(columns=common)
    assets_c = assets.reindex(columns=common)

    valid_years = sorted(set(gp_c.index) & set(assets_c.index))

    rows: dict[int, pd.Series] = {}
    for yr in valid_years:
        gp_y = gp_c.loc[yr]
        assets_y = assets_c.loc[yr]

        # Exclude firms with non-positive assets (meaningless denominator)
        valid_assets = assets_y.where(assets_y > 0)
        ratio = (gp_y / valid_assets).replace([np.inf, -np.inf], np.nan)

        lo = ratio.quantile(winsor_pct)
        hi = ratio.quantile(1.0 - winsor_pct)
        rows[int(yr)] = ratio.clip(lo, hi)

    gpr = pd.DataFrame(rows).T
    gpr.index = pd.Index(sorted(rows.keys()), name="year")
    gpr.columns.name = None
    return gpr


# ---------------------------------------------------------------------------
# Portfolio construction and returns
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
) -> pd.DataFrame:
    """Annual quintile-sorted returns on a (year × ticker) signal vs next-year returns.

    ``q = 0.20`` gives quintiles. The fifth quintile (highest ROE) should outperform
    the first (lowest ROE) if the quality premium is real.

    Returns a DataFrame with columns ``q1`` through ``q5``, ``market`` (equal-weight all),
    and ``hedge`` (q5 − q1, long high-ROE / short low-ROE), indexed by the signal year.
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
        row["hedge"] = quintile_rets[-1] - quintile_rets[0]  # long high-ROE, short low-ROE
        rows[int(y)] = row

    return pd.DataFrame(rows).T.sort_index()


def random_portfolio_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 500,
    seed: int = 200,
) -> pd.Series:
    """Empirical distribution of random-portfolio excess returns (quintile-portfolio size).

    For each year, draw ``n_draws`` random subsets of the same number of stocks as the
    top quintile (high-ROE portfolio). Returns a Series of excess-return-vs-market across
    all (year × draw) combinations — the null distribution for the high-ROE portfolio's edge.
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
    se_hac = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se_hac) if se_hac > 0 else float("nan")

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

"""Strategy for Study 529 (Inventory-Growth) — Belo & Lin (2012); Thomas & Zhang (2002).

**The inventory-growth anomaly**: firms that build inventory aggressively earn LOW
future returns. Thomas & Zhang (2002) show abnormal inventory changes negatively predict
returns; Belo & Lin (2012) embed inventory growth in a production-based asset-pricing
model and document a low-minus-high inventory-growth premium.

**Signal construction (change scaled by lagged assets — Thomas & Zhang 2002)**::

    INVG_t = (Inventory_t - Inventory_{t-1}) / Total_Assets_{t-1}

Higher INVG = more aggressive inventory build. The literature predicts HIGH INVG ->
LOW next-year return, so the tradable book is **long low-INVG, short high-INVG**.

**Execution discipline**:

- ONE reporting lag: fiscal-year-y inventory growth predicts calendar-year y+1 returns
  (the annual report is assumed not actionable until the following year). No look-ahead,
  no same-bar fill.
- Quintile sort each year; the long-short hedge is Q1 (low INVG) minus Q5 (high INVG).
- Costs: one-way bps x turnover (annual quintile rebalance turns over both legs), plus a
  borrow charge on the short leg. Gross and net are reported separately.

**Honesty controls**:

1. Equal-weight market of all valid names that year.
2. Placebo / label-shuffle null: permute the INVG labels within each year and re-run the
   hedge; the real hedge must beat this shuffled null.
3. Random same-size portfolio: does the low-INVG basket beat a blind draw of the same
   number of names?

**Survivorship bias**: the basket is inventory-heavy names still trading in 2026. Results
are upper bounds; failed over-builders are absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_BPS = 5.0   # one-way transaction cost per leg, in bps of notional
BORROW_BPS = 50.0   # annual borrow on the short leg, in bps


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def invg_signal(bs_long: pd.DataFrame) -> pd.DataFrame:
    """Build the inventory-growth signal from a long balance-sheet frame.

    ``bs_long`` columns: ``[ticker, fiscal_year, inventory, total_assets]``.

    INVG_t = (Inventory_t - Inventory_{t-1}) / Total_Assets_{t-1}, computed per ticker
    over consecutive fiscal years. Returns a (year x ticker) DataFrame of INVG signals
    (higher = more aggressive inventory build). Non-positive lagged assets -> NaN.
    """
    if bs_long.empty:
        return pd.DataFrame()

    inv = bs_long.pivot_table(index="fiscal_year", columns="ticker", values="inventory")
    ta = bs_long.pivot_table(index="fiscal_year", columns="ticker", values="total_assets")
    inv = inv.sort_index()
    ta = ta.sort_index()

    inv_chg = inv.diff()
    ta_lag = ta.shift(1)
    ta_lag = ta_lag.where(ta_lag > 0)
    invg = inv_chg / ta_lag
    invg = invg.dropna(how="all")
    invg.index.name = "year"
    invg.columns.name = None
    return invg


# ---------------------------------------------------------------------------
# Portfolio construction and returns
# ---------------------------------------------------------------------------
def quintile_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    min_names: int = 10,
) -> pd.DataFrame:
    """Annual quintile-sorted returns on (year x ticker) INVG signal vs next-year returns.

    Q1 = lowest INVG (disciplined inventory) should outperform Q5 = highest INVG (aggressive
    build) if the anomaly is real. Returns a DataFrame with columns ``q1``..``q5``,
    ``market`` (equal-weight all valid names), ``hedge`` (q1 - q5, long low / short high),
    and ``n`` (valid names that year), indexed by the signal year.

    For a thin basket, ``q=0.20`` may put very few names per quintile; the function uses at
    least one name per extreme bucket and tolerates years with as few as ``min_names``.
    """
    rows: dict[int, dict] = {}
    for y in signal.index:
        s = signal.loc[y].dropna()
        if len(s) < min_names or y not in fwd_ret.index:
            continue
        nxt = fwd_ret.loc[y].dropna()
        s, nxt = s.align(nxt, join="inner")
        if len(s) < min_names:
            continue

        n_q = int(round(1 / q))
        thresholds = [s.quantile(i * q) for i in range(n_q + 1)]
        quintile_rets = []
        for qi in range(n_q):
            if qi == n_q - 1:
                mask = s >= thresholds[qi]
            else:
                mask = (s >= thresholds[qi]) & (s < thresholds[qi + 1])
            vals = nxt[mask]
            quintile_rets.append(float(vals.mean()) if len(vals) else np.nan)

        mkt = float(nxt.mean())
        row: dict = {"market": mkt, "n": int(len(s))}
        for qi, qr in enumerate(quintile_rets, start=1):
            row[f"q{qi}"] = qr
        row["hedge"] = quintile_rets[0] - quintile_rets[-1]  # long low-INVG, short high-INVG
        rows[int(y)] = row

    out = pd.DataFrame(rows).T.sort_index()
    return out


def net_hedge(
    gross_hedge: pd.Series,
    trading_bps: float = TRADING_BPS,
    borrow_bps: float = BORROW_BPS,
    legs_turned: float = 2.0,
    turnover: float = 1.0,
) -> pd.Series:
    """Net the annual hedge return for trading cost + short borrow.

    A full annual quintile rebalance turns over both the long and short legs (``legs_turned``
    round-trips' worth of one-way cost) at ``trading_bps`` each, scaled by ``turnover`` (the
    fraction of the book actually replaced — 1.0 for a worst-case full rebuild). The short
    leg also pays ``borrow_bps`` per year. Returns the cost-adjusted annual hedge series.
    """
    cost = (legs_turned * turnover * trading_bps + borrow_bps) / 1e4
    return gross_hedge - cost


def placebo_hedge_t(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    min_names: int = 10,
    n_shuffles: int = 1000,
    seed: int = 529,
) -> tuple[float, np.ndarray]:
    """Label-shuffle null for the hedge mean.

    Each iteration permutes the INVG labels *within each year* (destroying any real
    relationship between the signal and forward returns while preserving the cross-sectional
    return distribution), re-runs the quintile hedge, and records its mean. Returns
    ``(p_value, null_means)`` where ``p_value`` is the one-sided fraction of shuffled means
    at least as large as the real hedge mean (small p = the real hedge is unusual under the
    null).
    """
    real = quintile_returns(signal, fwd_ret, q=q, min_names=min_names)
    real_mean = float(real["hedge"].mean())

    rng = np.random.default_rng(seed)
    null_means = np.empty(n_shuffles)
    cols = list(signal.columns)
    for i in range(n_shuffles):
        shuffled = signal.copy()
        for y in signal.index:
            perm = rng.permutation(len(cols))
            shuffled.loc[y] = signal.loc[y].to_numpy()[perm]
        h = quintile_returns(shuffled, fwd_ret, q=q, min_names=min_names)
        null_means[i] = float(h["hedge"].mean())

    p_value = float((null_means >= real_mean).mean())
    return p_value, null_means


def random_portfolio_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    q: float = 0.20,
    n_draws: int = 500,
    min_names: int = 10,
    seed: int = 529,
) -> pd.Series:
    """Excess-vs-market returns of random same-size portfolios (low-INVG quintile size)."""
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
    """Headline statistics for an annual return series (one-sample HAC t-stat, Sharpe, hit).

    Uses a Newey-West HAC standard error for the one-sample t-stat — appropriate for short
    annual series where even one-lag autocorrelation can inflate naive t-stats.
    """
    r = pd.Series(annual_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
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


def seed_robust_synthetic_t(
    premium: float,
    n_seeds: int = 20,
    n_firms: int = 200,
    n_years: int = 20,
    q: float = 0.20,
) -> dict:
    """Average the synthetic hedge t-stat over many seeds (a faithful-engine power check).

    Re-runs the synthetic panel over ``n_seeds`` distinct RNG seeds for a given planted
    ``premium`` and averages the hedge mean and HAC t-stat. This is ONLY a positive-control /
    power check — never cited to support a real-tape stamp. With ``premium = 0`` the averaged
    t-stat should hover near zero; with ``premium < 0`` it should reliably exceed +2.
    """
    from . import data as _d

    means, tstats = [], []
    for s in range(n_seeds):
        sig, fwd, _ = _d.synthetic_panel(
            n_firms=n_firms, n_years=n_years, premium=premium, seed=529 + s
        )
        h = quintile_returns(sig, fwd, q=q, min_names=10)
        st = summary(h["hedge"])
        means.append(st["mean"])
        tstats.append(st["tstat"])
    return {
        "premium": premium,
        "n_seeds": n_seeds,
        "mean_hedge": float(np.mean(means)),
        "mean_t": float(np.mean(tstats)),
        "frac_t_gt2": float(np.mean([t > 2 for t in tstats])),
    }

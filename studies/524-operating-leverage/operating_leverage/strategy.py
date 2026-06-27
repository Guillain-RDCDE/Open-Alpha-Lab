"""Strategy for Study 524 (Operating-Leverage) — Novy-Marx (2011).

**The operating-leverage premium**: a firm's *fixed* operating costs act like leverage on
the income statement. A firm whose operating costs (COGS + SG&A) are large relative to its
asset base behaves like a levered claim on demand — small swings in revenue produce large
swings in operating profit — so it carries more operating risk and (Novy-Marx 2011 argues)
earns a premium. This is the operating-leverage value channel, distinct from the
*financial* leverage anomaly ([154 Financial-Leverage], which sorts on balance-sheet debt).

**Signal construction (Novy-Marx 2011 flavour)**::

    OL_t = OperatingCosts_t / Total_Assets_t

where ``OperatingCosts`` = COGS + SG&A (or a total-operating-cost fallback when a filer does
not report the COGS/SG&A split cleanly). Higher OL = a more operating-cost-intensive cost
structure. The paper predicts HIGH OL -> HIGH next-year return, so we go LONG the high-OL
(operating-risk) leg and SHORT the low-OL (asset-heavy / capital-light) leg.

**Honest controls**:

1. Equal-weight market: all basket names with a valid OL that year.
2. Random-portfolio null: random same-size subsets, to check whether any apparent edge is
   the OL signal or just holding a small concentrated slice.
3. Placebo / label-shuffle null: shuffle the OL labels within each year and re-run the
   long-short many times — a real signal should beat its own shuffled self.

**Costs**: one-way bps x NAV x annual turnover (the basket is fully rebalanced each year,
so turnover is high) plus a borrow charge on the short leg. Gross vs net reported.

**Execution lag (one)**: OL for fiscal year y -> calendar-year y+1 return, entered one
trading day after the y+1 window opens. No same-bar fills, no look-ahead.

**Survivorship**: the basket is large-cap names still trading in 2026. High-operating-leverage
firms whose demand collapsed and went under are absent; all results are upper bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TR_DAYS = 252


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def ol_signal(fund: pd.DataFrame) -> pd.DataFrame:
    """Build OL = OperatingCosts_t / Total_Assets_t from a long-form fundamentals frame.

    ``fund`` has columns (ticker, year, cogs, sga, costs, assets). Operating costs are
    assembled per firm-year as:

    - ``COGS + SG&A`` when both are present (the Novy-Marx operating-cost measure);
    - else the total-operating-cost fallback ``costs`` (CostsAndExpenses / OperatingExpenses);
    - else SG&A alone (capital-light filers that report no COGS) — a fixed-cost-intensity proxy.

    The denominator is contemporaneous total assets (year t). Operating costs are reported
    positive in the income statement, so a HIGH OL means a HEAVY operating-cost structure.

    Returns a (year x ticker) DataFrame of OL ratios (higher = more operating-leverage);
    cells with a missing or non-positive asset denominator, or no usable cost figure, are
    left NaN.
    """
    f = fund.copy()

    def _opcost(row: pd.Series) -> float:
        cogs, sga, costs = row.get("cogs"), row.get("sga"), row.get("costs")
        has_cogs = pd.notna(cogs)
        has_sga = pd.notna(sga)
        if has_cogs and has_sga:
            return abs(cogs) + abs(sga)
        if pd.notna(costs):
            return abs(costs)
        if has_cogs:
            return abs(cogs)
        if has_sga:
            return abs(sga)
        return np.nan

    f["opcost"] = f.apply(_opcost, axis=1)
    cost = f.pivot_table(index="year", columns="ticker", values="opcost")
    ast = f.pivot_table(index="year", columns="ticker", values="assets")
    cost = cost.sort_index()
    ast = ast.sort_index()

    ol = cost / ast.where(ast > 0)
    ol = ol.dropna(how="all")
    ol.index.name = "year"
    return ol


def forward_year_returns(
    prices: pd.DataFrame,
    years: list[int],
    tickers: list[str],
    report_lag_months: int = 4,
    lag_days: int = 1,
    asof: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """12-month forward total return per ticker, after a report lag + one-day execution lag.

    A fiscal year y 10-K is not public until a few months after year-end. We therefore enter
    at the close ``lag_days`` trading days after a fixed report date (``report_lag_months``
    after the fiscal-year-end ~31 Dec y) and hold exactly 12 months. A window is **only used
    if it has fully completed by ``asof``** — no partial-year return ever enters a stamped
    run. ``fwd.loc[y]`` is that 12-month forward return for signal year y. The signal (year-y
    OL) is strictly public before the position is taken, and there is exactly one execution lag.
    """
    if asof is None:
        asof = pd.Timestamp.today().normalize()
    px = prices.reindex(columns=tickers).sort_index()
    out: dict[int, pd.Series] = {}
    for y in years:
        entry_dt = pd.Timestamp(year=int(y) + 1, month=1, day=1) + pd.DateOffset(months=report_lag_months)
        exit_dt = entry_dt + pd.DateOffset(months=12)
        if exit_dt > asof:
            continue  # window not complete -> never stamp a partial period
        win = px.loc[entry_dt:exit_dt]
        if len(win) < lag_days + 2:
            continue
        entry = win.iloc[lag_days]
        exit_ = win.iloc[-1]
        out[int(y)] = (exit_ / entry) - 1.0
    fwd = pd.DataFrame(out).T.sort_index()
    fwd.index.name = "year"
    return fwd


# ---------------------------------------------------------------------------
# Portfolio construction and returns
# ---------------------------------------------------------------------------
def sorted_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_groups: int = 3,
    min_names: int = 6,
) -> pd.DataFrame:
    """Annual group-sorted returns on the OL signal vs next-year returns.

    With a ~40-name basket, terciles (``n_groups=3``) keep each leg liquid (>=6 names).
    Group 1 = low OL (asset-heavy), the top group = high OL (operating-leverage). Returns a
    frame with ``g1..gK``, ``market`` (equal-weight all), ``n``, and ``hedge`` (gK - g1,
    long high-OL / short low-OL), indexed by signal year.
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

        ranks = s.rank(method="first")
        bins = np.ceil(ranks / len(s) * n_groups).clip(1, n_groups).astype(int)
        grp_rets = [float(nxt[bins == g].mean()) for g in range(1, n_groups + 1)]

        row: dict = {"market": float(nxt.mean()), "n": int(len(s))}
        for gi, gr in enumerate(grp_rets, start=1):
            row[f"g{gi}"] = gr
        row["hedge"] = grp_rets[-1] - grp_rets[0]  # long high-OL, short low-OL
        rows[int(y)] = row

    return pd.DataFrame(rows).T.sort_index()


def random_portfolio_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_groups: int = 3,
    n_draws: int = 500,
    seed: int = 524,
    min_names: int = 6,
) -> pd.Series:
    """Null distribution of random-portfolio excess returns (high-OL-leg size)."""
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
        leg = max(1, len(s) // n_groups)
        arr = nxt.to_numpy()
        mkt = arr.mean()
        for _ in range(n_draws):
            pick = rng.choice(len(arr), size=leg, replace=False)
            excesses.append(float(arr[pick].mean() - mkt))
    return pd.Series(excesses, name="random_excess")


def placebo_hedge_tstats(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    n_groups: int = 3,
    n_shuffles: int = 1000,
    seed: int = 524,
    min_names: int = 6,
) -> tuple[float, pd.Series]:
    """Label-shuffle null for the long-short hedge mean.

    Shuffle the OL labels within each year (breaking any signal-return link) and recompute
    the hedge each time. Returns ``(p_value, shuffled_means)`` where ``p_value`` is the
    fraction of shuffles whose mean hedge is at least as large as the real one (one-sided,
    real direction = positive high-minus-low). A real signal should sit in the right tail.
    """
    real = sorted_returns(signal, fwd_ret, n_groups=n_groups, min_names=min_names)
    real_mean = float(real["hedge"].mean())

    rng = np.random.default_rng(seed)
    shuffled: list[float] = []
    for _ in range(n_shuffles):
        shuf = signal.copy()
        for y in shuf.index:
            row = shuf.loc[y].dropna()
            if len(row) < min_names:
                continue
            vals = row.to_numpy().copy()
            rng.shuffle(vals)
            shuf.loc[y, row.index] = vals
        h = sorted_returns(shuf, fwd_ret, n_groups=n_groups, min_names=min_names)
        shuffled.append(float(h["hedge"].mean()))
    sh = pd.Series(shuffled, name="shuffled_hedge_mean")
    p = float((sh >= real_mean).mean())
    return p, sh


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------
def net_of_costs(
    hedge_mean: float,
    one_way_bps: float = 10.0,
    annual_turnover: float = 2.0,
    borrow_bps: float = 50.0,
) -> dict:
    """Charge trading costs and short borrow against a gross annual hedge return.

    The long-short book fully re-forms each year (annual rebalance of both legs), so
    ``annual_turnover`` ~ 2.0 round-trips of NAV/yr across the two legs. Costs =
    ``one_way_bps`` x turnover; borrow = ``borrow_bps`` x 1 (one short leg of NAV held all
    year). Returns gross, the cost drag, and the net annual hedge return.
    """
    cost = (one_way_bps / 1e4) * annual_turnover
    borrow = borrow_bps / 1e4
    net = hedge_mean - cost - borrow
    return {
        "gross": hedge_mean,
        "cost_drag": cost,
        "borrow_drag": borrow,
        "net": net,
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summary(annual_returns: pd.Series) -> dict:
    """Headline statistics for an annual return series (HAC t-stat, Sharpe, hit rate).

    Uses a Newey-West HAC standard error for the one-sample t-stat — appropriate for short
    annual series where one-lag autocorrelation can inflate naive t-stats.
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
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "mean": mu, "vol": std, "sharpe": sr, "tstat": tstat,
        "hit_rate": float((r > 0).mean()), "max_drawdown": max_dd, "n": n,
    }


def seed_robust_synthetic(
    premium: float,
    n_seeds: int = 25,
    n_firms: int = 200,
    n_years: int = 18,
    n_groups: int = 3,
) -> dict:
    """Average the synthetic hedge HAC-t over many seeds — the faithful-engine power check.

    A single lucky seed is never enough (house rule). We sweep ``n_seeds`` and report the
    mean hedge and mean HAC-t so the positive control is seed-robust.
    """
    from . import data as _d

    means, ts = [], []
    for sd in range(n_seeds):
        ol, fwd, _ = _d.synthetic_panel(n_firms=n_firms, n_years=n_years,
                                        premium=premium, seed=1000 + sd)
        h = sorted_returns(ol, fwd, n_groups=n_groups, min_names=6)
        s = summary(h["hedge"])
        means.append(s["mean"])
        ts.append(s["tstat"])
    return {
        "premium": premium,
        "mean_hedge": float(np.mean(means)),
        "mean_tstat": float(np.mean(ts)),
        "n_seeds": n_seeds,
    }

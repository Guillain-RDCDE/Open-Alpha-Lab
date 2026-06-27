"""The engine and its honest controls — Study 520 (External-Financing-Anomaly).

Bradshaw–Richardson–Sloan (2006): a firm's **net external financing** — the cash it raised from
debt *and* equity holders, net of what it returned — predicts *lower* subsequent stock returns.
The story is part overinvestment (raisers waste the cash), part market-timing (managers issue
when the stock is dear). The measure, straight off the cash-flow statement::

    XFIN = ΔEquity_external + ΔDebt_external
         = (equity issued − equity repurchased − dividends paid)
         + (debt issued − debt repaid)

scaled by *average total assets*. This module computes the scaled signal per name-year, sorts the
basket into raisers vs retirers, forms a long-retirers / short-raisers book, charges it costs and
borrow, runs a HAC *t*, a label-shuffle placebo null and a seed-robustness sweep, and proves the
engine is faithful on a synthetic world with a planted penalty.

Execution lag (documented, exactly one): the prior fiscal year's XFIN is treated as public only
after a reporting lag, and the book holds the **next 12 months** of return measured from one day
after that public date — no fundamental is ever used before it could have been read, no same-bar
fill. Costs are charged once per annual rebalance as one-way turnover × NAV (both legs trade),
plus an annual borrow on the short (raiser) leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# The signal: scaled net external financing
# ---------------------------------------------------------------------------
def scaled_xfin(fund: pd.DataFrame) -> pd.DataFrame:
    """Net external financing scaled by average total assets, per (ticker, fy_end).

    ``fund`` is the long-format fundamentals frame from ``data.fetch_fundamentals``. We build the
    BRS cash-flow measure, preferring net debt issuance when Yahoo reports it and falling back to
    gross issue − repay. Equity external financing = issuance − repurchases − dividends (all are
    flows between the firm and its equity holders). The denominator is the average of total assets
    at the two consecutive fiscal-year ends; when the prior year is missing we use current assets.

    Returns a (fy_end × ticker) wide DataFrame of the scaled signal (positive = net raiser).
    """
    if fund.empty:
        return pd.DataFrame()

    out_rows: list[dict] = []
    for tk, g in fund.groupby(level=0):
        g = g.droplevel(0).sort_index()
        assets = g["total_assets"]
        for fy in g.index:
            row = g.loc[fy]

            # Net debt external financing.
            if not np.isnan(row.get("debt_net", np.nan)):
                debt = row["debt_net"]
            else:
                di = row.get("debt_issue", np.nan)
                dr = row.get("debt_repay", np.nan)
                # repay is reported as a negative outflow; net = issue + repay.
                debt = np.nansum([di, dr]) if not (np.isnan(di) and np.isnan(dr)) else np.nan

            # Net equity external financing = issuance − repurchases − dividends.
            ei = row.get("equity_issue", np.nan)
            er = row.get("equity_repurchase", np.nan)  # already a negative outflow
            dv = row.get("dividends", np.nan)  # already a negative outflow
            eq_parts = [ei, er, dv]
            equity = np.nansum(eq_parts) if not all(np.isnan(x) for x in eq_parts) else np.nan

            xfin = np.nansum([debt, equity]) if not (np.isnan(debt) and np.isnan(equity)) else np.nan

            # Average total assets across the two consecutive year ends.
            a_now = assets.get(fy, np.nan)
            prior = assets[assets.index < fy]
            a_prev = prior.iloc[-1] if len(prior) else np.nan
            denom = np.nanmean([a_now, a_prev]) if not (np.isnan(a_now) and np.isnan(a_prev)) else np.nan

            if np.isnan(xfin) or np.isnan(denom) or denom <= 0:
                continue
            out_rows.append({"fy_end": fy, "ticker": tk, "xfin": float(xfin / denom)})

    if not out_rows:
        return pd.DataFrame()
    wide = pd.DataFrame(out_rows).pivot(index="fy_end", columns="ticker", values="xfin").sort_index()
    return wide


def by_calendar_year(signal: pd.DataFrame) -> pd.DataFrame:
    """Pool the scattered fiscal-year-end signal into one row per *calendar year of FY end*.

    Different names close their books on different dates (Apple in September, most in December).
    To form a sensible cross-section we bucket each name's signal into the calendar year its fiscal
    year *ends* in, taking that name's latest value within the year. The resulting (year × ticker)
    frame is dense enough to sort (40–45 names/year on this basket), at the cost of treating an
    Apple-FY2024 (ends Sep-2024) and a Microsoft-FY2024 (ends Dec-2024) as the same "2024" sort —
    a mild timing approximation that we state, not hide.

    The index is a Dec-31 timestamp for each calendar year, so the forward-return machinery (which
    anchors off FY-end + reporting lag) still applies cleanly with a conservative common lag.
    """
    if signal.empty:
        return signal
    years = signal.index.year
    out = signal.groupby(years).apply(lambda x: x.bfill().ffill().iloc[-1])
    out.index = pd.DatetimeIndex([pd.Timestamp(int(y), 12, 31) for y in out.index], name="fy_year")
    return out


# ---------------------------------------------------------------------------
# Forward returns aligned to the (lagged) signal date
# ---------------------------------------------------------------------------
def forward_returns(
    prices: pd.DataFrame,
    signal_index: pd.DatetimeIndex,
    report_lag_days: int = 90,
    hold_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Forward simple returns over ``hold_days`` starting ``report_lag_days`` after each FY end.

    For each fiscal-year-end in ``signal_index`` the public date is ``fy_end + report_lag_days``;
    we enter at the first available close on/after the *next* trading day (one execution lag) and
    exit ``hold_days`` later. Returns a (fy_end × ticker) frame of holding-period returns.
    """
    if prices.empty:
        return pd.DataFrame()
    px = prices.sort_index()
    rows: list[dict] = []
    for fy in signal_index:
        public = pd.Timestamp(fy) + pd.Timedelta(days=report_lag_days)
        entry_candidates = px.index[px.index > public]  # strictly after public date: no same-bar fill
        if len(entry_candidates) == 0:
            continue
        entry = entry_candidates[0]
        entry_pos = px.index.get_loc(entry)
        exit_pos = entry_pos + hold_days
        if exit_pos >= len(px.index):
            continue  # incomplete holding window -> drop (no partial-year bar in a stamp)
        exit_dt = px.index[exit_pos]
        p0 = px.loc[entry]
        p1 = px.loc[exit_dt]
        ret = (p1 / p0 - 1.0)
        for tk, r in ret.items():
            if np.isfinite(r):
                rows.append({"fy_end": fy, "ticker": tk, "ret": float(r)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot(index="fy_end", columns="ticker", values="ret").sort_index()


# ---------------------------------------------------------------------------
# Cross-sectional long-short
# ---------------------------------------------------------------------------
def long_short(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    quantile: float = 0.3,
    cost_bps: float = 0.0,
    borrow_ann_bps: float = 0.0,
    min_names: int = 6,
) -> pd.DataFrame:
    """Annual long-retirers / short-raisers spread, net of costs and borrow.

    Each fiscal year we rank the cross-section by scaled XFIN, go **long** the bottom
    ``quantile`` (the retirers — lowest financing, expected winners) and **short** the top
    ``quantile`` (the raisers — expected losers), equal-weighted within each leg. The spread is
    ``long_leg − short_leg``. Costs are one-way turnover × NAV on both legs; the short (raiser)
    leg pays an annual borrow. A year with fewer than ``min_names`` ranked names is dropped.

    Returns a DataFrame indexed by fy_end with columns ``long_ret``, ``short_ret``, ``spread``
    (gross), ``spread_net`` and ``n``.
    """
    common = signal.index.intersection(fwd.index)
    rows: list[dict] = []
    monthly_cost = 2.0 * cost_bps * 1e-4  # both legs traded at the annual rebalance (round trip)
    annual_borrow = borrow_ann_bps * 1e-4  # one year of short borrow
    for fy in common:
        s = signal.loc[fy].dropna()
        r = fwd.loc[fy].dropna()
        names = s.index.intersection(r.index)
        s = s[names]
        r = r[names]
        if len(names) < min_names:
            continue
        k = max(1, int(np.floor(len(names) * quantile)))
        ranked = s.sort_values()
        retirers = ranked.index[:k]   # lowest financing -> long
        raisers = ranked.index[-k:]   # highest financing -> short
        long_ret = float(r[retirers].mean())
        short_ret = float(r[raisers].mean())
        spread = long_ret - short_ret
        spread_net = spread - monthly_cost - annual_borrow
        rows.append(
            {
                "fy_end": fy,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "spread": spread,
                "spread_net": spread_net,
                "n": int(len(names)),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("fy_end").sort_index()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def hac_tstat(x: pd.Series | np.ndarray) -> float:
    """Newey-West (HAC) t-stat on the mean of an annual series (desk lag rule)."""
    a = pd.Series(x).astype(float).dropna().to_numpy()
    n = a.size
    if n < 3:
        return float("nan")
    mu = a.mean()
    e = a - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lags = min(lags, n - 1)
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def plain_tstat(x: pd.Series | np.ndarray) -> float:
    """One-sample t-stat on the mean (iid), the headline cross-sectional number."""
    a = pd.Series(x).astype(float).dropna().to_numpy()
    n = a.size
    if n < 2 or a.std(ddof=1) == 0:
        return float("nan")
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(n)))


def summary(spread: pd.Series, periods_per_year: int = 1) -> dict:
    """Headline stats for an annual spread series: mean, vol, Sharpe, plain & HAC t, hit, n."""
    r = pd.Series(spread).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in ("mean", "vol", "sharpe", "t", "tstat_hac", "hit", "n")}
    mean = float(r.mean()) * periods_per_year
    vol = float(r.std(ddof=1)) * np.sqrt(periods_per_year)
    return {
        "mean": mean,
        "vol": vol,
        "sharpe": mean / vol if vol > 0 else float("nan"),
        "t": plain_tstat(r),
        "tstat_hac": hac_tstat(r),
        "hit": float((r > 0).mean()),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    quantile: float = 0.3,
    n_shuffle: int = 1000,
    seed: int = 520,
    **ls_kwargs,
) -> dict:
    """Label-shuffle placebo: how often does a *random* sort beat the real spread mean?

    The real long-short ranks on true XFIN. The placebo shuffles the financing labels within each
    year (destroying any link between financing and forward return) and re-forms the long-short
    ``n_shuffle`` times. The placebo p-value is the share of shuffles whose mean spread is at least
    as large as the real one — a one-sided test that the real sort beats noise.

    Returns ``{"real_mean", "p_value", "null_mean", "null_std"}``.
    """
    real = long_short(signal, fwd, quantile=quantile, **ls_kwargs)
    if real.empty:
        return {"real_mean": float("nan"), "p_value": float("nan"), "null_mean": float("nan"), "null_std": float("nan")}
    real_mean = float(real["spread"].mean())
    rng = np.random.default_rng(seed)
    null_means = np.empty(n_shuffle)
    common = signal.index.intersection(fwd.index)
    for b in range(n_shuffle):
        shuffled = signal.copy()
        for fy in common:
            row = shuffled.loc[fy].dropna()
            perm = rng.permutation(row.values)
            shuffled.loc[fy, row.index] = perm
        ls = long_short(shuffled, fwd, quantile=quantile, **ls_kwargs)
        null_means[b] = float(ls["spread"].mean()) if not ls.empty else np.nan
    null_means = null_means[np.isfinite(null_means)]
    p = float((null_means >= real_mean).mean()) if null_means.size else float("nan")
    return {
        "real_mean": real_mean,
        "p_value": p,
        "null_mean": float(np.mean(null_means)) if null_means.size else float("nan"),
        "null_std": float(np.std(null_means)) if null_means.size else float("nan"),
    }


# ---------------------------------------------------------------------------
# Synthetic positive control — faithful-engine / power check ONLY
# ---------------------------------------------------------------------------
def synthetic_control(
    xfin_penalty: float = 0.40,
    quantile: float = 0.3,
    seeds: tuple[int, ...] = tuple(range(20)),
    n_stocks: int = 45,
    n_years: int = 12,
) -> dict:
    """Recover a planted external-financing penalty, averaged over many RNG seeds.

    For each seed we generate a synthetic world with the planted ``xfin_penalty``, run the exact
    same long-short, and record the spread's plain t. We report the mean t across seeds (a
    Welch-style robustness average, not one lucky seed) and the matched null (penalty=0). This is
    a *faithful-engine / power* check only; it is NEVER cited to support the real-tape stamp.

    Returns ``{"mean_t_planted", "mean_t_null", "mean_spread_planted", "frac_t_above_2", "seeds"}``.
    """
    from . import data as _data

    t_planted, t_null, spreads = [], [], []
    above = 0
    for sd in seeds:
        xf, fwd, _ = _data.synthetic_world(
            n_stocks=n_stocks, n_years=n_years, xfin_penalty=xfin_penalty, seed=sd
        )
        ls = long_short(xf, fwd, quantile=quantile)
        s = summary(ls["spread"])
        t_planted.append(s["t"])
        spreads.append(s["mean"])
        if np.isfinite(s["t"]) and s["t"] >= 2.0:
            above += 1

        xf0, fwd0, _ = _data.synthetic_world(
            n_stocks=n_stocks, n_years=n_years, xfin_penalty=0.0, seed=sd
        )
        ls0 = long_short(xf0, fwd0, quantile=quantile)
        t_null.append(summary(ls0["spread"])["t"])

    return {
        "mean_t_planted": float(np.nanmean(t_planted)),
        "mean_t_null": float(np.nanmean(t_null)),
        "mean_spread_planted": float(np.nanmean(spreads)),
        "frac_t_above_2": float(above / len(seeds)),
        "seeds": len(seeds),
    }

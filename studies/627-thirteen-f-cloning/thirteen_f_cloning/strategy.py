"""Strategy + inference for Study 627 — 13F Cloning (Berkshire Hathaway).

The folk claim: *copy Berkshire's 13F 45 days late and you still beat the market* — the
free-rider trade that "should not work" because everyone can see the filing. We build the
literal retail version:

    * **The clone.** At the close of the **first trading day after each original 13F-HR
      filing date** (the 45-day disclosure lag is *inside* the filing date), rebalance into
      Berkshire's **top-10 holdings by reported value**, either equal-weight (EW) or
      13F-value-weight (VW). Between filings the book drifts buy-and-hold. That filing-date
      rebalance is the study's **one execution lag** — the clone never sees a holding before
      the public did.

    * **The race.** Clone vs **SPY** (total-return, the buy-and-hold benchmark): CAGR, CAPM
      alpha/beta with **Newey-West (HAC) t** on monthly excess-vs-excess returns, plus the
      HAC t of the raw monthly active return. Costs are **one-way bps x traded NAV** at each
      rebalance (turnover measured against the drifted book).

    * **Third axis.** Does the clone beat **Berkshire itself** (BRK-B)? The 13F sleeve is
      ~always fully invested while Berkshire drags a giant cash pile — the "cash drag"
      question, raced clone-vs-BRK on the same tape.

    * **Random-manager placebo.** Same universe (the 28 priceable names that ever reached
      Berkshire's top 10), same calendar, same lag — but each "filing" picks 10 names at
      random. Averaged over >= 20 seeds (we use 200 draws), it separates *Berkshire's
      selection/timing* from *merely holding Berkshire-flavoured mega-caps*.

Inference: Newey-West t (lag = floor(0.75 * T^(1/3))) on monthly series; Sharpe races are
excess-vs-excess (^IRX risk-free leg); gross/net labeled everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as _data

TOP_N = 10
COST_BPS = (0.0, 5.0, 10.0, 20.0)


# --------------------------------------------------------------------------- #
# The clone engine
# --------------------------------------------------------------------------- #
def rebalance_events(prices: pd.DataFrame, holdings: pd.DataFrame,
                     top_n: int = TOP_N, weighting: str = "ew",
                     cusip_map: dict | None = None) -> list:
    """Turn the filings into (rebalance_date, {ticker: weight}) events.

    ``rebalance_date`` = first trading day **strictly after** the filing date (the single
    execution lag). CUSIPs map through ``cusip_map`` (default: the hardcoded Berkshire map;
    pass ``None``-keys-free map or ``{}``-like identity for synthetic worlds where the
    ``cusip`` column already holds tickers). Unpriceable names (delisted acquirees) are
    dropped and the remaining weights renormalised.
    """
    if cusip_map is None:
        cusip_map = {c: c for c in holdings["cusip"].unique()}
    dates = prices.index
    h = holdings[holdings["rank"] <= top_n].copy()
    h["ticker"] = h["cusip"].map(cusip_map)
    events = []
    for fd, sub in h.groupby("filing_date"):
        loc = dates.searchsorted(pd.Timestamp(fd), side="right")
        if loc >= len(dates):
            continue
        rd = dates[loc]
        sub = sub.dropna(subset=["ticker"])
        sub = sub[[t in prices.columns and not pd.isna(prices.at[rd, t])
                   for t in sub["ticker"]]]
        if len(sub) == 0:
            continue
        if weighting == "ew":
            w = {t: 1.0 / len(sub) for t in sub["ticker"]}
        elif weighting == "vw":
            tot = float(sub["value"].sum())
            w = {t: float(v) / tot for t, v in zip(sub["ticker"], sub["value"])}
        else:
            raise ValueError(weighting)
        events.append((rd, w))
    return sorted(events, key=lambda e: e[0])


def build_clone(prices: pd.DataFrame, holdings: pd.DataFrame, top_n: int = TOP_N,
                weighting: str = "ew", cost_bps: float = 0.0,
                cusip_map: dict | None = None) -> dict:
    """Daily clone returns with drifting weights, filing-date+1 rebalances and costs.

    The book holds the previous target through the rebalance day's close, then switches;
    the cost charged that day is ``cost_bps/1e4 x sum|w_new - w_drifted|`` (one-way cost x
    total traded NAV, buys + sells). Returns dict with the daily net return series
    (starting the day after the first rebalance), per-rebalance one-way turnover, and the
    annualised cost drag actually charged.
    """
    events = rebalance_events(prices, holdings, top_n=top_n, weighting=weighting,
                              cusip_map=cusip_map)
    if not events:
        raise ValueError("no rebalance events on this tape")
    dates = prices.index
    cols = list(prices.columns)
    jmap = {t: j for j, t in enumerate(cols)}
    ret_np = np.nan_to_num(prices.pct_change().to_numpy(dtype=float), nan=0.0)
    c = cost_bps / 1e4

    # map each event to its row; a later filing landing on the same day overwrites
    ev_by_loc: dict[int, np.ndarray] = {}
    for rd, wdict in events:
        tgt = np.zeros(len(cols))
        for t, wi in wdict.items():
            tgt[jmap[t]] = wi
        ev_by_loc[int(dates.get_loc(rd))] = tgt

    start_loc = min(ev_by_loc)
    w = np.zeros(len(cols))
    out_r = np.zeros(len(dates) - start_loc - 1)
    turnovers = []
    total_cost = 0.0
    for i in range(start_loc, len(dates)):
        r_row = ret_np[i]
        r_p = float(w @ r_row)
        if r_p != 0.0 or w.any():
            gross = w * (1.0 + r_row)
            s = gross.sum()
            if s > 0:
                w = gross / s
        cost = 0.0
        tgt = ev_by_loc.get(i)
        if tgt is not None:
            traded = float(np.abs(tgt - w).sum())
            cost = c * traded
            turnovers.append(traded / 2.0)          # one-way turnover
            total_cost += cost
            w = tgt.copy()
        if i > start_loc:                            # series starts after first entry
            out_r[i - start_loc - 1] = r_p - cost

    daily = pd.Series(out_r, index=dates[start_loc + 1:], name=f"clone_{weighting}")
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    return {
        "daily": daily,
        "n_rebalances": len(events),
        "avg_oneway_turnover": float(np.mean(turnovers[1:])) if len(turnovers) > 1 else 0.0,
        "cost_drag_ann_bps": (total_cost / years) * 1e4 if years > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Monthly aggregation + HAC inference
# --------------------------------------------------------------------------- #
def to_monthly(daily: pd.Series, drop_partial_first: bool = True) -> pd.Series:
    """Compound a daily return series into complete calendar months."""
    m = (1.0 + daily).resample("ME").prod() - 1.0
    if drop_partial_first and len(m) > 0 and daily.index[0].day > 3:
        m = m.iloc[1:]                               # first month starts mid-month
    if len(m) > 0 and (m.index[-1] - daily.index[-1]).days > 4:
        m = m.iloc[:-1]                              # last month is incomplete
    return m


def nw_tstat(x, lags: int | None = None) -> dict:
    """Newey-West (HAC) t of mean(x) vs 0. Default lag = floor(0.75 * T**(1/3))."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    if T < 8:
        return {"mean": float("nan"), "t": float("nan"), "lags": 0, "n": T}
    if lags is None:
        lags = int(np.floor(0.75 * T ** (1.0 / 3.0)))
    e = x - x.mean()
    s = float(e @ e) / T
    for l in range(1, lags + 1):
        s += 2.0 * (1.0 - l / (lags + 1.0)) * float(e[:-l] @ e[l:]) / T
    se = np.sqrt(s / T)
    return {"mean": float(x.mean()), "t": float(x.mean() / se) if se > 0 else float("nan"),
            "lags": lags, "n": T}


def capm_alpha_nw(port_ex: pd.Series, mkt_ex: pd.Series, lags: int | None = None) -> dict:
    """CAPM regression port_ex = a + b * mkt_ex with Newey-West (HAC) errors.

    Both legs are monthly EXCESS returns (excess-vs-excess). Alpha reported annualised (%).
    """
    df = pd.concat([port_ex, mkt_ex], axis=1).dropna()
    y = df.iloc[:, 0].to_numpy(dtype=float)
    m = df.iloc[:, 1].to_numpy(dtype=float)
    T = len(y)
    X = np.column_stack([np.ones(T), m])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    if lags is None:
        lags = int(np.floor(0.75 * T ** (1.0 / 3.0)))
    g = X * u[:, None]
    S = (g.T @ g) / T
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = (g[l:].T @ g[:-l]) / T
        S += w * (G + G.T)
    V = T * (XtX_inv @ S @ XtX_inv)
    se = np.sqrt(np.diag(V))
    return {"alpha_m": float(b[0]), "alpha_ann_pct": float(b[0] * 12.0 * 100.0),
            "beta": float(b[1]), "t_alpha": float(b[0] / se[0]),
            "t_beta": float(b[1] / se[1]), "lags": lags, "n": T}


def perf_stats(monthly: pd.Series, rf_m: pd.Series | None = None) -> dict:
    """CAGR, annualised vol, (excess) Sharpe, max drawdown of a monthly return series."""
    m = monthly.dropna()
    n = len(m)
    cum = float((1.0 + m).prod())
    cagr = cum ** (12.0 / n) - 1.0
    vol = float(m.std(ddof=1)) * np.sqrt(12.0)
    if rf_m is not None:
        ex = (m - rf_m.reindex(m.index).fillna(0.0)).to_numpy()
    else:
        ex = m.to_numpy()
    sharpe = float(ex.mean() / ex.std(ddof=1) * np.sqrt(12.0)) if ex.std(ddof=1) > 0 else float("nan")
    nav = (1.0 + m).cumprod()
    dd = float((nav / nav.cummax() - 1.0).min())
    return {"n_months": n, "cagr_pct": cagr * 100.0, "vol_pct": vol * 100.0,
            "sharpe_ex": sharpe, "max_dd_pct": dd * 100.0}


def monthly_rf(prices: pd.DataFrame, like: pd.Series) -> pd.Series:
    """Monthly risk-free from ^IRX, aligned to ``like``'s daily index then compounded."""
    rf_d = _data.daily_rf(prices).reindex(like.index).fillna(0.0)
    return (1.0 + rf_d).resample("ME").prod() - 1.0


# --------------------------------------------------------------------------- #
# The headline race
# --------------------------------------------------------------------------- #
def race(prices: pd.DataFrame, holdings: pd.DataFrame, weighting: str = "ew",
         cost_bps: float = 0.0, bench: str = "SPY",
         cusip_map: dict | None = "brk") -> dict:
    """Clone (net of costs) vs a benchmark: CAGR, HAC t of active, CAPM alpha (HAC), Sharpe."""
    cmap = _data.CUSIP_TO_TICKER if cusip_map == "brk" else cusip_map
    built = build_clone(prices, holdings, weighting=weighting, cost_bps=cost_bps,
                        cusip_map=cmap)
    clone_d = built["daily"]
    bench_d = prices[bench].pct_change().reindex(clone_d.index)
    clone_m = to_monthly(clone_d)
    bench_m = to_monthly(bench_d)
    idx = clone_m.index.intersection(bench_m.index)
    clone_m, bench_m = clone_m.loc[idx], bench_m.loc[idx]
    rf_m = monthly_rf(prices, clone_d).reindex(idx).fillna(0.0)

    active = clone_m - bench_m
    nw = nw_tstat(active.to_numpy())
    capm = capm_alpha_nw(clone_m - rf_m, bench_m - rf_m)
    return {
        "weighting": weighting, "cost_bps": cost_bps, "bench": bench,
        "start": str(idx[0].date()), "end": str(idx[-1].date()),
        "clone": perf_stats(clone_m, rf_m), "bench_stats": perf_stats(bench_m, rf_m),
        "active_ann_pct": nw["mean"] * 12.0 * 100.0, "t_active": nw["t"],
        "nw_lags": nw["lags"], "n_months": nw["n"],
        "alpha_ann_pct": capm["alpha_ann_pct"], "beta": capm["beta"],
        "t_alpha": capm["t_alpha"],
        "n_rebalances": built["n_rebalances"],
        "avg_oneway_turnover_pct": built["avg_oneway_turnover"] * 100.0,
        "cost_drag_ann_bps": built["cost_drag_ann_bps"],
        "clone_m": clone_m, "bench_m": bench_m, "rf_m": rf_m,
    }


# --------------------------------------------------------------------------- #
# Random-manager placebo (>= 20 seeds; we use 200)
# --------------------------------------------------------------------------- #
def random_manager_baseline(prices: pd.DataFrame, holdings: pd.DataFrame,
                            n_draws: int = 200, seed: int = 627,
                            cost_bps: float = 0.0) -> dict:
    """Random top-10 clones from the same universe / calendar / lag, EW, averaged.

    Each draw replaces every filing's top-10 with 10 names drawn uniformly from the 28
    priceable names that ever reached Berkshire's top 10. The distribution of their active
    returns vs SPY answers: is the clone's edge Berkshire's *selection*, or just *this
    universe of mega-caps*? (Averaged over ``n_draws`` >= 20 seeds — house rule.)
    """
    rng = np.random.default_rng(seed)
    universe = list(_data.TICKERS)
    fd_list = sorted(holdings["filing_date"].unique())
    actives, cagrs = [], []
    spy_d_all = prices["SPY"].pct_change()
    for _k in range(n_draws):
        rows = []
        for fd in fd_list:
            picks = rng.choice(universe, size=TOP_N, replace=False)
            for rank, t in enumerate(picks, 1):
                rows.append({"period": fd, "filing_date": fd, "rank": rank,
                             "cusip": t, "issuer": t, "value": 1.0, "total_value": 0.0})
        hh = pd.DataFrame(rows)
        built = build_clone(prices, hh, weighting="ew", cost_bps=cost_bps,
                            cusip_map={t: t for t in universe})
        cm = to_monthly(built["daily"])
        bm = to_monthly(spy_d_all.reindex(built["daily"].index))
        idx = cm.index.intersection(bm.index)
        act = (cm.loc[idx] - bm.loc[idx]).to_numpy()
        actives.append(act.mean() * 12.0 * 100.0)
        cagrs.append(perf_stats(cm.loc[idx])["cagr_pct"])
    actives = np.asarray(actives)
    return {"n_draws": n_draws, "mean_active_ann_pct": float(actives.mean()),
            "sd_active_ann_pct": float(actives.std(ddof=1)),
            "mean_cagr_pct": float(np.mean(cagrs)),
            "pct_above": None, "actives": actives}

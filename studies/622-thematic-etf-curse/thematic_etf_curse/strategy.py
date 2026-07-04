"""Strategy + inference for Study 622 — Thematic ETF Curse.

The claim (Ben-David, Franzoni, Kim & Moussawi, RFS 2023): specialized/thematic ETFs are
launched on peak-hype portfolios and deliver ~ -5%/yr risk-adjusted returns over their first
years — the launch date is the sell signal. We measure it the way the paper does, on a
calendar-time footing that handles the massive cross-correlation of overlapping young funds:

  * **Calendar-time "young thematics" portfolio.** Each month, equal-weight every thematic
    ETF within its first ``W`` complete months of life (W = 12 and 36). One time series →
    CAPM regression on SPY excess returns → the alpha and its Newey-West (HAC) t. This is
    the primary test; pooled per-fund-month regressions would pseudo-replicate the same
    calendar months dozens of times.

  * **Contrasts.** (a) The same construction on *seasoned* thematics (event months 37+):
    is it launch timing, or are thematics always bad? (b) The placebo: the same young-fund
    portfolio built from broad plain-vanilla index-ETF launches, which should show ~0 alpha.

  * **Event-time alpha curve.** Mean CAPM-abnormal return by event month (per-fund
    full-sample beta — a mild, documented look-ahead used for the curve only, never for the
    headline alpha), cumulated over the first 48 months.

  * **Third axis — buy the -50% dip?** First month-end a fund closes >= 50% below its
    post-launch running max, buy at the NEXT month's start (one clean lag), hold 36 months;
    calendar-time CAPM alpha of the dip-buyer book plus per-event forward runs vs SPY.

Execution: membership is known at month-end (a fund's launch date and age are public);
positions form at the month-end close and hold the following month — exactly ONE lag
everywhere (the dip signal too). Costs: one-way bps × NAV on entries/exits; the SHORT
implementation of the curse pays borrow on the full short leg. All prices auto-adjusted =
total-return; alphas are excess-vs-excess (fund minus T-bill on SPY minus T-bill).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12.0
NW_LAGS = 6


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def nw_tstat(x, lags: int = NW_LAGS) -> tuple[float, float, float]:
    """(mean, HAC se, t) of the mean of ``x`` with Newey-West (Bartlett) weights."""
    x = pd.Series(x).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 8:
        return float("nan"), float("nan"), float("nan")
    mu = x.mean()
    e = x - mu
    s = float(e @ e) / n
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(e[k:] @ e[:-k]) / n
        s += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    se = np.sqrt(max(s, 1e-18) / n)
    return float(mu), float(se), float(mu / se)


def capm_alpha_nw(port: pd.Series, mkt: pd.Series, rf: pd.Series,
                  lags: int = NW_LAGS) -> dict:
    """CAPM regression (port - rf) = alpha + beta (mkt - rf) + e, Newey-West t on alpha.

    Excess-vs-excess by construction. Returns monthly and annualized (x12) alpha,
    beta, the HAC t of alpha, and the sample size.
    """
    df = pd.concat({"p": port, "m": mkt, "rf": rf}, axis=1).dropna()
    n = len(df)
    if n < 24:
        return {"alpha_m": float("nan"), "alpha_ann_pct": float("nan"),
                "beta": float("nan"), "t_alpha": float("nan"), "n": n}
    y = (df["p"] - df["rf"]).to_numpy()
    x = (df["m"] - df["rf"]).to_numpy()
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    g = X * u[:, None]                       # moment conditions x_t * u_t
    S = g.T @ g
    L = min(lags, n - 1)
    for k in range(1, L + 1):
        gk = g[k:].T @ g[:-k]
        S += (1.0 - k / (L + 1.0)) * (gk + gk.T)
    V = XtX_inv @ S @ XtX_inv
    se_a = float(np.sqrt(max(V[0, 0], 1e-18)))
    return {"alpha_m": float(b[0]), "alpha_ann_pct": float(b[0]) * MONTHS * 100.0,
            "beta": float(b[1]), "t_alpha": float(b[0] / se_a), "n": n}


# --------------------------------------------------------------------------- #
# Event-month bookkeeping
# --------------------------------------------------------------------------- #
def event_months(univ: dict, tk: str) -> pd.Series:
    """Event month index k for every monthly return of fund ``tk`` (k=1 first complete month)."""
    r = univ["mret"][tk].dropna()
    per = r.index.to_period("M")
    k = per.astype("int64") - univ["launch"][tk].ordinal
    return pd.Series(k, index=r.index)


def calendar_time_portfolio(univ: dict, tickers: list[str],
                            k_lo: int = 1, k_hi: int = 36,
                            min_members: int = 3) -> tuple[pd.Series, pd.Series]:
    """Equal-weight monthly returns of funds whose event month is in [k_lo, k_hi].

    Returns (portfolio return series, member count series), keeping only months
    with >= ``min_members`` funds so a single fund never masquerades as a panel.
    """
    mret = univ["mret"]
    cols = {}
    for tk in tickers:
        k = event_months(univ, tk)
        sel = k[(k >= k_lo) & (k <= k_hi)].index
        if len(sel):
            cols[tk] = mret[tk].reindex(sel)
    if not cols:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    panel = pd.DataFrame(cols)
    cnt = panel.notna().sum(axis=1)
    port = panel.mean(axis=1)[cnt >= min_members].dropna()
    return port, cnt[cnt >= min_members]


def young_alpha(univ: dict, tickers: list[str], w: int = 36,
                lags: int = NW_LAGS, min_members: int = 3) -> dict:
    """Headline: CAPM alpha (NW t) of the first-``w``-months calendar-time portfolio."""
    port, cnt = calendar_time_portfolio(univ, tickers, 1, w, min_members)
    out = capm_alpha_nw(port, univ["mkt"], univ["rf"], lags=lags)
    out.update({"w": w, "avg_members": float(cnt.mean()) if len(cnt) else float("nan"),
                "start": (str(port.index.min().date()) if len(port) else ""),
                "end": (str(port.index.max().date()) if len(port) else "")})
    return out


def seasoned_alpha(univ: dict, tickers: list[str], k_lo: int = 37, k_hi: int = 10_000,
                   lags: int = NW_LAGS, min_members: int = 3) -> dict:
    """Contrast: the same construction on event months 37+ (past the launch window)."""
    port, cnt = calendar_time_portfolio(univ, tickers, k_lo, k_hi, min_members)
    out = capm_alpha_nw(port, univ["mkt"], univ["rf"], lags=lags)
    out.update({"avg_members": float(cnt.mean()) if len(cnt) else float("nan")})
    return out


def young_vs_seasoned_spread(univ: dict, tickers: list[str], w: int = 36,
                             lags: int = NW_LAGS) -> dict:
    """CAPM alpha of the young-MINUS-seasoned spread (beta-controlled).

    The raw young and seasoned books carry different betas (young ~1.6 vs
    seasoned ~1.4), so a raw return spread confounds beta with the launch-timing
    effect. We regress the spread on the market excess return; the intercept is
    the *extra* alpha bleed attributable to being young rather than merely
    thematic — the launch-timing component per se.
    """
    py, _ = calendar_time_portfolio(univ, tickers, 1, w)
    ps, _ = calendar_time_portfolio(univ, tickers, w + 1, 10_000)
    df = pd.concat({"y": py, "s": ps, "m": univ["mkt"], "rf": univ["rf"]},
                   axis=1).dropna()
    n = len(df)
    if n < 24:
        return {"alpha_ann_pct": float("nan"), "t_alpha": float("nan"),
                "beta": float("nan"), "n": n}
    yv = (df["y"] - df["s"]).to_numpy()
    x = (df["m"] - df["rf"]).to_numpy()
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ yv)
    u = yv - X @ b
    g = X * u[:, None]
    S = g.T @ g
    L = min(lags, n - 1)
    for k in range(1, L + 1):
        gk = g[k:].T @ g[:-k]
        S += (1.0 - k / (L + 1.0)) * (gk + gk.T)
    V = XtX_inv @ S @ XtX_inv
    se_a = float(np.sqrt(max(V[0, 0], 1e-18)))
    return {"alpha_ann_pct": float(b[0]) * MONTHS * 100.0, "beta": float(b[1]),
            "t_alpha": float(b[0] / se_a), "n": n}


# --------------------------------------------------------------------------- #
# Event-time alpha curve (descriptive)
# --------------------------------------------------------------------------- #
def fund_beta(univ: dict, tk: str) -> float:
    """Full-sample CAPM beta of one fund (used for the event curve only)."""
    r = univ["mret"][tk]
    df = pd.concat({"p": r, "m": univ["mkt"], "rf": univ["rf"]}, axis=1).dropna()
    if len(df) < 24:
        return float("nan")
    y = (df["p"] - df["rf"]).to_numpy()
    x = (df["m"] - df["rf"]).to_numpy()
    vx = x.var()
    return float(np.cov(x, y, ddof=0)[0, 1] / vx) if vx > 0 else float("nan")


def event_curve(univ: dict, tickers: list[str], k_max: int = 48) -> pd.DataFrame:
    """Mean CAPM-abnormal return by event month + cumulative sum + fund counts.

    Abnormal_t = (r_t - rf_t) - beta_i (mkt_t - rf_t) with the fund's FULL-SAMPLE
    beta — a documented look-ahead acceptable for a descriptive curve (the
    headline alpha uses no such thing).
    """
    rows = {k: [] for k in range(1, k_max + 1)}
    for tk in tickers:
        b = fund_beta(univ, tk)
        if not np.isfinite(b):
            continue
        k = event_months(univ, tk)
        r = univ["mret"][tk].dropna()
        ab = (r - univ["rf"].reindex(r.index)) - b * (
            univ["mkt"].reindex(r.index) - univ["rf"].reindex(r.index))
        for dt, kk in k.items():
            if 1 <= kk <= k_max and np.isfinite(ab.get(dt, np.nan)):
                rows[kk].append(float(ab[dt]))
    rec = []
    for kk in range(1, k_max + 1):
        v = np.asarray(rows[kk], dtype=float)
        rec.append({"k": kk, "n": len(v),
                    "mean_ab": float(v.mean()) if len(v) else np.nan})
    df = pd.DataFrame(rec).set_index("k")
    df["cum_ab_pct"] = df["mean_ab"].fillna(0.0).cumsum() * 100.0
    return df


# --------------------------------------------------------------------------- #
# Per-fund descriptive alphas
# --------------------------------------------------------------------------- #
def per_fund_alphas(univ: dict, tickers: list[str], w: int | None = 36) -> pd.DataFrame:
    """Per-fund CAPM alpha over its first ``w`` months (or full life if None)."""
    rows = []
    for tk in tickers:
        r = univ["mret"][tk].dropna()
        if w is not None:
            k = event_months(univ, tk)
            r = r.reindex(k[(k >= 1) & (k <= w)].index).dropna()
        res = capm_alpha_nw(r, univ["mkt"], univ["rf"])
        if np.isfinite(res["alpha_ann_pct"]):
            rows.append({"ticker": tk, "alpha_ann_pct": res["alpha_ann_pct"],
                         "beta": res["beta"], "n": res["n"]})
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Third axis — buy the -50% dip
# --------------------------------------------------------------------------- #
def dip_events(univ: dict, tickers: list[str], dd: float = 0.50) -> dict:
    """First month-end each fund closes >= ``dd`` below its post-launch running max.

    Uses month-end total-return levels rebuilt from monthly returns (level 1 at
    the end of event month 0). Returns {ticker: signal month-end Timestamp}.
    """
    out = {}
    for tk in tickers:
        r = univ["mret"][tk].dropna()
        if len(r) < 6:
            continue
        lvl = (1.0 + r).cumprod()
        drawdown = lvl / lvl.cummax() - 1.0
        hit = drawdown[drawdown <= -dd]
        if len(hit):
            out[tk] = hit.index[0]
    return out


def dip_portfolio(univ: dict, tickers: list[str], dd: float = 0.50,
                  hold: int = 36, lags: int = NW_LAGS,
                  min_members: int = 3) -> dict:
    """Calendar-time book of dip-bought funds: enter the month AFTER the signal, hold ``hold`` months.

    The dip signal is observed at the month-end close; the position starts with the
    next month's return — one clean execution lag. CAPM alpha + NW t of the book,
    plus per-event forward 12m and 36m total returns vs SPY over the same windows.
    """
    ev = dip_events(univ, tickers, dd=dd)
    mret = univ["mret"]
    cols = {}
    fwd12, fwd36 = [], []
    for tk, t0 in ev.items():
        r = mret[tk].dropna()
        after = r[r.index > t0]
        take = after.iloc[:hold]
        if len(take) >= 6:
            cols[tk] = take
        spy = univ["mkt"].reindex(after.index)
        for horizon, sink in ((12, fwd12), (36, fwd36)):
            seg = after.iloc[:horizon]
            if len(seg) == horizon:
                fund_tr = float((1 + seg).prod() - 1)
                spy_tr = float((1 + spy.reindex(seg.index)).prod() - 1)
                sink.append(fund_tr - spy_tr)
    if not cols:
        return {"n_events": len(ev), "alpha_ann_pct": float("nan"),
                "t_alpha": float("nan"), "beta": float("nan"), "n": 0,
                "fwd12": fwd12, "fwd36": fwd36}
    panel = pd.DataFrame(cols)
    cnt = panel.notna().sum(axis=1)
    port = panel.mean(axis=1)[cnt >= min_members].dropna()
    res = capm_alpha_nw(port, univ["mkt"], univ["rf"], lags=lags)
    res.update({"n_events": len(ev), "avg_members": float(cnt[cnt >= min_members].mean()),
                "fwd12": fwd12, "fwd36": fwd36})
    return res


# --------------------------------------------------------------------------- #
# Tradability — the short implementation
# --------------------------------------------------------------------------- #
def short_overlay(univ: dict, tickers: list[str], w: int = 36,
                  cost_bps: float = 5.0, borrow_bps_ann: float = 300.0,
                  lags: int = NW_LAGS) -> dict:
    """Short the young-thematics book, hedge with SPY long, charge costs + borrow.

    Monthly P&L = -(port - rf) + beta_hat * (mkt - rf) - borrow/12 - turnover costs,
    with beta_hat the regression beta of the young portfolio (the realistic hedge
    ratio a desk would run). Turnover: membership entries/exits — each fund's
    entry + exit is a round trip of 2 x cost_bps on its sleeve; charged as
    (2 * cost_bps / w) per month per unit NAV (amortized), plus the same on the
    hedge. Borrow is paid on the FULL short notional. Alpha framing: the net
    monthly mean and its NW t.
    """
    port, _cnt = calendar_time_portfolio(univ, tickers, 1, w)
    capm = capm_alpha_nw(port, univ["mkt"], univ["rf"], lags=lags)
    beta = capm["beta"]
    df = pd.concat({"p": port, "m": univ["mkt"], "rf": univ["rf"]}, axis=1).dropna()
    gross = -(df["p"] - df["rf"]) + beta * (df["m"] - df["rf"])
    amort = 2.0 * (cost_bps / 1e4) / w            # fund sleeve round trips
    hedge = 2.0 * (cost_bps / 1e4) / w * abs(beta)
    borrow = (borrow_bps_ann / 1e4) / MONTHS
    net = gross - amort - hedge - borrow
    mu_g, _, t_g = nw_tstat(gross, lags=lags)
    mu_n, _, t_n = nw_tstat(net, lags=lags)
    return {"gross_ann_pct": mu_g * MONTHS * 100, "t_gross": t_g,
            "net_ann_pct": mu_n * MONTHS * 100, "t_net": t_n,
            "beta_hedge": beta, "borrow_bps_ann": borrow_bps_ann,
            "cost_bps": cost_bps, "n": len(df)}

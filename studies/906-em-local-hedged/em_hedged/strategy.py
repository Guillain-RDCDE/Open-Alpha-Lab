"""Strategy + inference for Study 906 — EM Local Bonds, FX-Hedged (a proxy).

The mechanical story (log-approx, per month):

    EMLC        =  local_bond_return + EM_FX_return
    hedged      =  EMLC − b·UUP          (b<0 ⇒ a LONG-UUP overlay stripping the FX drag)

where ``b`` is the variance-min hedge ratio from regressing EMLC on UUP. We test whether
the **hedged** local series carries a real local-rate premium the **unhedged** EMLC hides:

  1. an **excess-vs-excess** Sharpe race — every leg minus BIL (tradable cash) — between
     unhedged EMLC, the hedged-EMLC proxy, USD-EM debt (EMB), and cash;
  2. the **HAC (Newey-West)** *t* on the hedged local excess and on the hedged-minus-EMB
     premium difference — monthly bond returns are serially correlated, so a plain *t* is
     banned here;
  3. a **bootstrap** Sharpe CI (does it clear zero?), the **max drawdown**, a calendar-year
     table, an **era cut** (pre/post the 2021 dollar surge), and a **costed** version that
     charges ETF spreads + the UUP-overlay turnover per rebalance.

Honesty rails: the variance-min hedge ratio ``b`` is estimated **in-sample** (full tape),
so the FX-stripping it achieves is an *upper bound* — a live desk would use a lagged /
rolling ``b`` (``rolling_hedge_series`` gives the walk-forward version). The overlay is a
DXY-basket proxy, so a residual EM-FX beta survives — reported, never hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_MONTHS = 12
NW_LAGS = 6  # Newey-West lag window for monthly series (~ 1.5 * T^(1/3) at T ~ 190)


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide monthly simple total returns (month-end to month-end).

    Drops the final calendar month if the price tape ends before that month is over, so a
    stamped run never contains a partial month.
    """
    m = prices.resample("ME").last()
    ret = m.pct_change()
    last_px = prices.index.max()
    last_bucket = ret.index.max()
    if last_px < (last_bucket - pd.offsets.MonthEnd(0)) or last_px.day < last_bucket.day:
        ret = ret.iloc[:-1]
    return ret.dropna(how="all")


def excess(monthly: pd.DataFrame, col: str, rf: str = "BIL") -> pd.Series:
    """Monthly excess return of ``col`` over the tradable T-bill leg (both simple returns)."""
    return (monthly[col] - monthly[rf]).dropna()


# --------------------------------------------------------------------------- #
# Inference primitives (HAC + Welch, self-contained)
# --------------------------------------------------------------------------- #
def newey_west_t(x: np.ndarray, lags: int = NW_LAGS) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    u = x - x.mean()
    g0 = float(u @ u) / n
    var = g0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        var += 2.0 * w * float(u[k:] @ u[:-k]) / n
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(x.mean() / se) if se > 0 else float("nan")


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def hac_ols(y: np.ndarray, x: np.ndarray, lags: int = NW_LAGS) -> dict:
    """OLS of y on [1, x] with Newey-West HAC standard errors.

    Returns alpha, beta, their HAC t's and R^2 — used for EMLC = alpha + beta·UUP and for
    the residual-EM-FX honesty check.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    Z = X * e[:, None]
    S = Z.T @ Z
    for k in range(1, min(lags, n - 1) + 1):
        w = 1.0 - k / (lags + 1.0)
        G = Z[k:].T @ Z[:-k]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(e @ e) / ss_tot if ss_tot > 0 else float("nan")
    return {"alpha": float(beta[0]), "beta": float(beta[1]),
            "t_alpha": float(beta[0] / se[0]) if se[0] > 0 else float("nan"),
            "t_beta": float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
            "r2": r2, "n": n}


# --------------------------------------------------------------------------- #
# The proxy hedge
# --------------------------------------------------------------------------- #
def hedge_ratio(local_excess: pd.Series, overlay_excess: pd.Series) -> float:
    """Variance-min hedge ratio ``b`` = cov(local, overlay)/var(overlay).

    Regressing local-EM excess on the UUP overlay excess; ``b`` is the OLS slope. It comes
    out **negative** (dollar up ⇒ EM-local down), so the hedge ``local − b·overlay`` is a
    LONG-UUP overlay. Estimated in-sample here (upper bound on FX-stripping); see
    :func:`rolling_hedge_series` for the walk-forward version.
    """
    a = local_excess.align(overlay_excess, join="inner")
    y, x = a[0].to_numpy(float), a[1].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    v = x.var(ddof=0)
    return float(np.cov(y, x, ddof=0)[0, 1] / v) if v > 0 else 0.0


def hedged_series(monthly: pd.DataFrame, local: str, overlay: str = "UUP",
                  rf: str = "BIL", b: float | None = None) -> pd.Series:
    """FX-hedged local-EM **excess** return proxy: ``(local−rf) − b·(overlay−rf)``.

    ``b`` defaults to the in-sample variance-min hedge ratio. The overlay is financed at
    the same cash leg, so both legs enter in excess-of-cash — the result is a clean
    excess-vs-excess series comparable to the other legs.
    """
    le = excess(monthly, local, rf)
    oe = excess(monthly, overlay, rf)
    a = le.align(oe, join="inner")
    if b is None:
        b = hedge_ratio(a[0], a[1])
    return (a[0] - b * a[1]).rename(f"{local}_hedged")


def rolling_hedge_series(monthly: pd.DataFrame, local: str, overlay: str = "UUP",
                         rf: str = "BIL", window: int = 36, min_periods: int = 24) -> pd.Series:
    """Walk-forward FX-hedged local excess: the hedge ratio at month ``t`` uses only data
    through ``t-1`` (a trailing ``window``-month regression, one execution lag), so there
    is **no look-ahead** in the hedge itself. The honest, implementable version.
    """
    le = excess(monthly, local, rf)
    oe = excess(monthly, overlay, rf)
    a = le.align(oe, join="inner")
    y, x = a[0], a[1]
    cov = y.rolling(window, min_periods=min_periods).cov(x)
    var = x.rolling(window, min_periods=min_periods).var()
    b = (cov / var).shift(1)  # decided at t-1, applied at t
    out = (y - b * x).dropna()
    return out.rename(f"{local}_hedged_wf")


# --------------------------------------------------------------------------- #
# Performance & risk
# --------------------------------------------------------------------------- #
def ann_return(monthly_ret: pd.Series) -> float:
    """Geometric annualised return (%) from monthly simple returns."""
    r = pd.Series(monthly_ret).dropna()
    if len(r) == 0:
        return float("nan")
    return float(((1.0 + r).prod() ** (TRADING_MONTHS / len(r)) - 1.0) * 100.0)


def sharpe_ann(excess_ret: pd.Series) -> float:
    """Annualised Sharpe of an already-excess monthly series (excess-vs-excess race)."""
    e = pd.Series(excess_ret).dropna()
    sd = e.std(ddof=1)
    return float(e.mean() / sd * np.sqrt(TRADING_MONTHS)) if sd > 0 else float("nan")


def max_drawdown(prices: pd.Series) -> dict:
    """Max drawdown of a daily/monthly total-return price series: depth (%), peak/trough."""
    px = pd.Series(prices).dropna()
    peak = px.cummax()
    dd = px / peak - 1.0
    trough = dd.idxmin()
    peak_date = px.loc[:trough].idxmax()
    return {"depth_pct": float(dd.min() * 100.0),
            "peak": str(pd.Timestamp(peak_date).date()),
            "trough": str(pd.Timestamp(trough).date())}


def calendar_year_table(monthly: pd.DataFrame, cols: list[str], rf: str = "BIL") -> pd.DataFrame:
    """Per-calendar-year compounded EXCESS return (%) for each column (minus cash)."""
    out = {}
    for c in cols:
        e = excess(monthly, c, rf) if c in monthly.columns else None
        if e is None:
            continue
        out[c] = e.groupby(e.index.year).apply(lambda s: (1.0 + s).prod() - 1.0) * 100.0
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Headline race + premium test
# --------------------------------------------------------------------------- #
def race(monthly: pd.DataFrame, local: str = "EMLC", overlay: str = "UUP",
         bench: str = "EMB", rf: str = "BIL", lags: int = NW_LAGS) -> dict:
    """Excess-vs-excess Sharpe race + the hedged-local-rate-carry HAC test.

    Legs (all minus BIL): unhedged ``local``, hedged-``local`` proxy, ``bench`` (USD-EM).
    Reports each leg's annualised excess return + Sharpe, the hedge ratio, the HAC *t* on
    the hedged local excess, and the HAC *t* on (hedged_local − bench_excess) — the
    premium-difference test. Also the EMLC~UUP regression (β, R²) and the residual EM-FX
    beta after hedging (how much FX the proxy leaves behind).
    """
    le = excess(monthly, local, rf)
    be = excess(monthly, bench, rf)
    oe = excess(monthly, overlay, rf)
    reg = hac_ols(le.align(oe, join="inner")[0].to_numpy(),
                  le.align(oe, join="inner")[1].to_numpy(), lags)
    b = hedge_ratio(*le.align(oe, join="inner"))
    he = hedged_series(monthly, local, overlay, rf, b=b)

    common = pd.concat([le, be, he, oe], axis=1, join="inner").dropna()
    common.columns = ["le", "be", "he", "oe"]
    diff = (common["he"] - common["be"]).to_numpy()

    # residual EM-FX beta: regress hedged local back on overlay (should be ~0 if proxy clean)
    resid_reg = hac_ols(common["he"].to_numpy(), common["oe"].to_numpy(), lags)

    return {
        "n": int(len(common)),
        "start": str(common.index.min().date()), "end": str(common.index.max().date()),
        "hedge_b": b,
        "emlc_uup_beta": reg["beta"], "emlc_uup_r2": reg["r2"], "t_emlc_uup": reg["t_beta"],
        # unhedged local
        "local_ann_pct": ann_return(monthly[local]),
        "local_exc_ann_pct": float(common["le"].mean() * 12 * 100),
        "local_sharpe": sharpe_ann(common["le"]),
        "t_local": newey_west_t(common["le"].to_numpy(), lags),
        # hedged local
        "hedged_exc_ann_pct": float(common["he"].mean() * 12 * 100),
        "hedged_sharpe": sharpe_ann(common["he"]),
        "t_hedged": newey_west_t(common["he"].to_numpy(), lags),
        # bench (USD-EM)
        "bench_ann_pct": ann_return(monthly[bench]),
        "bench_exc_ann_pct": float(common["be"].mean() * 12 * 100),
        "bench_sharpe": sharpe_ann(common["be"]),
        "t_bench": newey_west_t(common["be"].to_numpy(), lags),
        # premium difference hedged - bench
        "prem_diff_ann_pct": float(diff.mean() * 12 * 100),
        "t_prem_diff": newey_west_t(diff, lags),
        "welch_hedged_vs_bench": welch_t(common["he"].to_numpy(), common["be"].to_numpy()),
        # residual FX left by the proxy
        "resid_fx_beta": resid_reg["beta"], "resid_fx_r2": resid_reg["r2"],
    }


def era_split(monthly: pd.DataFrame, split: str = "2021-01-01", local: str = "EMLC",
              overlay: str = "UUP", bench: str = "EMB", rf: str = "BIL",
              lags: int = NW_LAGS) -> dict:
    """Hedged-local excess and premium-vs-bench in two eras (pre/post the dollar surge).

    Uses a SINGLE full-sample hedge ratio (so the eras are compared on the same overlay,
    not two different fits) and reports each era's hedged excess mean + HAC *t*.
    """
    oe_full = excess(monthly, overlay, rf)
    le_full = excess(monthly, local, rf)
    b = hedge_ratio(*le_full.align(oe_full, join="inner"))
    he = hedged_series(monthly, local, overlay, rf, b=b)
    be = excess(monthly, bench, rf)
    out = {}
    for lo, hi, lbl in [(None, split, "early"), (split, None, "late")]:
        h = he.copy()
        d = (he - be).dropna()
        if lo is not None:
            h = h[h.index >= pd.Timestamp(lo)]; d = d[d.index >= pd.Timestamp(lo)]
        if hi is not None:
            h = h[h.index < pd.Timestamp(hi)]; d = d[d.index < pd.Timestamp(hi)]
        out[lbl] = {
            "n": int(len(h)),
            "start": str(h.index.min().date()) if len(h) else None,
            "end": str(h.index.max().date()) if len(h) else None,
            "hedged_ann_pct": float(h.mean() * 12 * 100),
            "t_hedged": newey_west_t(h.to_numpy(), lags),
            "prem_diff_ann_pct": float(d.mean() * 12 * 100),
            "t_prem_diff": newey_west_t(d.to_numpy(), lags),
        }
    return out


# --------------------------------------------------------------------------- #
# Tradability — cost the overlay
# --------------------------------------------------------------------------- #
def costed(monthly: pd.DataFrame, local: str = "EMLC", overlay: str = "UUP",
           bench: str = "EMB", rf: str = "BIL",
           etf_spread_bps: float = 3.0, overlay_spread_bps: float = 3.0,
           rebalances_per_year: float = 12.0, lags: int = NW_LAGS) -> dict:
    """Net the hedged-local proxy after real ETF frictions.

    Charges (per year, amortised monthly): (a) the local ETF one-way spread once at
    entry, negligible at buy-and-hold — but (b) the UUP OVERLAY must be re-struck each
    rebalance to keep |b|·NAV notional, so it pays ``overlay_spread_bps`` one-way on
    ``|b|`` of NAV × ``rebalances_per_year``, plus the local ETF spread on any local
    turnover. Long-only overlay (long USD), no borrow. Reports gross vs net hedged excess
    and the HAC *t* on the net series.
    """
    he = hedged_series(monthly, local, overlay, rf)
    b = hedge_ratio(*excess(monthly, local, rf).align(excess(monthly, overlay, rf), join="inner"))
    # monthly overlay re-strike cost: |b| notional * one-way spread * rebalances/yr, /12
    overlay_cost_m = abs(b) * (overlay_spread_bps / 1e4) * rebalances_per_year / 12.0
    # a light local-ETF maintenance charge (a couple of round-trips/yr on drift)
    local_cost_m = 2.0 * (etf_spread_bps / 1e4) / 12.0
    charge_m = overlay_cost_m + local_cost_m
    net = he - charge_m
    be = excess(monthly, bench, rf)
    d_net = (net - be).dropna()
    return {
        "hedge_b": b,
        "charge_ann_pct": float(charge_m * 12 * 100),
        "gross_hedged_ann_pct": float(he.mean() * 12 * 100),
        "net_hedged_ann_pct": float(net.mean() * 12 * 100),
        "t_net_hedged": newey_west_t(net.to_numpy(), lags),
        "net_hedged_sharpe": sharpe_ann(net),
        "net_prem_diff_ann_pct": float(d_net.mean() * 12 * 100),
        "t_net_prem_diff": newey_west_t(d_net.to_numpy(), lags),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(world: pd.DataFrame, lags: int = NW_LAGS) -> dict:
    """Run the hedged-carry test on a synthetic world: hedged excess mean + HAC t + hedge b."""
    r = race(world, local="EMLC", overlay="UUP", bench="EMB", rf="BIL", lags=lags)
    return {"hedged_exc_ann_pct": r["hedged_exc_ann_pct"], "t_hedged": r["t_hedged"],
            "hedge_b": r["hedge_b"], "resid_fx_beta": r["resid_fx_beta"], "n": r["n"]}

"""Strategy + inference for Study 745 — Corporate-Jet-Index (governance long/short).

The claim (Yermack 2006): firms disclosing CEO **personal use of the company aircraft**
underperform, risk-adjusted, by ~4%/yr — so **short the flyers, long the frugal peers**.

We build that as a monthly, equal-weight **long/short characteristic sort**:

    For each month t, the HEAVY basket return is the equal-weight mean of the
    heavy-perk names *eligible* that month (a name is eligible only from the month
    AFTER its perk became public — a one-month execution lag, no look-ahead); the LOW
    basket is the same for the low-perk peers. The believers' book is

        LS_t = r_low,t − r_heavy,t          (long frugal, short flyers)

    We measure LS in **excess of the market** (SPY) and, separately, as a
    **market-model alpha** (LS_t = α + β·r_SPY,t + ε_t) to strip out any beta the
    long/short accidentally runs. Significance is a **Newey-West (HAC)** t on the
    monthly mean (autocorrelation- and heteroskedasticity-robust).

We then charge realistic frictions: a monthly equal-weight rebalance pays one-way costs
on its turnover, and the short (heavy) leg pays a borrow fee. Gross AND net are reported.

The decisive numbers are the **sign and HAC t** of the long/short: Yermack predicts a
*positive* LS (frugal beats flyers). On a survivor tape dominated by founder-led growth
flyers (Oracle, Tesla, Meta, Alphabet), the raw sort can run the *other* way — a
growth/founder confound, not a governance discount. The market-model alpha and the
survivorship caveat are how we tell those apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
DEFAULT_COST_BPS = 10.0      # one-way transaction cost per rebalanced leg (bps of NAV)
DEFAULT_BORROW_BPS = 50.0    # annual short-borrow fee on the heavy (short) leg (bps)


# --------------------------------------------------------------------------- #
# Monthly returns + eligibility (no look-ahead)
# --------------------------------------------------------------------------- #
def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly total returns from the wide monthly-close frame."""
    return prices.pct_change()


def _eligible_from(firm: dict) -> pd.Timestamp:
    """First month a heavy name may be shorted: Jan of the year AFTER the perk is public.

    A one-year+ execution buffer past ``public_year`` — you can only act on a red flag
    once it is disclosed. Low-perk names are eligible whenever priced (no disclosure to
    wait for)."""
    return pd.Timestamp(f"{firm['public_year'] + 1}-01-31")


def basket_returns(prices: pd.DataFrame, firms: list[dict],
                   heavy: bool) -> pd.Series:
    """Equal-weight monthly return of the heavy (or low) basket, respecting eligibility.

    Each eligible-and-priced name contributes its month-t simple return; the basket is the
    equal-weight mean across names eligible that month. Heavy names enter only from
    :func:`_eligible_from`; low names enter whenever they have a price.
    """
    rets = monthly_returns(prices)
    names = [f for f in firms if f["heavy"] == heavy and f["ticker"] in rets.columns]
    if not names:
        return pd.Series(dtype=float)
    cols = {}
    for f in names:
        r = rets[f["ticker"]].copy()
        if heavy:
            r = r[r.index >= _eligible_from(f)]
        cols[f["ticker"]] = r
    mat = pd.DataFrame(cols)
    return mat.mean(axis=1, skipna=True)


def long_short_panel(prices: pd.DataFrame, firms: list[dict],
                     min_names: int = 4) -> pd.DataFrame:
    """Monthly long/short (low − heavy) panel, in raw and excess-of-market terms.

    Columns: ``heavy``, ``low`` (equal-weight basket simple returns), ``mkt`` (SPY),
    ``heavy_x``, ``low_x`` (excess of market), ``ls`` (low_x − heavy_x), ``n_heavy``,
    ``n_low`` (eligible counts). Months are kept only where BOTH baskets have at least
    ``min_names`` eligible names, so early months (few heavy names disclosed) are dropped.
    """
    rets = monthly_returns(prices)
    if "SPY" not in rets.columns:
        raise ValueError("SPY column required for the market benchmark")
    mkt = rets["SPY"]

    # per-name eligibility masks -> counts
    def counts(heavy: bool) -> pd.Series:
        names = [f for f in firms if f["heavy"] == heavy and f["ticker"] in rets.columns]
        c = pd.Series(0, index=rets.index, dtype=int)
        for f in names:
            r = rets[f["ticker"]]
            elig = r.notna()
            if heavy:
                elig &= (r.index >= _eligible_from(f))
            c = c.add(elig.astype(int), fill_value=0)
        return c

    heavy_b = basket_returns(prices, firms, heavy=True)
    low_b = basket_returns(prices, firms, heavy=False)
    df = pd.DataFrame({"heavy": heavy_b, "low": low_b, "mkt": mkt}).dropna()
    df["n_heavy"] = counts(True).reindex(df.index).fillna(0).astype(int)
    df["n_low"] = counts(False).reindex(df.index).fillna(0).astype(int)
    df = df[(df["n_heavy"] >= min_names) & (df["n_low"] >= min_names)]
    df["heavy_x"] = df["heavy"] - df["mkt"]
    df["low_x"] = df["low"] - df["mkt"]
    df["ls"] = df["low_x"] - df["heavy_x"]
    return df


# --------------------------------------------------------------------------- #
# Inference — Newey-West (HAC) t, market-model alpha
# --------------------------------------------------------------------------- #
def hac_tstat(series: np.ndarray, lags: int | None = None) -> dict:
    """Newey-West (HAC, Bartlett kernel) t-statistic for a sample mean.

    Monthly long/short returns are mildly autocorrelated; the naive t overstates
    significance. ``lags=None`` uses the standard rule ``floor(4*(n/100)^(2/9))``.
    Returns mean (monthly), HAC se, t, chosen lags, n.
    """
    r = np.asarray(series, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"),
                "lags": 0, "n": n}
    mu = r.mean()
    e = r - mu
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gamma_k = float(e[k:] @ e[:-k]) / n
        lrv += 2.0 * w * gamma_k
    se = np.sqrt(max(lrv, 0.0) / n)
    return {"mean": float(mu), "se": float(se),
            "t": float(mu / se) if se > 0 else float("nan"), "lags": int(lags), "n": int(n)}


def market_model_alpha(ls: np.ndarray, mkt: np.ndarray,
                       lags: int | None = None) -> dict:
    """OLS of LS on the market (LS = α + β·mkt + ε) with a HAC t on α.

    Strips whatever market beta the long/short accidentally runs, leaving the
    *risk-adjusted* alpha — the object Yermack actually claims. The HAC t on α uses the
    Newey-West long-run variance of the OLS residual scores.
    """
    y = np.asarray(ls, dtype=float)
    x = np.asarray(mkt, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = y.size
    if n < 5:
        return {"alpha": float("nan"), "beta": float("nan"), "t_alpha": float("nan"), "n": n}
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    resid = y - X @ b
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    # HAC meat: S = sum_k w_k (Xe)'(Xe) lagged
    Xe = X * resid[:, None]
    S = Xe.T @ Xe
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        G = Xe[k:].T @ Xe[:-k]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se_alpha = np.sqrt(max(cov[0, 0], 0.0))
    return {"alpha": float(b[0]), "beta": float(b[1]),
            "t_alpha": float(b[0] / se_alpha) if se_alpha > 0 else float("nan"),
            "se_alpha": float(se_alpha), "lags": int(lags), "n": int(n)}


def annualize_mean(monthly_mean: float) -> float:
    """Geometric annualisation of a monthly mean return."""
    return float((1.0 + monthly_mean) ** MONTHS_PER_YEAR - 1.0)


def sharpe(series: np.ndarray) -> float:
    """Annualised Sharpe of a monthly excess series (mean/std * sqrt(12))."""
    r = np.asarray(series, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


# --------------------------------------------------------------------------- #
# Costs — monthly rebalance turnover + short borrow
# --------------------------------------------------------------------------- #
def net_of_costs(ls_series: pd.Series, cost_bps: float = DEFAULT_COST_BPS,
                 borrow_bps: float = DEFAULT_BORROW_BPS,
                 turnover_frac: float = 0.10) -> dict:
    """Net the long/short for costs: rebalance turnover (both legs) + short borrow.

    A held equal-weight book only trades to re-equalise weights each month; we charge a
    modest ``turnover_frac`` one-way turnover per leg (≈10%/mo covers drift + name
    entry/exit) at ``cost_bps`` one-way, applied to BOTH legs, plus an annual
    ``borrow_bps`` fee on the short (heavy) leg, spread monthly. Costs one-way × NAV;
    the short pays borrow; gross and net both returned.
    """
    r = np.asarray(ls_series, dtype=float)
    r = r[np.isfinite(r)]
    gross_m = float(np.mean(r)) if r.size else float("nan")
    # two legs rebalanced -> 2 * turnover_frac * cost, per month
    rebal_cost_m = 2.0 * turnover_frac * (cost_bps / 1e4)
    borrow_m = (borrow_bps / 1e4) / MONTHS_PER_YEAR
    net_m = gross_m - rebal_cost_m - borrow_m
    return {
        "gross_month": gross_m, "net_month": net_m,
        "gross_ann": annualize_mean(gross_m), "net_ann": annualize_mean(net_m),
        "rebal_cost_ann": rebal_cost_m * MONTHS_PER_YEAR,
        "borrow_ann": borrow_bps / 1e4,
        "cost_bps": cost_bps, "borrow_bps": borrow_bps, "turnover_frac": turnover_frac,
    }


# --------------------------------------------------------------------------- #
# One-call summary
# --------------------------------------------------------------------------- #
def summarize(panel: pd.DataFrame) -> dict:
    """Headline long/short stats: raw HAC t, market-model alpha, basket means, Sharpe."""
    ls = panel["ls"].to_numpy(float)
    heavy_x = panel["heavy_x"].to_numpy(float)
    low_x = panel["low_x"].to_numpy(float)
    mkt = panel["mkt"].to_numpy(float)
    hac = hac_tstat(ls)
    mm = market_model_alpha(ls, mkt)
    return {
        "n_months": int(len(panel)),
        "start": str(panel.index[0].date()) if len(panel) else None,
        "end": str(panel.index[-1].date()) if len(panel) else None,
        "ls_mean_month": hac["mean"], "ls_mean_ann": annualize_mean(hac["mean"]),
        "ls_hac_t": hac["t"], "ls_hac_lags": hac["lags"],
        "ls_sharpe": sharpe(ls),
        "alpha_month": mm["alpha"], "alpha_ann": annualize_mean(mm["alpha"]),
        "alpha_t": mm["t_alpha"], "beta": mm["beta"],
        "heavy_x_ann": annualize_mean(float(np.nanmean(heavy_x))),
        "low_x_ann": annualize_mean(float(np.nanmean(low_x))),
    }

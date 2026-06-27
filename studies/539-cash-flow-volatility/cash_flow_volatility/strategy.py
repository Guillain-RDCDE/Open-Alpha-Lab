"""The Cash-Flow-Volatility strategy -- Huang (2009) cash-flow-uncertainty anomaly.

Claim under test: firms with more *volatile* operating cash flows earn LOWER subsequent
returns. So the bet is a quality-like long-short:

  * LONG  the low-CF-volatility tercile (stable cash flows),
  * SHORT the high-CF-volatility tercile (uncertain cash flows),
  * equal-weight within each leg, held over the out-of-sample price window.

Signal construction (real tape):
  1. For each name, ``cfvol`` = std of (quarterly operating cash flow / total assets) over
     the available yfinance quarterly history (formed BEFORE the return window starts).
  2. Cross-sectionally rank names by ``cfvol``; split into terciles.
  3. The long-short return each month is (low-tercile mean) - (high-tercile mean).

Execution lag (exactly one, documented): the signal is a stale cross-sectional
characteristic; the first held month begins one trading day AFTER the price window opens,
so there is no same-bar fill and no look-ahead from the formation history into the returns.

Inference: one-sample t-stat (Newey-West HAC) on the monthly long-short series; a
label-shuffle placebo null (permute the cfvol labels across names, re-sort, recompute);
and a deterministic synthetic positive control proving the engine recovers a planted
premium and returns noise under the null.

Costs: one-way bps x NAV x turnover, plus a borrow charge on the short leg. Reported gross
and net.

**Survivorship bias:** the basket is names still trading in 2026 -- named on the Signal axis.
**Fundamental-data limitation:** yfinance serves only a thin quarterly history, so cfvol
rests on few quarters; reported as a limitation, not hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Cross-sectional split: top/bottom tercile.
TERCILE = 1.0 / 3.0

# Costs (defaults; overridable).
ONE_WAY_BPS = 10.0          # 10 bps one-way transaction cost
BORROW_BPS_ANNUAL = 100.0   # 100 bps/yr stock-borrow on the short leg
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# HAC one-sample t and summary
# ---------------------------------------------------------------------------
def hac_tstat(returns: pd.Series) -> float:
    """Newey-West HAC one-sample t-stat that the mean of ``returns`` is zero."""
    arr = pd.Series(returns).astype(float).dropna().to_numpy()
    n = arr.size
    if n < 3:
        return float("nan")
    mu = arr.mean()
    e = arr - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summary(monthly_returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised stats for a monthly long-short return series + HAC t-stat."""
    r = pd.Series(monthly_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}
    mean_m = float(r.mean())
    vol_m = float(r.std(ddof=1))
    mean_ann = mean_m * periods_per_year
    vol_ann = vol_m * np.sqrt(periods_per_year)
    sr = mean_ann / vol_ann if vol_ann > 0 else float("nan")
    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "mean": mean_ann, "vol": vol_ann, "sharpe": sr,
        "tstat": hac_tstat(r), "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd, "n": n,
    }


# ---------------------------------------------------------------------------
# Cross-sectional long-short on a (month x stock) returns panel + cfvol labels
# ---------------------------------------------------------------------------
def long_short_from_labels(
    monthly_rets: pd.DataFrame,
    cfvol: pd.Series,
    tercile: float = TERCILE,
) -> pd.DataFrame:
    """Long low-CF-vol tercile, short high-CF-vol tercile; equal-weight each leg.

    ``monthly_rets`` is (period x ticker) monthly returns; ``cfvol`` is a per-ticker
    Series of the (stale, pre-window) cash-flow-volatility characteristic. The same sort
    is held for the whole window (a single formation, as the yfinance fundamentals give
    one usable snapshot).

    Returns a DataFrame indexed by period with columns:
      ``low_ret`` (long leg mean), ``high_ret`` (short leg mean),
      ``ls_ret`` = low_ret - high_ret, ``n_low``, ``n_high``.
    """
    cf = cfvol.dropna()
    cols = [c for c in monthly_rets.columns if c in cf.index]
    if len(cols) < 6:
        return pd.DataFrame()
    cf = cf.loc[cols].sort_values()
    k = max(1, int(np.floor(len(cf) * tercile)))
    low_names = list(cf.index[:k])         # lowest CF-vol  -> LONG
    high_names = list(cf.index[-k:])       # highest CF-vol -> SHORT

    low = monthly_rets[low_names].mean(axis=1)
    high = monthly_rets[high_names].mean(axis=1)
    out = pd.DataFrame({
        "low_ret": low, "high_ret": high, "ls_ret": low - high,
    })
    out["n_low"] = len(low_names)
    out["n_high"] = len(high_names)
    return out.dropna(subset=["ls_ret"])


# ---------------------------------------------------------------------------
# Real-tape pipeline: prices + cfvol snapshot -> monthly long-short
# ---------------------------------------------------------------------------
def monthly_returns_from_prices(prices: pd.DataFrame, exec_lag: int = 1) -> pd.DataFrame:
    """Month-end compounded returns from a daily price panel, with one execution-lag day.

    ``exec_lag`` drops the first ``exec_lag`` trading days of the panel so the first held
    month begins AFTER the signal is public (no same-bar fill). Returns a (month-period x
    ticker) frame of monthly decimal returns.
    """
    px = prices.iloc[exec_lag:] if exec_lag > 0 else prices
    daily = px.pct_change()
    monthly = (1.0 + daily).resample("ME").prod() - 1.0
    monthly = monthly.iloc[1:]  # first partial month is unreliable
    monthly.index = monthly.index.to_period("M")
    return monthly


def build_real(
    prices: pd.DataFrame,
    cfvol_panel: pd.DataFrame,
    tercile: float = TERCILE,
    exec_lag: int = 1,
) -> pd.DataFrame:
    """End-to-end real long-short: daily prices -> monthly LS using the cfvol snapshot."""
    if prices.empty or cfvol_panel.empty:
        return pd.DataFrame()
    monthly = monthly_returns_from_prices(prices, exec_lag=exec_lag)
    cf = cfvol_panel["cfvol"] if "cfvol" in cfvol_panel.columns else cfvol_panel.iloc[:, 0]
    return long_short_from_labels(monthly, cf, tercile=tercile)


# ---------------------------------------------------------------------------
# Placebo / label-shuffle null
# ---------------------------------------------------------------------------
def placebo_pvalue(
    monthly_rets: pd.DataFrame,
    cfvol: pd.Series,
    n_perm: int = 500,
    tercile: float = TERCILE,
    seed: int = 539,
) -> dict:
    """Label-shuffle null: permute the cfvol labels across names, re-sort, recompute mean LS.

    Returns the real mean LS, the permutation p-value (two-sided, fraction of |perm mean| >=
    |real mean|), and the null mean/std.
    """
    real = long_short_from_labels(monthly_rets, cfvol, tercile=tercile)
    if real.empty:
        return {"real_mean": float("nan"), "p_value": float("nan"),
                "null_mean": float("nan"), "null_std": float("nan"), "n_perm": 0}
    real_mean = float(real["ls_ret"].mean())

    cf = cfvol.dropna()
    cols = [c for c in monthly_rets.columns if c in cf.index]
    cf = cf.loc[cols]
    rng = np.random.default_rng(seed)
    null_means = np.empty(n_perm)
    vals = cf.values
    for i in range(n_perm):
        shuffled = pd.Series(rng.permutation(vals), index=cf.index)
        perm = long_short_from_labels(monthly_rets, shuffled, tercile=tercile)
        null_means[i] = float(perm["ls_ret"].mean()) if not perm.empty else np.nan
    null_means = null_means[np.isfinite(null_means)]
    if null_means.size == 0:
        p = float("nan")
    else:
        p = float((np.abs(null_means) >= abs(real_mean)).mean())
    return {
        "real_mean": real_mean, "p_value": p,
        "null_mean": float(np.nanmean(null_means)) if null_means.size else float("nan"),
        "null_std": float(np.nanstd(null_means)) if null_means.size else float("nan"),
        "n_perm": int(null_means.size),
    }


# ---------------------------------------------------------------------------
# Costs: one-way bps x turnover (+ borrow on the short leg)
# ---------------------------------------------------------------------------
def apply_costs(
    ls: pd.DataFrame,
    one_way_bps: float = ONE_WAY_BPS,
    borrow_bps_annual: float = BORROW_BPS_ANNUAL,
    monthly_turnover: float | None = None,
) -> dict:
    """Charge transaction + borrow costs against the gross long-short.

    Because the sort is formed once and held, the trading turnover is the initial
    establishment of both legs plus monthly equal-weight rebalancing drift. We model a
    conservative ``monthly_turnover`` (two-sided, fraction of NAV traded per month); the
    default assumes the sort holds (low turnover) but the legs are rebalanced to
    equal-weight monthly (~15%/month two-sided as a desk-standard placeholder).

    Costs per month = one_way_bps * turnover  +  borrow on the (1x notional) short leg.
    Returns gross/net annualised means, Sharpes and HAC t-stats.
    """
    r = ls["ls_ret"].astype(float).dropna()
    if r.empty:
        return {k: float("nan") for k in
                ("gross_mean", "net_mean", "gross_sharpe", "net_sharpe",
                 "gross_t", "net_t", "turnover", "cost_drag")}
    if monthly_turnover is None:
        monthly_turnover = 0.15  # two-sided NAV fraction traded per month (rebalance drift)

    tx_monthly = (one_way_bps / 1e4) * monthly_turnover
    borrow_monthly = (borrow_bps_annual / 1e4) / 12.0
    cost_monthly = tx_monthly + borrow_monthly
    net = r - cost_monthly

    g, n = summary(r), summary(net)
    return {
        "gross_mean": g["mean"], "net_mean": n["mean"],
        "gross_sharpe": g["sharpe"], "net_sharpe": n["sharpe"],
        "gross_t": g["tstat"], "net_t": n["tstat"],
        "turnover": float(monthly_turnover),
        "cost_drag": float(cost_monthly * 12.0),  # annualised drag
    }


# ---------------------------------------------------------------------------
# Synthetic positive control -- deterministic, multi-seed faithful-engine check
# ---------------------------------------------------------------------------
def synthetic_control(
    premiums: tuple[float, ...] = (0.0, 0.04, 0.08),
    n_seeds: int = 20,
    n_stocks: int = 60,
    n_months: int = 120,
) -> pd.DataFrame:
    """Faithful-engine / power check: does the long-short recover a PLANTED premium?

    For each planted ``cfvol_premium`` we average the long-short HAC t-stat over ``n_seeds``
    independent synthetic panels (Welch-style averaging, per the house rule for any
    synthetic-dependent claim). At premium 0 the engine must return ~noise (|t| small);
    at positive premiums the long-short mean must be positive and significant.

    This is ONLY an engine/power check -- it is never cited to support a real-tape stamp.
    """
    from . import data as _data

    rows: list[dict] = []
    for prem in premiums:
        means, tstats = [], []
        for s in range(n_seeds):
            rets, cfvol, _ = _data.synthetic_panel(
                n_stocks=n_stocks, n_months=n_months,
                cfvol_premium=prem, seed=539 + s,
            )
            snap = cfvol.iloc[-1]  # time-invariant characteristic
            ls = long_short_from_labels(rets, snap)
            s_ls = summary(ls["ls_ret"])
            means.append(s_ls["mean"])
            tstats.append(s_ls["tstat"])
        rows.append({
            "premium": prem,
            "mean_ls_%": float(np.nanmean(means)) * 100.0,
            "avg_t": float(np.nanmean(tstats)),
            "frac_sig": float(np.mean(np.abs(np.array(tstats)) >= 2.0)),
            "n_seeds": n_seeds,
        })
    return pd.DataFrame(rows)

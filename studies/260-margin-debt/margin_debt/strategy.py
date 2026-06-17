"""Strategy and honest controls -- Study 260 (Margin-Debt).

The folklore: "record / fast-rising margin debt is a contrarian SELL signal --
the market is over-leveraged and about to crash."  We test the predictive,
contrarian version on the real tape:

    Does the year-over-year change in NYSE/FINRA customer margin debt forecast
    the *next* 12 months' S&P 500 return, with a NEGATIVE sign (more leverage ->
    lower forward returns)?

Four honest comparisons settle the matter:

1. **Predictive regression** -- OLS of forward 12m return on standardized YoY
   margin growth, with a Newey-West HAC t-stat (overlapping-window aware, though
   we use non-overlapping calendar years).  The contrarian claim needs a
   *negative* slope with |t| >= 2.

2. **Extreme-tercile bins** -- forward return after the top third of YoY margin
   growth ("leverage surging") vs the bottom third ("leverage contracting").
   The contrarian story predicts surge-years underperform.

3. **The "record high" event test** -- forward return in the years when margin
   debt printed a fresh all-time high vs all other years.  This is the literal
   newspaper headline ("margin debt hits record").

4. **Buy-and-hold benchmark + a tradable contrarian rule** -- sit out (cash) the
   year after a margin surge / record, otherwise hold the S&P; net of a one-way
   cost on each in/out switch.  Compared against passive buy-and-hold.

**No look-ahead**: the YoY signal at year Y uses margin debt through December Y;
the position is entered after a one-month reporting/execution lag and the forward
return is the *next* 12 months.  **Price-only** returns (dividends excluded) --
which, for a contrarian "sell" claim, is the *conservative* direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def standardized_yoy(panel: pd.DataFrame, col: str = "yoy") -> pd.Series:
    """Z-score the YoY margin-growth column (full-sample standardization).

    Returned Series is aligned to ``panel.index``.  Full-sample standardization
    is fine here because we only use it descriptively / in-sample for the
    regression; the *sign* and *t-stat* are scale-invariant.
    """
    x = panel[col].astype(float)
    mu, sd = x.mean(), x.std(ddof=0)
    if sd == 0:
        return pd.Series(0.0, index=panel.index, name="yoy_z")
    return ((x - mu) / sd).rename("yoy_z")


# ---------------------------------------------------------------------------
# HAC-aware OLS (single regressor + intercept)
# ---------------------------------------------------------------------------
def ols_hac(y: np.ndarray, x: np.ndarray) -> dict:
    """OLS of y on [1, x] with a Newey-West HAC standard error on the slope.

    Returns a dict with ``alpha``, ``beta``, ``beta_t`` (HAC), ``r2``, ``n``.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y, x = y[mask], x[mask]
    n = y.size
    if n < 5:
        return {"alpha": np.nan, "beta": np.nan, "beta_t": np.nan, "r2": np.nan, "n": n}

    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef

    # Newey-West HAC covariance
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        Xr = X * resid[:, None]
        G = Xr[L:].T @ Xr[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    beta_se = float(np.sqrt(max(cov[1, 1], 0.0)))

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "beta_t": float(coef[1] / beta_se) if beta_se > 0 else np.nan,
        "r2": float(r2),
        "n": int(n),
    }


def predictive_regression(panel: pd.DataFrame) -> dict:
    """Regress forward 12m return on standardized YoY margin growth (HAC t-stat).

    The contrarian claim is a NEGATIVE beta with |t| >= 2: a one-sigma surge in
    margin growth should predict a meaningfully lower forward return.
    """
    z = standardized_yoy(panel)
    return ols_hac(panel["fwd_ret"].to_numpy(), z.to_numpy())


# ---------------------------------------------------------------------------
# Extreme-tercile bins
# ---------------------------------------------------------------------------
def tercile_forward_returns(panel: pd.DataFrame) -> dict:
    """Mean forward return after high/mid/low YoY margin-growth terciles.

    Returns a dict with the per-tercile mean forward return, the high-minus-low
    spread, and a Welch t-test of high vs low.  The contrarian story predicts
    ``high`` (leverage surging) < ``low`` (leverage contracting), i.e. a negative
    spread.
    """
    from scipy import stats as scipy_stats

    p = panel.dropna(subset=["yoy", "fwd_ret"]).copy()
    q1, q2 = p["yoy"].quantile([1 / 3, 2 / 3])
    low = p[p["yoy"] <= q1]["fwd_ret"].to_numpy()
    mid = p[(p["yoy"] > q1) & (p["yoy"] < q2)]["fwd_ret"].to_numpy()
    high = p[p["yoy"] >= q2]["fwd_ret"].to_numpy()

    if len(high) >= 2 and len(low) >= 2:
        wt, wp = scipy_stats.ttest_ind(high, low, equal_var=False)
    else:
        wt, wp = np.nan, np.nan

    return {
        "low_mean": float(low.mean()) if len(low) else np.nan,
        "mid_mean": float(mid.mean()) if len(mid) else np.nan,
        "high_mean": float(high.mean()) if len(high) else np.nan,
        "spread_high_minus_low": float(high.mean() - low.mean())
        if len(high) and len(low)
        else np.nan,
        "welch_t": float(wt),
        "welch_p": float(wp),
        "n_low": int(len(low)),
        "n_mid": int(len(mid)),
        "n_high": int(len(high)),
    }


# ---------------------------------------------------------------------------
# Record-high event test
# ---------------------------------------------------------------------------
def record_high_event(panel: pd.DataFrame) -> dict:
    """Forward return in fresh-record margin-debt years vs all other years.

    A 'record' year is one whose December margin debt exceeds all prior Decembers
    in the panel.  The newspaper claim: a record is a SELL signal, so record-years
    should have *lower* forward returns.

    Returns means, the difference, a Welch t-test, and the unconditional up-rate.
    """
    from scipy import stats as scipy_stats

    p = panel.dropna(subset=["fwd_ret"]).copy()
    running_max = p["margin_debt"].cummax()
    is_record = p["margin_debt"] >= running_max  # cumulative high incl. current
    rec = p[is_record]["fwd_ret"].to_numpy()
    non = p[~is_record]["fwd_ret"].to_numpy()

    if len(rec) >= 2 and len(non) >= 2:
        wt, wp = scipy_stats.ttest_ind(rec, non, equal_var=False)
    else:
        wt, wp = np.nan, np.nan

    return {
        "record_mean": float(rec.mean()) if len(rec) else np.nan,
        "nonrecord_mean": float(non.mean()) if len(non) else np.nan,
        "diff": float(rec.mean() - non.mean()) if len(rec) and len(non) else np.nan,
        "record_up_rate": float((rec > 0).mean()) if len(rec) else np.nan,
        "uncond_up_rate": float((p["fwd_ret"] > 0).mean()),
        "welch_t": float(wt),
        "welch_p": float(wp),
        "n_record": int(len(rec)),
        "n_nonrecord": int(len(non)),
    }


# ---------------------------------------------------------------------------
# Tradable contrarian rule vs buy-and-hold
# ---------------------------------------------------------------------------
def contrarian_timing_backtest(
    panel: pd.DataFrame,
    surge_z: float = 1.0,
    one_way_bps: float = 10.0,
) -> dict:
    """Sit out (cash, 0% return) the year after a margin surge; else hold the S&P.

    A 'surge' year is one whose standardized YoY margin growth exceeds ``surge_z``.
    The contrarian rule goes to cash for the following 12 months and otherwise
    earns the S&P forward (price-only) return.  Each in/out switch costs
    ``one_way_bps`` on NAV (round trips counted on entry and exit).

    Returns annualized geometric returns for the rule (gross & net) and for
    buy-and-hold, plus the number of years spent in cash.
    """
    p = panel.dropna(subset=["yoy", "fwd_ret"]).copy().reset_index(drop=True)
    z = standardized_yoy(p)
    in_market = (z <= surge_z).to_numpy()  # True = hold S&P this forward year

    fwd = p["fwd_ret"].to_numpy()
    strat_gross = np.where(in_market, fwd, 0.0)

    # Cost: a one-way cost whenever the position state changes between years.
    cost = one_way_bps * 1e-4
    prev_state = True  # assume start fully invested
    strat_net = strat_gross.copy()
    for i in range(len(strat_net)):
        if in_market[i] != prev_state:
            strat_net[i] -= cost
        prev_state = in_market[i]

    n = len(fwd)
    bah_ann = float(np.prod(1.0 + fwd) ** (1.0 / n) - 1.0)
    gross_ann = float(np.prod(1.0 + strat_gross) ** (1.0 / n) - 1.0)
    net_ann = float(np.prod(1.0 + np.clip(strat_net, -0.99, None)) ** (1.0 / n) - 1.0)

    return {
        "bah_ann": bah_ann,
        "strat_gross_ann": gross_ann,
        "strat_net_ann": net_ann,
        "years_in_cash": int((~in_market).sum()),
        "n_years": int(n),
        "shortfall_net_pp": float((net_ann - bah_ann) * 100.0),
    }


# ---------------------------------------------------------------------------
# Compact summary -- mirrors the R dict and docs/results.md
# ---------------------------------------------------------------------------
def summarize(panel: pd.DataFrame, one_way_bps: float = 10.0) -> dict:
    """One-stop summary dict for the real merged panel."""
    reg = predictive_regression(panel)
    ter = tercile_forward_returns(panel)
    rec = record_high_event(panel)
    bt = contrarian_timing_backtest(panel, surge_z=1.0, one_way_bps=one_way_bps)

    return {
        "n": int(len(panel.dropna(subset=["fwd_ret", "yoy"]))),
        "reg_beta": round(reg["beta"], 4),
        "reg_beta_t": round(reg["beta_t"], 3),
        "reg_r2": round(reg["r2"], 4),
        "ter_low_mean": round(ter["low_mean"], 4),
        "ter_high_mean": round(ter["high_mean"], 4),
        "ter_spread": round(ter["spread_high_minus_low"], 4),
        "ter_welch_t": round(ter["welch_t"], 3),
        "rec_mean": round(rec["record_mean"], 4),
        "nonrec_mean": round(rec["nonrecord_mean"], 4),
        "rec_diff": round(rec["diff"], 4),
        "rec_up_rate": round(rec["record_up_rate"], 4),
        "uncond_up_rate": round(rec["uncond_up_rate"], 4),
        "rec_welch_t": round(rec["welch_t"], 3),
        "bah_ann": round(bt["bah_ann"], 4),
        "strat_net_ann": round(bt["strat_net_ann"], 4),
        "years_in_cash": bt["years_in_cash"],
        "shortfall_net_pp": round(bt["shortfall_net_pp"], 2),
    }

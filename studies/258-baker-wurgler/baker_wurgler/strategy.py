"""Strategy and honest controls -- Study 258 (Baker-Wurgler sentiment index).

Baker & Wurgler (2006, 2007) is a **contrarian conditional predictor**: when the
sentiment index is HIGH at the end of month t, the aggregate market return in
month t+1 is LOW (and vice versa).  The effect is concentrated in hard-to-value /
hard-to-arbitrage names (small, young, non-dividend, distressed/extreme-growth),
which we proxy with small-minus-large (^RUT - ^GSPC).

We settle the question three honest ways:

1. **Regime sort** -- split months into HIGH / MID / LOW sentiment tertiles (by
   prior-month sentiment) and compare the *next-month* return in each regime.
   The BW prediction is LOW-regime return > HIGH-regime return (contrarian).

2. **Predictive regression** -- regress next-month return on lagged sentiment with
   a Newey-West HAC t-stat.  A negative slope that clears |t| >= 2 is the bar.

3. **Long-low/short-high timing rule** -- a tradable overlay: hold the market
   (long) when prior sentiment is in the low tertile, go flat (or short) when it
   is in the high tertile.  Compared net of a one-way cost x NAV on each switch,
   against unconditional buy-and-hold (the positive-control benchmark).

**No look-ahead**: sentiment is read at month-end t (a published, lagged figure)
and predicts month t+1.  This is already a one-month execution lag.

**Price-only / total-return**: ^GSPC returns here are price-only (no dividends);
this understates levels but the cross-regime *spread* is roughly dividend-neutral.
Labeled on the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Summary statistics (HAC t-stat)
# ---------------------------------------------------------------------------
def summarize(series: pd.Series) -> dict:
    """Headline stats for a monthly return/spread series, with a Newey-West HAC t-stat."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")}

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
        "mean": float(mu),
        "vol": float(std),
        "sharpe": float(sr),
        "tstat": float(tstat),
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": float(max_dd),
        "n": n,
    }


def annualise_monthly(stats: dict, periods: int = 12) -> dict:
    """Convert monthly stats to annualised equivalents."""
    out = dict(stats)
    if "mean" in out and np.isfinite(out["mean"]):
        out["mean_ann"] = float(out["mean"] * periods)
    if "vol" in out and np.isfinite(out["vol"]):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if "mean_ann" in out and "vol_ann" in out and out["vol_ann"] > 0:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out


# ---------------------------------------------------------------------------
# Regime sort
# ---------------------------------------------------------------------------
def regime_sort(
    df: pd.DataFrame,
    ret_col: str = "ret",
    n_bins: int = 3,
) -> pd.DataFrame:
    """Split months by prior-month sentiment into ``n_bins`` regimes and tabulate next-month return.

    ``df`` must have columns ``bw_sent`` (the lagged signal) and ``ret_col``.
    Returns a DataFrame indexed by regime label (e.g. 'low','mid','high' for 3)
    with columns: mean (monthly), mean_ann, vol_ann, tstat (HAC), hit_rate, n.
    """
    d = df.dropna(subset=["bw_sent", ret_col]).copy()
    labels = _bin_labels(n_bins)
    d["regime"] = pd.qcut(d["bw_sent"], n_bins, labels=labels, duplicates="drop")

    rows = []
    for lab in labels:
        sub = d.loc[d["regime"] == lab, ret_col]
        if len(sub) < 3:
            continue
        s = summarize(sub)
        a = annualise_monthly(s)
        rows.append({
            "regime": lab,
            "mean": s["mean"],
            "mean_ann": a.get("mean_ann", np.nan),
            "vol_ann": a.get("vol_ann", np.nan),
            "tstat": s["tstat"],
            "hit_rate": s["hit_rate"],
            "n": s["n"],
        })
    out = pd.DataFrame(rows).set_index("regime")
    return out


def low_minus_high(
    df: pd.DataFrame,
    ret_col: str = "ret",
    n_bins: int = 3,
) -> pd.Series:
    """Monthly spread series: (return in low-sentiment months) - (return in high months).

    Because high/low regimes occur in different months, this is *not* a paired
    time series; instead we return the demeaned contribution: for each month, the
    next-month return signed by regime (+ if low-sentiment regime, - if high), 0
    otherwise.  Summarising this series with ``summarize`` gives the HAC t-stat of
    the low-minus-high contrarian spread under a balanced-regime assumption.
    """
    d = df.dropna(subset=["bw_sent", ret_col]).copy()
    labels = _bin_labels(n_bins)
    d["regime"] = pd.qcut(d["bw_sent"], n_bins, labels=labels, duplicates="drop")
    low_lab, high_lab = labels[0], labels[-1]
    low_ret = d.loc[d["regime"] == low_lab, ret_col]
    high_ret = d.loc[d["regime"] == high_lab, ret_col]
    # Stack both legs: low leg positive, high leg negated -> mean = low_mean - high_mean over the pooled set
    spread = pd.concat([low_ret, -high_ret])
    spread.name = "low_minus_high"
    return spread.sort_index()


def _bin_labels(n_bins: int) -> list[str]:
    if n_bins == 3:
        return ["low", "mid", "high"]
    if n_bins == 2:
        return ["low", "high"]
    return [f"q{i+1}" for i in range(n_bins)]


# ---------------------------------------------------------------------------
# Predictive regression (slope with HAC t)
# ---------------------------------------------------------------------------
def predictive_regression(
    df: pd.DataFrame,
    ret_col: str = "ret",
) -> dict:
    """OLS of next-month return on lagged sentiment, with Newey-West HAC t on the slope.

    Returns dict: alpha, beta (slope), beta_t (HAC), r2, n.
    The Baker-Wurgler prediction is a *negative* slope (high sentiment -> low return).
    """
    d = df.dropna(subset=["bw_sent", ret_col]).copy()
    x = d["bw_sent"].to_numpy(dtype=float)
    y = d[ret_col].to_numpy(dtype=float)
    n = len(x)
    if n < 5:
        return {"alpha": np.nan, "beta": np.nan, "beta_t": np.nan, "r2": np.nan, "n": n}

    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    resid = y - X @ coef

    # Newey-West HAC covariance
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = X * resid[:, None]
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    beta = float(coef[1])
    se_beta = float(np.sqrt(cov[1, 1]))
    beta_t = beta / se_beta if se_beta > 0 else float("nan")

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "alpha": float(coef[0]),
        "beta": beta,
        "beta_t": float(beta_t),
        "r2": float(r2),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Tradable timing overlay
# ---------------------------------------------------------------------------
def timing_overlay(
    df: pd.DataFrame,
    ret_col: str = "ret",
    n_bins: int = 3,
    short_high: bool = False,
    one_way_bps: float = 5.0,
) -> pd.DataFrame:
    """Long-when-low-sentiment timing rule vs unconditional buy-and-hold.

    Rule: be fully long the market when prior-month sentiment is in the LOW
    tertile, flat (cash) in MID, and flat (``short_high=False``) or short
    (``short_high=True``) in the HIGH tertile.  Switching the position incurs a
    one-way cost ``one_way_bps`` applied to NAV on the traded fraction.

    Returns a DataFrame indexed by month with columns:
      - ``ret``       : the underlying market return that month
      - ``position``  : -1 / 0 / +1
      - ``gross``     : strategy gross return
      - ``cost``      : transaction-cost drag this month
      - ``net``       : gross - cost
      - ``buyhold``   : unconditional long return (the benchmark)
    """
    d = df.dropna(subset=["bw_sent", ret_col]).copy()
    labels = _bin_labels(n_bins)
    d["regime"] = pd.qcut(d["bw_sent"], n_bins, labels=labels, duplicates="drop")
    low_lab, high_lab = labels[0], labels[-1]

    pos_map = {low_lab: 1.0, high_lab: (-1.0 if short_high else 0.0)}
    position = d["regime"].map(lambda r: pos_map.get(r, 0.0)).astype(float)

    prev_pos = position.shift(1).fillna(0.0)
    traded = (position - prev_pos).abs()
    cost = traded * one_way_bps * 1e-4

    gross = position * d[ret_col]
    net = gross - cost

    out = pd.DataFrame({
        "ret": d[ret_col],
        "position": position,
        "gross": gross,
        "cost": cost,
        "net": net,
        "buyhold": d[ret_col],
    })
    out.index.name = "date"
    return out


def turnover_count(overlay: pd.DataFrame) -> dict:
    """How many position switches and the implied annual turnover from an overlay frame."""
    pos = overlay["position"]
    switches = int((pos != pos.shift(1)).sum())
    n = len(pos)
    return {
        "switches": switches,
        "n_months": n,
        "switches_per_year": float(switches / max(1, n) * 12.0),
        "total_cost_bps": float(overlay["cost"].sum() * 1e4),
    }

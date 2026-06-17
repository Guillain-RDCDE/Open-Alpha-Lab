"""Strategy and honest controls -- Study 263 (Insider-Buying).

The contrarian aggregate-insider story: when the aggregate buy/sell ratio is HIGH
(insiders buying), go long the S&P; when it is LOW (insiders selling), go flat (or
short). We test this market-timing overlay against the brutally hard benchmark --
buy-and-hold -- and decompose the result into:

1. **Predictive regression** -- next-month return on the (demeaned, standardised)
   buy/sell ratio, with a Newey-West HAC t-stat. This is the cleanest test of "does
   the ratio forecast returns?"
2. **Timing strategy vs buy-and-hold** -- a long/flat overlay that is long when the
   ratio is above its trailing median (an honest, look-ahead-free threshold) and
   flat otherwise, net of costs, vs always-long.
3. **Tercile sort** -- next-month return conditioned on low/mid/high ratio terciles,
   the non-parametric view.
4. **Positive control** -- the synthetic tape with a planted ``beta`` confirms the
   regression t-stat fires when an effect exists.

**No look-ahead.** The signal at month t is the ratio observed at the end of month t
(Form-4 filings are public within 2 business days, so end-of-month data is available
by the start of month t+1). The trailing-median threshold uses only past ratios. The
return it predicts is the t->t+1 return. One-month execution lag, applied once.

**Curated-proxy cap.** The real tape uses a curated buy/sell ratio (see ``data.py``),
so the Signal axis is capped at WEAK regardless of the t-stat -- a reconstructed
series cannot earn REAL.

**Costs.** The long/flat overlay pays a one-way cost x NAV every time it switches
between long and flat. A short variant would additionally pay borrow; we report the
long/flat version as the realistic case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Summary statistics with Newey-West HAC t-stat
# ---------------------------------------------------------------------------
def summarize(series: pd.Series, periods: int = 12) -> dict:
    """Headline statistics for a monthly return series with a HAC (Newey-West) t-stat."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: float("nan") for k in
                ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}

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
        "mean": mu,
        "vol": std,
        "sharpe": sr,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": max_dd,
        "mean_ann": mu * periods,
        "vol_ann": std * np.sqrt(periods),
        "sharpe_ann": sr * np.sqrt(periods) if np.isfinite(sr) else float("nan"),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Predictive regression: fwd_ret ~ standardised ratio
# ---------------------------------------------------------------------------
def predictive_regression(df: pd.DataFrame) -> dict:
    """Regress next-month return on the standardised buy/sell ratio (HAC t-stat).

    ``df`` must have columns ``bs_ratio`` (signal at t) and ``fwd_ret`` (t->t+1
    return). The ratio is z-scored over the full sample (in-sample standardisation
    is fine here -- it does not create look-ahead in the *coefficient sign/size*,
    which is what we test; the strategy in ``timing_overlay`` uses a strictly
    trailing threshold instead).

    Returns a dict with ``beta`` (slope, in return per 1-sigma of ratio), ``alpha``,
    ``tstat`` (HAC t on the slope), ``r2``, ``n``.
    """
    d = df.dropna(subset=["bs_ratio", "fwd_ret"]).copy()
    x = d["bs_ratio"].to_numpy(dtype=float)
    y = d["fwd_ret"].to_numpy(dtype=float)
    n = len(y)
    if n < 5:
        return {"beta": float("nan"), "alpha": float("nan"), "tstat": float("nan"),
                "r2": float("nan"), "n": n}

    xm = (x - x.mean())
    xs = xm.std(ddof=0)
    z = xm / xs if xs > 0 else xm

    X = np.column_stack([np.ones(n), z])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef

    # Newey-West HAC covariance of the OLS coefficients.
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        Xk = X[k:] * resid[k:, None]
        Xk0 = X[:-k] * resid[:-k, None]
        gamma = Xk.T @ Xk0
        S += w * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_beta = float(np.sqrt(max(cov[1, 1], 0.0)))
    tstat = float(coef[1] / se_beta) if se_beta > 0 else float("nan")

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "beta": float(coef[1]),   # return per 1-sigma increase in ratio
        "alpha": float(coef[0]),
        "tstat": tstat,
        "r2": r2,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Tercile sort
# ---------------------------------------------------------------------------
def tercile_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Next-month return conditioned on low/mid/high buy/sell-ratio terciles.

    Returns a DataFrame indexed by ['low', 'mid', 'high'] with columns
    ``mean_ann`` (annualised mean fwd return), ``tstat`` (HAC), ``n``.
    The high-minus-low row is appended as 'high_minus_low'.
    """
    d = df.dropna(subset=["bs_ratio", "fwd_ret"]).copy()
    q1, q2 = d["bs_ratio"].quantile([1 / 3, 2 / 3])
    d["bucket"] = np.where(d["bs_ratio"] <= q1, "low",
                           np.where(d["bs_ratio"] >= q2, "high", "mid"))

    rows = {}
    for b in ("low", "mid", "high"):
        s = summarize(d.loc[d["bucket"] == b, "fwd_ret"])
        rows[b] = {"mean_ann": s["mean_ann"], "tstat": s["tstat"], "n": s["n"]}

    hi = d.loc[d["bucket"] == "high", "fwd_ret"].mean()
    lo = d.loc[d["bucket"] == "low", "fwd_ret"].mean()
    rows["high_minus_low"] = {
        "mean_ann": float((hi - lo) * 12),
        "tstat": float("nan"),
        "n": int((d["bucket"] != "mid").sum()),
    }
    return pd.DataFrame(rows).T[["mean_ann", "tstat", "n"]]


# ---------------------------------------------------------------------------
# Timing overlay: long when ratio above trailing median, else flat
# ---------------------------------------------------------------------------
def timing_overlay(
    df: pd.DataFrame,
    lookback: int = 24,
    one_way_bps: float = 5.0,
) -> pd.DataFrame:
    """Long/flat S&P overlay driven by a strictly-trailing ratio threshold.

    Position at month t = LONG if the ratio at t is above its trailing median over
    the prior ``lookback`` months (look-ahead-free), else FLAT. The position earns
    ``fwd_ret`` (the t->t+1 return). Switching long<->flat costs ``one_way_bps`` x NAV.

    Returns a DataFrame with columns:
      ``pos``      : 1 (long) or 0 (flat) for month t,
      ``gross``    : pos * fwd_ret,
      ``cost``     : one-way cost x NAV on switches,
      ``net``      : gross - cost,
      ``bah``      : fwd_ret (always-long benchmark).
    """
    d = df.dropna(subset=["bs_ratio", "fwd_ret"]).copy()
    ratio = d["bs_ratio"]
    # Trailing median EXCLUDING the current month (shift(1)) -> strictly past data.
    trail_med = ratio.shift(1).rolling(lookback, min_periods=max(6, lookback // 2)).median()
    pos = (ratio > trail_med).astype(float)
    pos[trail_med.isna()] = 1.0  # before threshold is defined, default long (benchmark-neutral)

    gross = pos * d["fwd_ret"]
    switch = pos.diff().abs().fillna(0.0)
    cost = switch * one_way_bps * 1e-4
    net = gross - cost

    out = pd.DataFrame({
        "pos": pos,
        "gross": gross,
        "cost": cost,
        "net": net,
        "bah": d["fwd_ret"],
    })
    return out


def overlay_turnover(overlay: pd.DataFrame) -> float:
    """Average monthly switching turnover of the long/flat overlay (fraction of months)."""
    return float(overlay["pos"].diff().abs().fillna(0.0).mean())

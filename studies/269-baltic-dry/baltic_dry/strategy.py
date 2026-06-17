"""Strategy and honest controls -- Study 269 (Baltic-Dry).

The folk claim: a rising Baltic Dry Index (BDI) signals strengthening real
global demand and therefore foreshadows a rising stock market.  We test it three
ways, all with a one-month execution lag (signal at month-end *t*, return earned
in *t+1*):

1. **Predictive regression** -- regress next month's S&P log-return on this
   month's BDI signal (a smoothed BDI momentum).  The HAC *t* on the slope is the
   inference bar; a real lead-lag needs ``|t| >= 2``.

2. **Timing strategy** -- go long the S&P when the BDI signal is positive
   (BDI rising), else hold cash (0%); compare gross/net to buy-and-hold.  This is
   the tradable read of the claim.  Costs are charged one-way on each switch.

3. **Sign / hit-rate** -- does the sign of the BDI signal predict the sign of
   next month's equity return better than the unconditional base rate?

**No look-ahead**: the signal uses only BDI data available at month-end *t*; the
return it predicts is earned strictly afterwards (during *t+1*).  The BDI is
published daily with at most a one-day lag, so a month-end reading is actionable.

**Curated-series caveat**: the real BDI tape here is a curated monthly
reconstruction (see ``data.py``).  Its turning points are chosen with hindsight
knowledge of history; an in-sample fit can be flattered.  Named on the Signal
axis -- real numbers are indicative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def bdi_signal(
    df: pd.DataFrame,
    lookback: int = 3,
) -> pd.Series:
    """BDI momentum signal at each month-end: smoothed log-change of the index.

    The signal at month-end *t* is the ``lookback``-month log change of the BDI::

        signal[t] = log(bdi[t]) - log(bdi[t - lookback])

    A positive signal means freight rates have risen over the window (the folk
    "demand is strengthening" read).  Uses only data through *t*; the value it
    is paired with downstream is the *next* month's S&P return.

    Returns a Series aligned to ``df.index`` (NaN for the first ``lookback``
    rows, the burn-in).
    """
    logbdi = np.log(df["bdi"].astype(float))
    sig = logbdi - logbdi.shift(lookback)
    return sig.rename("bdi_signal")


def forward_sp_return(df: pd.DataFrame) -> pd.Series:
    """Next-month S&P simple return (the thing the signal is asked to predict).

    ``fwd[t]`` is the return earned *during* month *t+1*: the position is formed
    at the close of month *t* and held through *t+1* (one-month execution lag).
    """
    ret = df["sp500"].astype(float).pct_change()
    fwd = ret.shift(-1)
    return fwd.rename("sp_fwd")


# ---------------------------------------------------------------------------
# Predictive regression
# ---------------------------------------------------------------------------
def predictive_regression(
    signal: pd.Series,
    fwd_ret: pd.Series,
) -> dict:
    """OLS of next-month S&P return on the BDI signal, with a HAC t-stat.

    Returns a dict with ``alpha``, ``beta`` (slope on the signal), HAC ``tstat``
    on the slope, ``r2``, and ``n``.  The HAC (Newey-West) standard error
    accounts for residual autocorrelation, which is the honest inference bar.
    """
    s = pd.concat([signal.rename("x"), fwd_ret.rename("y")], axis=1).dropna()
    n = int(len(s))
    if n < 12:
        return {"alpha": np.nan, "beta": np.nan, "tstat": np.nan, "r2": np.nan, "n": n}

    x = s["x"].to_numpy()
    y = s["y"].to_numpy()
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    resid = y - X @ coef

    # Newey-West HAC covariance
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        Xe = X * resid[:, None]
        Gamma = Xe[k:].T @ Xe[:-k]
        S += w * (Gamma + Gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_beta = float(np.sqrt(max(cov[1, 1], 0.0)))
    tstat = float(coef[1] / se_beta) if se_beta > 0 else float("nan")

    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "alpha": float(coef[0]),
        "beta": float(coef[1]),
        "tstat": tstat,
        "r2": float(r2),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Timing strategy
# ---------------------------------------------------------------------------
def timing_returns(
    signal: pd.Series,
    fwd_ret: pd.Series,
    one_way_bps: float = 5.0,
) -> pd.DataFrame:
    """Long-when-BDI-rising timing overlay vs buy-and-hold.

    Position at month-end *t*: 1.0 (fully long the S&P next month) if
    ``signal[t] > 0``, else 0.0 (cash).  The realised return for month *t+1* is
    ``position[t] * fwd_ret[t]``.  Costs are charged one-way (``one_way_bps``)
    each time the position changes, applied to the traded notional.

    Returns a DataFrame indexed by *t* with columns:
    - ``position`` : 0/1 desired exposure for the coming month
    - ``buy_hold`` : the S&P forward return (the benchmark leg)
    - ``timing_gross`` : position * forward return (no costs)
    - ``cost`` : one-way cost charged on the switch this month
    - ``timing_net`` : timing_gross - cost
    """
    s = pd.concat([signal.rename("sig"), fwd_ret.rename("fwd")], axis=1).dropna()
    pos = (s["sig"] > 0).astype(float)

    prev = pos.shift(1).fillna(0.0)
    turn = (pos - prev).abs()           # 0 or 1 on a switch
    cost = turn * (one_way_bps * 1e-4)

    gross = pos * s["fwd"]
    net = gross - cost

    out = pd.DataFrame({
        "position": pos,
        "buy_hold": s["fwd"],
        "timing_gross": gross,
        "cost": cost,
        "timing_net": net,
    })
    return out


# ---------------------------------------------------------------------------
# Summary statistics (HAC t-stat)
# ---------------------------------------------------------------------------
def summarize(series: pd.Series) -> dict:
    """Headline statistics for a monthly return series (Newey-West HAC t-stat)."""
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


def sign_hit_rate(signal: pd.Series, fwd_ret: pd.Series) -> dict:
    """Does the sign of the BDI signal predict the sign of next month's return?

    Returns the conditional hit rate (fraction of months where
    ``sign(signal) == sign(fwd_ret)``), the unconditional base rate (fraction of
    months the market rose), and the count.
    """
    s = pd.concat([signal.rename("x"), fwd_ret.rename("y")], axis=1).dropna()
    if len(s) == 0:
        return {"hit": np.nan, "base_rate": np.nan, "n": 0}
    agree = (np.sign(s["x"]) == np.sign(s["y"]))
    return {
        "hit": float(agree.mean()),
        "base_rate": float((s["y"] > 0).mean()),
        "n": int(len(s)),
    }

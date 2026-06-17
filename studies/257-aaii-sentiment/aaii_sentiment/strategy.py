"""Strategy and honest controls -- Study 257 (AAII-Sentiment).

The contrarian AAII rule: when the bull-bear spread is **low** (panic), lean
*into* the market; when it is **high** (euphoria), lean *out*.  We implement and
stress-test three flavours of the timing rule on monthly S&P 500 price returns:

1. **Regime sort** -- split months into low / mid / high sentiment terciles by
   the *prior* month-end spread; compare the next-month mean return across
   regimes (the cleanest test of "does sentiment predict?").

2. **Contrarian timing overlay** -- a long/flat (or long/short) rule that goes
   long when the spread is below a threshold and flat (or short) when above.
   Compared head-to-head against **buy-and-hold** on the same price index.

3. **Predictive regression** -- regress next-month return on the (standardised)
   prior spread; report the slope, its HAC *t*-stat, and the R-squared.  A real
   contrarian effect needs a *negative* slope with HAC ``|t| >= 2``.

Honesty rails baked in:

- **One-month execution lag**: the signal at month-end *t* trades month *t+1*.
- **One-way costs x NAV**: the overlay pays ``one_way_bps`` each time it changes
  state; a short leg additionally pays a borrow haircut.
- **Gross vs net** and **price-only** are labelled (the timing rule and
  buy-and-hold share the same price-only index, so dividends cancel in the
  *relative* spread but the absolute numbers are price-return).
- **Vintage / survivorship** on sentiment: the curated monthly snapshot is not
  point-in-time; any edge is an upper bound (named on the Signal axis).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Summary statistics (HAC / Newey-West t-stat)
# ---------------------------------------------------------------------------
def summarize(series: pd.Series) -> dict:
    """Headline statistics for a monthly return series with a Newey-West HAC t-stat."""
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")} | {"n": n}

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
        "n": n,
    }


def annualise_monthly(stats: dict, periods: int = 12) -> dict:
    """Convert monthly stats to annualised equivalents."""
    out = dict(stats)
    if "mean" in out and np.isfinite(out.get("mean", np.nan)):
        out["mean_ann"] = float(out["mean"] * periods)
    if "vol" in out and np.isfinite(out.get("vol", np.nan)):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if out.get("vol_ann", 0.0) > 0:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out


# ---------------------------------------------------------------------------
# 1. Regime sort: next-month return by prior-spread tercile
# ---------------------------------------------------------------------------
def regime_returns(panel: pd.DataFrame, q: float = 1.0 / 3.0) -> pd.DataFrame:
    """Bucket months by the *prior* (already-observed) bull-bear spread tercile.

    ``panel`` must have columns ``spread`` (observed at month-end t) and ``ret``
    (return earned during month t+1).  No look-ahead: the tercile thresholds use
    only the spread column, and each row's ``ret`` is already the forward return.

    Returns a per-row frame with a ``regime`` label in {"low", "mid", "high"}
    plus the forward ``ret``; ``low`` = most bearish/panicked (the contrarian
    *buy* zone), ``high`` = most bullish/euphoric (the contrarian *sell* zone).
    """
    df = panel.dropna(subset=["spread", "ret"]).copy()
    lo = df["spread"].quantile(q)
    hi = df["spread"].quantile(1.0 - q)
    regime = pd.Series("mid", index=df.index)
    regime[df["spread"] <= lo] = "low"
    regime[df["spread"] >= hi] = "high"
    df["regime"] = regime
    return df[["spread", "ret", "regime"]]


def regime_summary(panel: pd.DataFrame, q: float = 1.0 / 3.0) -> pd.DataFrame:
    """Mean forward return and HAC t-stat per sentiment regime.

    A contrarian signal predicts: low-sentiment months earn MORE next month than
    high-sentiment months (the ``low - high`` spread is positive).
    """
    reg = regime_returns(panel, q=q)
    rows = []
    for name in ("low", "mid", "high"):
        sub = reg.loc[reg["regime"] == name, "ret"]
        s = summarize(sub)
        rows.append({
            "regime": name,
            "n": s["n"],
            "mean_mo": s.get("mean", np.nan),
            "mean_ann": s.get("mean", np.nan) * 12 if np.isfinite(s.get("mean", np.nan)) else np.nan,
            "tstat": s.get("tstat", np.nan),
        })
    return pd.DataFrame(rows).set_index("regime")


def regime_spread(panel: pd.DataFrame, q: float = 1.0 / 3.0) -> pd.Series:
    """The (low-sentiment minus high-sentiment) forward-return series.

    Each month is signed +ret if it falls in the low-sentiment regime, -ret if
    high, 0 if mid -- i.e. the realised return of a self-financing rule that is
    long the panic regime and short the euphoria regime.  The HAC t-stat of this
    series is the headline contrarian test.
    """
    reg = regime_returns(panel, q=q)
    out = pd.Series(0.0, index=reg.index, name="contrarian")
    out[reg["regime"] == "low"] = reg.loc[reg["regime"] == "low", "ret"]
    out[reg["regime"] == "high"] = -reg.loc[reg["regime"] == "high", "ret"]
    return out


# ---------------------------------------------------------------------------
# 2. Contrarian timing overlay vs buy-and-hold
# ---------------------------------------------------------------------------
def timing_overlay(
    panel: pd.DataFrame,
    buy_below: float | None = None,
    sell_above: float | None = None,
    allow_short: bool = False,
    one_way_bps: float = 5.0,
    borrow_bps_ann: float = 100.0,
) -> pd.DataFrame:
    """Long/flat (or long/short) contrarian overlay vs buy-and-hold, net of costs.

    Position for month t+1 is decided by the spread observed at month-end t:
      - spread <= ``buy_below``  -> long  (+1)
      - spread >= ``sell_above`` -> flat (0) or short (-1) if ``allow_short``
      - otherwise                 -> long (+1)   (default-long contrarian)

    Defaults for the thresholds are the in-sample 33rd/67th spread percentiles
    when ``None``.

    Costs: ``one_way_bps`` x NAV each time the position changes; a short month
    additionally pays ``borrow_bps_ann/12`` borrow.  Returns a frame with columns
    ``pos``, ``ret`` (market price return), ``gross``, ``cost``, ``net`` and the
    buy-and-hold ``bh`` return for comparison.
    """
    df = panel.dropna(subset=["spread", "ret"]).copy()
    if buy_below is None:
        buy_below = df["spread"].quantile(1.0 / 3.0)
    if sell_above is None:
        sell_above = df["spread"].quantile(2.0 / 3.0)

    pos = pd.Series(1.0, index=df.index)
    pos[df["spread"] >= sell_above] = -1.0 if allow_short else 0.0
    pos[df["spread"] <= buy_below] = 1.0

    gross = pos * df["ret"]
    turn = pos.diff().abs().fillna(pos.abs())
    cost = turn * one_way_bps * 1e-4
    borrow = (pos < 0).astype(float) * (borrow_bps_ann / 12.0) * 1e-4
    net = gross - cost - borrow

    out = pd.DataFrame({
        "pos": pos,
        "ret": df["ret"],
        "gross": gross,
        "cost": cost + borrow,
        "net": net,
        "bh": df["ret"],
    })
    return out


# ---------------------------------------------------------------------------
# 3. Predictive regression: next-month return on prior spread
# ---------------------------------------------------------------------------
def predictive_regression(panel: pd.DataFrame) -> dict:
    """OLS of forward return on the standardised prior spread, HAC t on the slope.

    Returns a dict with ``alpha``, ``beta`` (slope on the standardised spread,
    in return units per +1 sd of spread), ``tstat`` (Newey-West HAC t on beta),
    ``r2``, and ``n``.  A contrarian effect => ``beta < 0`` with ``|tstat| >= 2``.
    """
    df = panel.dropna(subset=["spread", "ret"]).copy()
    x = df["spread"].to_numpy(dtype=float)
    y = df["ret"].to_numpy(dtype=float)
    n = len(x)
    if n < 5:
        return {"alpha": np.nan, "beta": np.nan, "tstat": np.nan, "r2": np.nan, "n": n}

    xm, xs = x.mean(), x.std(ddof=0)
    xz = (x - xm) / xs if xs > 0 else x - xm

    X = np.column_stack([np.ones(n), xz])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - X @ coef

    # HAC (Newey-West) covariance of the slope.
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        u = X * resid[:, None]
        gamma = u[k:].T @ u[:-k]
        S += w * (gamma + gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_beta = float(np.sqrt(cov[1, 1])) if cov[1, 1] > 0 else float("nan")
    tstat = float(beta / se_beta) if se_beta and se_beta > 0 else float("nan")

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"alpha": alpha, "beta": beta, "tstat": tstat, "r2": r2, "n": n}

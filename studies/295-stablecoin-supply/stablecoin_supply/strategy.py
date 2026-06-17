"""Strategy and honest controls -- Study 295 (Stablecoin-Supply).

The "dry powder" thesis: stablecoins are cash parked on-chain, waiting to be
deployed.  When the total stablecoin supply *grows*, new capital is entering the
crypto ecosystem; the claim is that this growth *leads* the next leg up in BTC.

We implement supply growth as a forward-return timing signal:

1. At month-end t, observe supply growth over month t (or a trailing window).
2. If growth exceeds a threshold (the dry-powder signal fires), go LONG BTC for
   month t+1; otherwise stay flat (in cash).  No look-ahead: only data through t
   is used to size the month t+1 position.
3. Compare the timed series against simple buy-and-hold BTC.

Honest comparisons settle the matter:

- **Timed vs buy-and-hold** -- does conditioning on supply growth beat just
  holding BTC?  (Risk-adjusted and net of costs.)
- **Predictive regression** -- regress month-t+1 BTC return on month-t
  standardised supply growth; report the HAC t-stat on the slope.  This is the
  cleanest read on whether supply growth carries *forward* information.
- **Contemporaneous vs lagged** -- the reflexivity check.  Supply and price both
  surge together, so the *contemporaneous* correlation is large and misleading.
  Only the *lagged* (one-month-ahead) relationship is tradable.

**Reflexivity / coincidence (named on the Signal axis).**  Supply growth is
largely a coincident indicator of crypto risk appetite.  A large contemporaneous
correlation that collapses under a one-month execution lag is the signature of a
non-tradable, reflexive series -- not a leading signal.

**Costs.**  Each switch (flat<->long) is a one-way trade; we charge a one-way
cost on the traded notional (NAV) each time the position changes.  BTC spot is
liquid but not free; we sweep 10/25/50 bps one-way.

**Price-only / total-return.**  BTC spot has no dividend, so price return equals
total return.  No survivorship issue for a single-asset BTC tape, but the curated
supply series is itself an approximation -- noted on the Signal axis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def supply_growth(panel: pd.DataFrame, window: int = 1) -> pd.Series:
    """Trailing supply growth over ``window`` months, observed at month-end t.

    ``window=1`` is the one-month percent change in supply; larger windows give
    a smoother (trailing) growth rate.  This is the raw signal input -- it uses
    only data through month t, so it is look-ahead free for predicting t+1.
    """
    supply = panel["supply_bn"].astype(float)
    g = supply.pct_change(periods=window)
    return g.rename("supply_growth")


def timing_signal(
    panel: pd.DataFrame,
    window: int = 1,
    threshold: float = 0.0,
) -> pd.Series:
    """Binary long/flat timing weight for month t+1, decided at month-end t.

    Returns a Series of weights in {0.0, 1.0} indexed by month-end t: 1.0 means
    "hold BTC during month t+1" (the dry-powder signal fired because supply
    growth over month t exceeded ``threshold``), 0.0 means "stay in cash".

    The weight at index t applies to the *forward* return earned in month t+1;
    callers must shift accordingly (see ``timed_returns``).  No look-ahead: the
    signal at t uses supply growth realised by the close of month t only.
    """
    g = supply_growth(panel, window=window)
    w = (g > threshold).astype(float)
    # Burn-in: the first ``window`` rows have no growth -> stay flat
    w.iloc[:window] = 0.0
    return w.rename("weight")


# ---------------------------------------------------------------------------
# Backtest: timed vs buy-and-hold, with a strict one-month execution lag
# ---------------------------------------------------------------------------
def timed_returns(
    panel: pd.DataFrame,
    window: int = 1,
    threshold: float = 0.0,
    one_way_bps: float = 0.0,
) -> pd.DataFrame:
    """Monthly returns of the supply-timed strategy vs buy-and-hold BTC.

    The weight decided at month-end t (``timing_signal``) is applied to the BTC
    return earned in month t+1 -- a strict one-month execution lag.  Cost is
    charged on every change in weight (``|w_t - w_{t-1}|``) at ``one_way_bps`` on
    the traded NAV.

    Returns a DataFrame indexed by the month the return is *earned*, with
    columns:
      - ``btc``    : buy-and-hold BTC monthly return
      - ``timed``  : supply-timed monthly return (gross of cost)
      - ``timed_net``: supply-timed monthly return net of switching cost
      - ``weight`` : the weight actually applied that month (lagged signal)
    """
    btc = panel["btc_ret"].astype(float)
    w_signal = timing_signal(panel, window=window, threshold=threshold)

    # Apply the month-end t weight to the month t+1 return -> shift weight by 1
    w_applied = w_signal.shift(1).fillna(0.0)

    timed_gross = (w_applied * btc).rename("timed")

    # Switching cost: charged when the applied weight changes month to month
    dw = w_applied.diff().abs().fillna(w_applied.abs())
    cost = dw * (one_way_bps * 1e-4)
    timed_net = (timed_gross - cost).rename("timed_net")

    out = pd.DataFrame(
        {
            "btc": btc,
            "timed": timed_gross,
            "timed_net": timed_net,
            "weight": w_applied.rename("weight"),
        }
    )
    # Drop the first row (no forward return yet / pure burn-in)
    return out.dropna(subset=["btc"])


def turnover(panel: pd.DataFrame, window: int = 1, threshold: float = 0.0) -> float:
    """Average monthly turnover (fraction of months the position switches).

    A switch is any month where the applied weight differs from the prior month.
    Reported as a fraction in [0, 1]; with a binary long/flat signal it is also
    the per-month round-trip rate.
    """
    w_signal = timing_signal(panel, window=window, threshold=threshold)
    w_applied = w_signal.shift(1).fillna(0.0)
    dw = w_applied.diff().abs().fillna(0.0)
    return float(dw.mean())


# ---------------------------------------------------------------------------
# Predictive regression -- the cleanest forward-information read
# ---------------------------------------------------------------------------
def predictive_regression(
    panel: pd.DataFrame,
    window: int = 1,
    lag: int = 1,
) -> dict:
    """Regress month t+lag BTC return on standardised month-t supply growth.

    ``lag=1`` is the tradable, look-ahead-free specification (signal at t
    predicts the return earned in t+1).  ``lag=0`` is the *contemporaneous*
    regression -- the reflexivity check, NOT tradable.

    Returns a dict with the slope ``beta``, its HAC ``tstat`` (Newey-West), the
    ``r2``, the correlation ``corr``, and the sample size ``n``.
    """
    g = supply_growth(panel, window=window)
    z = (g - g.mean()) / (g.std(ddof=0) + 1e-12)
    y = panel["btc_ret"].astype(float)

    if lag != 0:
        z = z.shift(lag)

    df = pd.concat([z.rename("z"), y.rename("y")], axis=1).dropna()
    if len(df) < 5:
        return {"beta": np.nan, "tstat": np.nan, "r2": np.nan, "corr": np.nan, "n": len(df)}

    x = df["z"].to_numpy()
    yv = df["y"].to_numpy()
    n = len(x)

    # OLS with intercept
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ coef
    beta = float(coef[1])

    # Newey-West HAC standard error on beta
    XtX_inv = np.linalg.inv(X.T @ X)
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        u = X * resid[:, None]
        Gamma = u[k:].T @ u[:-k]
        S += w * (Gamma + Gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_beta = float(np.sqrt(max(cov[1, 1], 0.0)))
    tstat = beta / se_beta if se_beta > 0 else float("nan")

    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    ss_res = float((resid ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(x, yv)[0, 1])

    return {"beta": beta, "tstat": float(tstat), "r2": float(r2), "corr": corr, "n": n}


# ---------------------------------------------------------------------------
# Summary statistics
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
